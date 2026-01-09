"""
Тут смотрим, как меняются:
- recall@K по критичным дефектам,
- стоимость политики (проверки + пропуски),
если по-разному определить что считается критичным.
Плюс диагностика: совпадают ли top-K у разных моделей

"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


DATA_PATH = r"faults.csv"

LABEL_COLS = [
    "Pastry", "Z_Scratch", "K_Scatch",
    "Stains", "Dirtiness", "Bumps", "Other_Faults"
]

# Держим те же параметры сплита
TEST_SIZE = 0.20
RANDOM_STATE = 42

K_LIST = [0.01, 0.03, 0.05, 0.10]

# Сценарии критичности — это управляемая гипотеза бизнеса.
# Здесь мы проверяем, насколько решение зависит от того, что именно объявлено критичным.
CRIT_SCENARIOS = {
    "crit_unknown_other_faults": ["Other_Faults"],
    "crit_scratches": ["K_Scatch", "Z_Scratch"],
    "crit_rare_stains_dirt": ["Stains", "Dirtiness"],
    "crit_broad_previous_high": ["Other_Faults", "K_Scatch", "Z_Scratch"],  # как было раньше
}

# Простая cost-модель политики кого отправить на доп. контроль
C_CHECK = 1.0
ESCAPE_CRIT = 100.0
ESCAPE_NONCRIT = 5.0


def build_y(df: pd.DataFrame) -> pd.Series:
    # Если таргет не one-hot, policy-метрики становятся бессмысленными
    y_onehot = df[LABEL_COLS]
    row_sum = y_onehot.sum(axis=1)
    if not (row_sum == 1).all():
        raise ValueError("Target is not valid one-hot.")
    return y_onehot.idxmax(axis=1)


def risk_score_from_proba(proba: np.ndarray, class_order: list, crit_classes: list) -> np.ndarray:
    # Риск определяем как суммарную вероятность критичных классов:
    idx = [class_order.index(c) for c in crit_classes]
    return proba[:, idx].sum(axis=1)


def recall_at_k(y_true: np.ndarray, risk: np.ndarray, crit_set: set, k: float) -> float:
    # recall@K измеряет качество именно очереди: сколько критичных мы поймали, проверяя только K%
    n = len(y_true)
    m = int(np.ceil(n * k))
    order = np.argsort(-risk)[:m]
    is_crit = np.array([yt in crit_set for yt in y_true])
    crit_total = is_crit.sum()
    if crit_total == 0:
        return np.nan
    crit_caught = is_crit[order].sum()
    return float(crit_caught / crit_total)


def expected_cost_policy(y_true: np.ndarray, risk: np.ndarray, crit_set: set, k: float) -> float:
    """
    Политика:
    - отправляем top-K% на проверку (платим C_CHECK за каждый)
    - всё, что НЕ проверили:
        - если критично - платим ESCAPE_CRIT
        - если не критично - платим ESCAPE_NONCRIT
    Возвращаем cost на 1000 объектов.

    Это грубая модель, но она нужна, чтобы сравнивать политики и сценарии,

    """
    n = len(y_true)
    m = int(np.ceil(n * k))
    order = np.argsort(-risk)
    checked = np.zeros(n, dtype=bool)
    checked[order[:m]] = True

    check_cost = checked.sum() * C_CHECK

    escape_cost = 0.0
    for i, yt in enumerate(y_true):
        if not checked[i]:
            if yt in crit_set:
                escape_cost += ESCAPE_CRIT
            else:
                escape_cost += ESCAPE_NONCRIT

    total_cost = check_cost + escape_cost
    return float(total_cost / n * 1000.0)


def random_baseline_recall(y_true: np.ndarray, crit_set: set, k: float) -> float:
    """
    Бенчмарк для интерпретации lift:
    при случайном выборе K% ожидаемый recall@K ≈ K,
    то есть мы ловим критичные пропорционально доле выборки в контроле.
    """
    return float(k)


def topk_overlap(risk_a: np.ndarray, risk_b: np.ndarray, k: float) -> float:
    # Диагностика почему одинаково
    n = len(risk_a)
    m = int(np.ceil(n * k))
    top_a = set(np.argsort(-risk_a)[:m])
    top_b = set(np.argsort(-risk_b)[:m])
    return len(top_a & top_b) / max(1, len(top_a))


def run_model(name, model, X_train, y_train, X_test):
    # В этом шаге важны вероятности, потому что политика строится на ранжировании по риску.
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    return proba, classes


def main():
    df = pd.read_csv(DATA_PATH)
    y = build_y(df)

    feature_cols = [c for c in df.columns if c not in LABEL_COLS]
    X = df[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    y_test = np.array(y_test)

    print(f"Test size: {len(y_test)}")
    print("Test class counts:")
    print(pd.Series(y_test).value_counts())
    print()

    # RF baseline
    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    proba_rf, classes_rf = run_model("RF", rf, X_train, y_train, X_test)

    # CatBoost baseline
    cb = CatBoostClassifier(
        loss_function="MultiClass",
        depth=8,
        learning_rate=0.08,
        iterations=1200,
        random_seed=RANDOM_STATE,
        verbose=False
    )
    proba_cb, classes_cb = run_model("CatBoost", cb, X_train, y_train, X_test)

    # Сценарий
    for scen_name, crit_classes in CRIT_SCENARIOS.items():
        crit_set = set(crit_classes)

        # Важно понимать базовую редкость» критичных на тесте
        # она влияет на интерпретацию recall@K
        crit_share = np.mean([yt in crit_set for yt in y_test])

        print("-" * 80)
        print(f"Сценарий: {scen_name}")
        print(f"Критичные классы: {crit_classes}")
        print(f"Критичные тест: {crit_share:.3f}")
        print("-" * 80)

        risk_rf = risk_score_from_proba(proba_rf, classes_rf, crit_classes)
        risk_cb = risk_score_from_proba(proba_cb, classes_cb, crit_classes)

        rows = []
        for k in K_LIST:
            r_rf = recall_at_k(y_test, risk_rf, crit_set, k)

            # lift показывает, насколько модель лучше случайного отбора при том же QC-бюджете K
            lift_rf = (r_rf / random_baseline_recall(y_test, crit_set, k)) if k > 0 else np.nan
            cost_rf = expected_cost_policy(y_test, risk_rf, crit_set, k)

            r_cb = recall_at_k(y_test, risk_cb, crit_set, k)
            lift_cb = (r_cb / random_baseline_recall(y_test, crit_set, k)) if k > 0 else np.nan
            cost_cb = expected_cost_policy(y_test, risk_cb, crit_set, k)

            row = {
                "K": k,
                "RF_recall@K": r_rf,
                "RF_lift_vs_random": lift_rf,
                "RF_cost_per_1000": cost_rf,
                "CB_recall@K": r_cb,
                "CB_lift_vs_random": lift_cb,
                "CB_cost_per_1000": cost_cb,
                # Если overlap высокий, различий в политике ждать не стоит — модели выбирают почти одни и те же изделия
                "topK_overlap_RF_CB": topk_overlap(risk_rf, risk_cb, k),
            }

            rows.append(row)

        out = pd.DataFrame(rows)
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(out)
        print()

if __name__ == "__main__":
    main()
