"""
Сравниваем:
multiclass - риск = сумма P(критичных классов)
binary (критично и некритично) - риск = P(crit)

Идея: если реальная цель — ловить конкретный критичный набор, то бинарная модель
может дать более точное ранжирование очереди на контроль (выше recall@K и ниже cost),
чем попытка решить всю мультиклассовую задачу в лоб.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier

DATA_PATH = r"faults.csv"

LABEL_COLS = [
    "Pastry", "Z_Scratch", "K_Scatch",
    "Stains", "Dirtiness", "Bumps", "Other_Faults"
]

TEST_SIZE = 0.20
RANDOM_STATE = 42

K_LIST = [0.01, 0.03, 0.05, 0.10]

# Основной сценарий: фиксируем одну бизнес-версию критичности
CRIT_CLASSES = ["K_Scatch", "Z_Scratch"]

# cost-модель (условная)
C_CHECK = 1.0
ESCAPE_CRIT = 100.0
ESCAPE_NONCRIT = 5.0


def build_y(df: pd.DataFrame) -> pd.Series:
    # Вся логика ниже предполагает строгий one-hot
    y_onehot = df[LABEL_COLS]
    row_sum = y_onehot.sum(axis=1)
    if not (row_sum == 1).all():
        raise ValueError("Target is not valid one-hot.")
    return y_onehot.idxmax(axis=1)


def recall_at_k(y_true: np.ndarray, risk: np.ndarray, crit_set: set, k: float) -> float:
    # Это ключевая метрика политики сколько критичных мы поймали, проверяя только K% изделий
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
    # Cost считаем именно по политике проверка стоит денег, а пропуск дефекта стоит ещё дороже
    n = len(y_true)
    m = int(np.ceil(n * k))
    order = np.argsort(-risk)
    checked = np.zeros(n, dtype=bool)
    checked[order[:m]] = True

    check_cost = checked.sum() * C_CHECK

    escape_cost = 0.0
    for i, yt in enumerate(y_true):
        if not checked[i]:
            escape_cost += ESCAPE_CRIT if yt in crit_set else ESCAPE_NONCRIT

    return float((check_cost + escape_cost) / n * 1000.0)


def lift_vs_random(recall_k: float, k: float) -> float:
    # Lift нужен для интерпретации насколько учше слепого отбора при том же бюджете проверок K
    return recall_k / k if k > 0 else np.nan


def topk_set(risk: np.ndarray, k: float) -> set:
    n = len(risk)
    m = int(np.ceil(n * k))
    return set(np.argsort(-risk)[:m])


def overlap(a: np.ndarray, b: np.ndarray, k: float) -> float:
    # Диагностика почему всё одинаково
    A = topk_set(a, k)
    B = topk_set(b, k)
    return len(A & B) / max(1, len(A))


def main():
    df = pd.read_csv(DATA_PATH)
    y = build_y(df)

    feature_cols = [c for c in df.columns if c not in LABEL_COLS]
    X = df[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    y_test = np.array(y_test)
    crit_set = set(CRIT_CLASSES)

    # Бинаризация под политику
    y_train_bin = np.array([1 if yy in crit_set else 0 for yy in y_train])
    y_test_bin  = np.array([1 if yy in crit_set else 0 for yy in y_test])

    print(f"Test size: {len(y_test)} | Доля критичных: {y_test_bin.mean():.3f}")
    print()

    # RF multiclass
    # Риск = сумма вероятностей критичных классов 
    rf_mc = RandomForestClassifier(
        n_estimators=600,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    rf_mc.fit(X_train, y_train)
    proba_rf = rf_mc.predict_proba(X_test)
    classes_rf = list(rf_mc.classes_)
    idx_rf = [classes_rf.index(c) for c in CRIT_CLASSES]
    risk_rf_mc = proba_rf[:, idx_rf].sum(axis=1)

    # CatBoost multiclass
    cb_mc = CatBoostClassifier(
        loss_function="MultiClass",
        depth=8,
        learning_rate=0.08,
        iterations=1200,
        random_seed=RANDOM_STATE,
        verbose=False
    )
    cb_mc.fit(X_train, y_train)
    proba_cb = cb_mc.predict_proba(X_test)
    classes_cb = list(cb_mc.classes_)
    idx_cb = [classes_cb.index(c) for c in CRIT_CLASSES]
    risk_cb_mc = proba_cb[:, idx_cb].sum(axis=1)

    # CatBoost binary
    # Здесь оптимизируем именно вероятность критичности class_weights усиливает цену ошибки на критичных
    cb_bin = CatBoostClassifier(
        loss_function="Logloss",
        depth=6,
        learning_rate=0.06,
        iterations=1200,
        random_seed=RANDOM_STATE,
        verbose=False,
        # class_weights: усиливаем критичные (pos=1)
        class_weights=[1.0, 4.0]
    )
    cb_bin.fit(X_train, y_train_bin)
    risk_cb_bin = cb_bin.predict_proba(X_test)[:, 1]

    # LogisticRegression (sanity)
    # Линейный baseline как контрольная точка
    lr_bin = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])
    lr_bin.fit(X_train, y_train_bin)
    risk_lr_bin = lr_bin.predict_proba(X_test)[:, 1]

    # evaluate
    models = {
        "RF_multiclass_sumP": risk_rf_mc,
        "CB_multiclass_sumP": risk_cb_mc,
        "CB_binary_P(crit)": risk_cb_bin,
        "LR_binary_P(crit)": risk_lr_bin,
    }

    rows = []
    for k in K_LIST:
        row = {"K": k}
        for name, risk in models.items():
            r = recall_at_k(y_test, risk, crit_set, k)
            row[f"{name}_recall@K"] = r
            row[f"{name}_lift"] = lift_vs_random(r, k)
            row[f"{name}_cost/1000"] = expected_cost_policy(y_test, risk, crit_set, k)

        # Overlap считаем относительно CB_binary
        # поэтому важно понимать, насколько его top-K действительно отличается от multiclass ранжирования
        row["overlap_RFmc_vs_CBbin"] = overlap(risk_rf_mc, risk_cb_bin, k)
        row["overlap_CBmc_vs_CBbin"] = overlap(risk_cb_mc, risk_cb_bin, k)
        rows.append(row)

    out = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(out)

if __name__ == "__main__":
    main()
