# Step 5.3 — код: sweep class weights (multiclass + binary) и выбор лучшего

"""
Перебор весов классов для CatBoost в двух постановках:
- multiclass (вся задача целиком) с усилением критичных классов,
- binary (критично и некритично) с усилением positive-класса.

Цель перебора — найти такой вес,
который даёт лучшую политику QC: минимальный cost/1000 при заданном бюджете K,
и при этом приемлемый recall@K по критичным дефектам.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

DATA_PATH = r"faults.csv"

LABEL_COLS = [
    "Pastry", "Z_Scratch", "K_Scatch",
    "Stains", "Dirtiness", "Bumps", "Other_Faults"
]

CRIT_CLASSES = ["K_Scatch", "Z_Scratch"]
K_LIST = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
WEIGHTS = [1, 2, 4, 8, 16]

TEST_SIZE = 0.20
RANDOM_STATE = 42

# cost model
C_CHECK = 1.0
ESCAPE_CRIT = 100.0
ESCAPE_NONCRIT = 5.0


def build_y(df: pd.DataFrame) -> pd.Series:
    y_onehot = df[LABEL_COLS]
    if not (y_onehot.sum(axis=1) == 1).all():
        raise ValueError("Target is not valid one-hot.")
    return y_onehot.idxmax(axis=1)


def risk_from_mc_proba(proba: np.ndarray, classes: list, crit_classes: list) -> np.ndarray:
    # Risk-score для политики
    idx = [classes.index(c) for c in crit_classes]
    return proba[:, idx].sum(axis=1)


def recall_at_k(y_true: np.ndarray, risk: np.ndarray, crit_set: set, k: float) -> float:
    # recall@K — базовая метрика качества очереди на контроль
    n = len(y_true)
    m = int(np.ceil(n * k))
    top = np.argsort(-risk)[:m]
    is_crit = np.array([yt in crit_set for yt in y_true])
    denom = is_crit.sum()
    return float(is_crit[top].sum() / denom) if denom > 0 else np.nan


def cost_per_1000(y_true: np.ndarray, risk: np.ndarray, crit_set: set, k: float) -> float:
    # Экономика политики: платим за проверки и платим за пропуски (критичные дороже)
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


def lift(recall_k: float, k: float) -> float:
    # Lift к случайному выбору: у random ожидаемый recall@K ≈ K
    return recall_k / k if k > 0 else np.nan


def main():
    df = pd.read_csv(DATA_PATH)
    y = build_y(df)
    X = df[[c for c in df.columns if c not in LABEL_COLS]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    y_test = np.array(y_test)
    crit_set = set(CRIT_CLASSES)
    y_train_bin = np.array([1 if yy in crit_set else 0 for yy in y_train])

    # Этот список здесь не используется напрямую, но фиксирует идею: порядок классов важен для весов.
    # Ниже мы берём фактический порядок из model.classes_ после обучения и строим class_weights уже под него.
    all_classes = sorted(y.unique().tolist())

    rows = []

    for w in WEIGHTS:
        # Первый fit делаем без весов, чтобы достать фактический порядок classes_ (и не гадать, как он отсортирован)
        cb_mc = CatBoostClassifier(
            loss_function="MultiClass",
            depth=8,
            learning_rate=0.08,
            iterations=1200,
            random_seed=RANDOM_STATE,
            verbose=False
        )

        cb_mc.fit(X_train, y_train)

        # classes learned
        classes_mc = list(cb_mc.classes_)

        # Теперь строим веса в точном порядке classes_:
        # критичным ставим w, остальным 1.0 — это управляемое смещение оптимизации в сторону критичных ошибок
        class_weights = []
        for c in classes_mc:
            class_weights.append(float(w) if c in crit_set else 1.0)

        cb_mc_w = CatBoostClassifier(
            loss_function="MultiClass",
            depth=8,
            learning_rate=0.08,
            iterations=1200,
            random_seed=RANDOM_STATE,
            verbose=False,
            class_weights=class_weights
        )
        cb_mc_w.fit(X_train, y_train)

        proba_mc = cb_mc_w.predict_proba(X_test)
        risk_mc = risk_from_mc_proba(proba_mc, list(cb_mc_w.classes_), CRIT_CLASSES)

        # В бинарной постановке мы напрямую оптимизируем P(crit),
        # а вес w усиливает штраф за промах по критичным (pos=1).
        cb_bin = CatBoostClassifier(
            loss_function="Logloss",
            depth=6,
            learning_rate=0.06,
            iterations=1200,
            random_seed=RANDOM_STATE,
            verbose=False,
            class_weights=[1.0, float(w)]
        )
        cb_bin.fit(X_train, y_train_bin)
        risk_bin = cb_bin.predict_proba(X_test)[:, 1]

        for k in K_LIST:
            r_mc = recall_at_k(y_test, risk_mc, crit_set, k)
            r_bin = recall_at_k(y_test, risk_bin, crit_set, k)
            rows.append({
                "w": w,
                "K": k,
                "MC_recall@K": r_mc,
                "MC_lift": lift(r_mc, k),
                "MC_cost/1000": cost_per_1000(y_test, risk_mc, crit_set, k),
                "BIN_recall@K": r_bin,
                "BIN_lift": lift(r_bin, k),
                "BIN_cost/1000": cost_per_1000(y_test, risk_bin, crit_set, k),
            })

    out = pd.DataFrame(rows)

    with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 220):
        print(out)

    # Выбор лучшего по cost — это и есть decision-first критерий.
    # Мы отдельно выбираем лучший multiclass и лучший binary для каждого K, потому что бюджет QC задаётся извне.
    print("\nЛучший (min cost/1000) per K — Multiclass:")
    best_mc = out.loc[
        out.groupby("K")["MC_cost/1000"].idxmin(),
        ["K", "w", "MC_recall@K", "MC_lift", "MC_cost/1000"]
    ]
    print(best_mc.to_string(index=False))

    print("\nЛучший (min cost/1000) per K — Binary:")
    best_bin = out.loc[
        out.groupby("K")["BIN_cost/1000"].idxmin(),
        ["K", "w", "BIN_recall@K", "BIN_lift", "BIN_cost/1000"]
    ]
    print(best_bin.to_string(index=False))


if __name__ == "__main__":
    main()
