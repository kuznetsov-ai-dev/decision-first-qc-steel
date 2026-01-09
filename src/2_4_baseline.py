"""
Валидация, метрики и decision-first baseline.

Скрипт сравнивает несколько базовых моделей не только по ML-метрикам,
но и по бизнес-метрикам QC: сколько дефектов ловим при проверке top-K%
и во сколько это обходится с учётом стоимости проверок и «утечек».
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = r"faults.csv"

# Спецификация таргета
LABEL_COLS = [
    "Pastry", "Z_Scratch", "K_Scatch",
    "Stains", "Dirtiness", "Bumps", "Other_Faults"
]

# Почти-дубликаты по корреляции из шага 1.
# Для линейных моделей это принципиально, иначе коэффициенты становятся неустойчивыми.
DROP_COLS_LINEAR = [
    "TypeOfSteel_A400",  
    "Y_Maximum",       
    "X_Maximum",    
]

# Гипотеза критичности для policy-метрик
SEVERITY_GROUPS = {
    "High":   ["Other_Faults", "K_Scatch", "Z_Scratch"],
    "Medium": ["Bumps", "Pastry", "Stains"],
    "Low":    ["Dirtiness"],
}

# Политика QC: проверяем только top-K% по риск-скорингу
K_LIST = [0.01, 0.03, 0.05, 0.10]

# Стоимости условные: важны относительные масштабы, а не абсолютные цифры
C_CHECK = 1.0
ESCAPE_COST = {  # цена «пропуска» дефекта, если изделие не проверили
    "Other_Faults": 100.0,
    "K_Scatch":     100.0,
    "Z_Scratch":    80.0,
    "Bumps":        30.0,
    "Pastry":       20.0,
    "Stains":       20.0,
    "Dirtiness":    10.0,
}
# -------------------------


def build_y(df: pd.DataFrame) -> pd.Series:
    """
    Жёсткая проверка one-hot таргета.
    Если здесь ошибка — дальнейшее моделирование не имеет смысла.
    """
    y_onehot = df[LABEL_COLS]
    row_sum = y_onehot.sum(axis=1)
    if not (row_sum == 1).all():
        bad = (row_sum != 1).sum()
        raise ValueError(f"Target is not valid one-hot. bad rows: {bad}")
    return y_onehot.idxmax(axis=1)


def compute_ml_metrics(y_true, y_pred, title=""):
    """
    Классические ML-метрики нужны как sanity-check:
    мы должны понимать, не упала ли модель,
    но оптимизация будет идти не по ним.
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print("=" * 80)
    print(f"ML METRICS — {title}")
    print("=" * 80)
    print(f"macro-F1: {macro_f1:.4f}\n")

    print("Confusion matrix (rows=true, cols=pred):")
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    print("\nPer-class recall:")
    # recall по классам важен, чтобы видеть, какие дефекты модель системно игнорирует
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    recall = {k: rep[k]["recall"] for k in rep.keys() if k in labels}
    recall_s = pd.Series(recall).sort_values(ascending=False)
    print(recall_s.apply(lambda x: f"{x:.3f}"))
    print()

    return macro_f1, cm_df, recall_s


def risk_score_from_proba(proba: np.ndarray, class_order: list, high_classes: list) -> np.ndarray:
    """
    Risk-score — это не argmax класса, а агрегированный риск.
    Здесь он определяется как сумма вероятностей критичных дефектов.
    """
    idx = [class_order.index(c) for c in high_classes]
    return proba[:, idx].sum(axis=1)


def policy_metrics(y_true, proba, class_order, k_list, high_classes):
    """
    Decision-first политика:
    считаем risk_score
    проверяем только top-K% изделий
    для непроверенных изделий платим цену утечки дефекта

    Это приближение реальной логики QC, а не абстрактной классификации.
    """
    y_true = np.array(y_true)
    risk = risk_score_from_proba(proba, class_order, high_classes)

    critical_set = set(high_classes)
    is_critical = np.array([yt in critical_set for yt in y_true])

    rows = []
    n = len(y_true)

    # Проверяем изделия в порядке убывания риска
    order = np.argsort(-risk)

    for k in k_list:
        m = int(np.ceil(n * k))
        checked_idx = set(order[:m])
        checked = np.array([i in checked_idx for i in range(n)])

        # Бизнес-KPI: сколько критичных дефектов реально поймали
        crit_total = int(is_critical.sum())
        crit_caught = int((is_critical & checked).sum())
        crit_recall_k = (crit_caught / crit_total) if crit_total > 0 else np.nan

        # Экономика: цена проверок + цена пропущенных дефектов
        check_cost = checked.sum() * C_CHECK
        escape_cost = 0.0
        for i in range(n):
            if not checked[i]:
                escape_cost += ESCAPE_COST.get(y_true[i], 0.0)

        total_cost = check_cost + escape_cost
        cost_per_1000 = total_cost / n * 1000.0

        rows.append({
            "K": k,
            "checked_cnt": int(checked.sum()),
            "critical_total": crit_total,
            "critical_caught": crit_caught,
            "critical_recall@K": crit_recall_k,
            "check_cost_total": check_cost,
            "escape_cost_total": escape_cost,
            "total_cost_per_1000": cost_per_1000,
        })

    return pd.DataFrame(rows)


def fit_predict_model(name, model, X_train, y_train, X_test, y_test, proba_supported=True):
    """
    Обёртка, чтобы все модели сравнивались в одинаковых условиях.
    """
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    proba = None
    class_order = None
    if proba_supported and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        class_order = list(model.classes_)

    return pred, proba, class_order


def main():
    df = pd.read_csv(DATA_PATH)
    y = build_y(df)

    # Явно отделяем признаки от таргета, чтобы исключить утечки
    feature_cols = [c for c in df.columns if c not in LABEL_COLS]
    X = df[feature_cols].copy()

    # Holdout-валидация как базовый и максимально прозрачный вариант
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test  size: {len(X_test)}")
    print("Test class distribution:")
    print(pd.Series(y_test).value_counts())
    print()

    # Logistic Regression
    # Для линейной модели убираем коллинеарные признаки
    X_train_lin = X_train.drop(columns=[c for c in DROP_COLS_LINEAR if c in X_train.columns])
    X_test_lin  = X_test.drop(columns=[c for c in DROP_COLS_LINEAR if c in X_test.columns])

    logreg = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            multi_class="multinomial",
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    y_pred_lr, proba_lr, classes_lr = fit_predict_model(
        "LogReg", logreg, X_train_lin, y_train, X_test_lin, y_test
    )
    compute_ml_metrics(
        y_test, y_pred_lr,
        title="LogisticRegression (scaled, class_weight=balanced)"
    )

    if proba_lr is not None:
        high_classes = SEVERITY_GROUPS["High"]
        pm_lr = policy_metrics(y_test, proba_lr, classes_lr, K_LIST, high_classes)
        print("BUSINESS METRICS (policy top-K QC) — LogReg")
        print(pm_lr.to_string(index=False))
        print()

    # RandomForest
    # Деревья используем как более гибкий baseline
    rf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    y_pred_rf, proba_rf, classes_rf = fit_predict_model(
        "RF", rf, X_train, y_train, X_test, y_test
    )
    compute_ml_metrics(
        y_test, y_pred_rf,
        title="RandomForest (class_weight=balanced_subsample)"
    )

    if proba_rf is not None:
        high_classes = SEVERITY_GROUPS["High"]
        pm_rf = policy_metrics(y_test, proba_rf, classes_rf, K_LIST, high_classes)
        print("BUSINESS METRICS (policy top-K QC) — RandomForest")
        print(pm_rf.to_string(index=False))
        print()

    # CatBoost
    # Как сильный табличный baseline
    try:
        from catboost import CatBoostClassifier

        cb = CatBoostClassifier(
            loss_function="MultiClass",
            depth=8,
            learning_rate=0.08,
            iterations=2000,
            random_seed=42,
            verbose=200
        )

        y_pred_cb, proba_cb, classes_cb = fit_predict_model(
            "CatBoost", cb, X_train, y_train, X_test, y_test
        )
        compute_ml_metrics(
            y_test, y_pred_cb,
            title="CatBoost (baseline params)"
        )

        if proba_cb is not None:
            high_classes = SEVERITY_GROUPS["High"]
            pm_cb = policy_metrics(y_test, proba_cb, classes_cb, K_LIST, high_classes)
            print("BUSINESS METRICS (policy top-K QC) — CatBoost")
            print(pm_cb.to_string(index=False))
            print()

    except Exception as e:
        print("Ошибка", repr(e))

if __name__ == "__main__":
    main()
