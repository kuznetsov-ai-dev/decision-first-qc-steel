'''
Анализ и проверка ошибок

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from catboost import CatBoostClassifier

DATA_PATH = r"faults.csv"
LABEL_COLS = ["Pastry","Z_Scratch","K_Scatch","Stains","Dirtiness","Bumps","Other_Faults"]
CRIT_CLASSES = ["K_Scatch","Z_Scratch"]

TEST_SIZE = 0.20
RANDOM_STATE = 42

TOP_N = 15 

def build_y(df):
    y_onehot = df[LABEL_COLS]
    assert (y_onehot.sum(axis=1) == 1).all()
    return y_onehot.idxmax(axis=1)

def main():
    df = pd.read_csv(DATA_PATH)
    y = build_y(df)
    feature_cols = [c for c in df.columns if c not in LABEL_COLS]
    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cb = CatBoostClassifier(
        loss_function="MultiClass",
        depth=8,
        learning_rate=0.08,
        iterations=1200,
        random_seed=RANDOM_STATE,
        verbose=False
    )
    cb.fit(X_train, y_train)

    proba = cb.predict_proba(X_test)
    classes = list(cb.classes_)
    y_pred = cb.predict(X_test).astype(str)

    # Общий репорт нужен как, какие классы системно проваливаются по precision/recall
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix фиксирует типичные подмены, какие дефекты модель путает между собой
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df)

    # Риск-скоринг для политики QC сумма вероятностей критичных классов
    crit_set = set(CRIT_CLASSES)
    idx_crit = [classes.index(c) for c in CRIT_CLASSES]
    risk = proba[:, idx_crit].sum(axis=1)

    test_df = X_test.copy()
    test_df["y_true"] = np.array(y_test)
    test_df["y_pred"] = y_pred
    test_df["risk_scratches"] = risk

    # Самые опасные FN для бизнеса критичные случаи с низким риском
    crit_mask = test_df["y_true"].isin(crit_set)
    fn_crit = test_df[crit_mask].sort_values("risk_scratches", ascending=True).head(TOP_N)
    print(f"\nТоп-{TOP_N} ОСОБО ОПАСНЫЕ случаи")
    print(fn_crit[["y_true","y_pred","risk_scratches"]].to_string(index=False))

    # Пожиратели очереди, некритичные случаи с высоким риском — они забирают QC-ресурс у критичных
    noncrit_mask = ~test_df["y_true"].isin(crit_set)
    fp_like = test_df[noncrit_mask].sort_values("risk_scratches", ascending=False).head(TOP_N)
    print(f"\nТоп-{TOP_N}")
    print(fp_like[["y_true","y_pred","risk_scratches"]].to_string(index=False))

    # Важности признаков здесь используются как ориентир для гипотез:
    # какие факторы модель реально использует для разделения дефектов (а значит, где искать причины ошибок)
    importances = cb.get_feature_importance()
    imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(15)
    print("\nTop-15 feature importances:")
    print(imp.to_string())

if __name__ == "__main__":
    main()
