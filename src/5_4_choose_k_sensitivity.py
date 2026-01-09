'''
Анализ чувствительности политики контроля качества к стоимости проверок и пропусков дефектов.

При фиксированной модели перебираются различные стоимости проверки и утечки критичных дефектов,
и для каждой комбинации выбирается оптимальный бюджет контроля K (доля проверяемых изделий),
минимизирующий ожидаемые затраты при сохранении приемлемого recall по критичным дефектам.


'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

DATA_PATH = r"faults.csv"
LABEL_COLS = ["Pastry","Z_Scratch","K_Scatch","Stains","Dirtiness","Bumps","Other_Faults"]
CRIT_CLASSES = ["K_Scatch","Z_Scratch"]

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Сетка K 
K_LIST = [0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20]

# Сетка стоимостей — это чувствительность бизнес-решения к экономике:
# при каких ценах проверка выгоднее/невыгоднее, и какой K становится оптимальным
C_CHECK_LIST = [1, 2, 5, 10, 20]
ESCAPE_CRIT_LIST = [50, 100, 200, 500]
ESCAPE_NONCRIT = 0  

def build_y(df):
    y_onehot = df[LABEL_COLS]
    assert (y_onehot.sum(axis=1) == 1).all()
    return y_onehot.idxmax(axis=1)

def risk_from_proba(proba, classes, crit_classes):
    # Risk-score для политики
    idx = [classes.index(c) for c in crit_classes]
    return proba[:, idx].sum(axis=1)

def recall_precision_at_k(y_true, risk, crit_set, k):
    # Для политики нужны две стороны:
    # recall@K — сколько критичных поймали при данном бюджете;
    # precision@K — насколько чистая очередь: доля критичных среди отправленных на проверку
    n = len(y_true)
    m = int(np.ceil(n*k))
    top = np.argsort(-risk)[:m]
    is_crit = np.array([yt in crit_set for yt in y_true])
    crit_total = is_crit.sum()
    crit_caught = is_crit[top].sum()
    recall = crit_caught/crit_total if crit_total else np.nan
    precision = crit_caught/m if m else np.nan
    return float(recall), float(precision), int(m), int(crit_caught), int(crit_total)

def cost_per_1000(y_true, risk, crit_set, k, C_check, ESC_crit, ESC_noncrit):
    """
    Cost-модель политики:
    - top-K% отправляем на проверку - платим C_check за каждый
    - остальные не проверяем - платим цену утечки (критичная/некритичная)
    Возвращаем cost на 1000 изделий, чтобы масштаб был удобен для сравнения.
    """
    n = len(y_true)
    m = int(np.ceil(n*k))
    order = np.argsort(-risk)
    checked = np.zeros(n, dtype=bool)
    checked[order[:m]] = True

    check_cost = checked.sum() * C_check
    escape_cost = 0.0
    for i, yt in enumerate(y_true):
        if not checked[i]:
            escape_cost += ESC_crit if yt in crit_set else ESC_noncrit

    return float((check_cost + escape_cost)/n*1000.0)

def main():
    df = pd.read_csv(DATA_PATH)
    y = build_y(df)
    X = df[[c for c in df.columns if c not in LABEL_COLS]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    y_test = np.array(y_test)
    crit_set = set(CRIT_CLASSES)

    # Здесь модель фиксируем намеренно:
    # цель этого шага оптимизация политики K под разные стоимости 
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
    risk = risk_from_proba(proba, classes, CRIT_CLASSES)

    rows = []
    for C_check in C_CHECK_LIST:
        for ESC_crit in ESCAPE_CRIT_LIST:
            # При фиксированных стоимостях выбираем K, который минимизирует cost/1000:
            best = None
            for k in K_LIST:
                rec, prec, m, caught, total = recall_precision_at_k(y_test, risk, crit_set, k)
                cost = cost_per_1000(y_test, risk, crit_set, k, C_check, ESC_crit, ESCAPE_NONCRIT)
                cand = (cost, k, rec, prec, m, caught, total)
                if best is None or cand[0] < best[0]:
                    best = cand
            cost, k, rec, prec, m, caught, total = best
            rows.append({
                "C_check": C_check,
                "ESCAPE_CRIT": ESC_crit,
                "best_K": k,
                "recall@K": rec,
                "precision@K": prec,
                "checked_cnt": m,
                "crit_caught": caught,
                "crit_total": total,
                "cost_per_1000": cost
            })

    out = pd.DataFrame(rows).sort_values(["C_check","ESCAPE_CRIT"])
    with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 180):
        print(out)

if __name__ == "__main__":
    main()
