"""
Проверки корректности данных и целостности таргета.

Скрипт намеренно останавливается при первой ошибке,
если нарушена спецификация таргета или базовые предположения о данных,
чтобы избежать тихих и незаметных проблем при обучении модели.
"""


import pandas as pd
import numpy as np

DATA_PATH = r"faults.csv"

# Таргет ожидаем именно в one-hot формате
LABEL_COLS = [
    "Pastry", "Z_Scratch", "K_Scatch",
    "Stains", "Dirtiness", "Bumps", "Other_Faults"
]

# Гипотеза о критичности: помогает сразу видеть долю риска по группам
SEVERITY_GROUPS = {
    "High":   ["Other_Faults", "K_Scatch", "Z_Scratch"],
    "Medium": ["Bumps", "Pastry", "Stains"],
    "Low":    ["Dirtiness"],
}

def pct(x: float) -> str:
    return f"{100 * x:.2f}%"

def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 80)
    print("ШАГ 1 — СПЕЦИФИКАЦИЯ ДАННЫХ И ЦЕЛОСТНОСТЬ ТАРГЕТА")
    print("=" * 80)
    print(f"Загружено: {DATA_PATH}")
    print(f"Размер: {df.shape[0]} строк × {df.shape[1]} колонок\n")

    # Эти две метрики выводим первыми: они сразу показывают, насколько можно доверять сырым данным
    missing_total = int(df.isna().sum().sum())
    dup_rows = int(df.duplicated().sum())
    print(f"Пропуски (всего значений): {missing_total}")
    print(f"Дубликаты строк: {dup_rows}\n")

    # Принципиально падать сразу, если спецификация таргета не совпала: иначе можно тихо обучить не то
    missing_label_cols = [c for c in LABEL_COLS if c not in df.columns]
    if missing_label_cols:
        raise ValueError(f"Колонки таргета не найдены: {missing_label_cols}")

    y_onehot = df[LABEL_COLS].copy()

    # Для one-hot допускаем только {0,1}
    allowed = {0, 1}
    bad_vals = ~y_onehot.stack().isin(allowed)
    if bad_vals.any():
        bad_examples = y_onehot.stack()[bad_vals].head(10)
        raise ValueError(
            "(ожидается 0/1). Примеры:\n"
            f"{bad_examples}"
        )

    # Проверка sum(one-hot)==1 защищает от двух критичных сценариев:
    # Нет класса (все нули) и 2) несколько классов одновременно (несколько единиц)
    row_sum = y_onehot.sum(axis=1)
    print(row_sum.value_counts().sort_index())
    print()

    n_bad = int((row_sum != 1).sum())
    print(f"Строк, где sum(one-hot) != 1: {n_bad}")
    if n_bad > 0:
        print("Примеры некорректных строк (первые 5):")
        print(df.loc[row_sum != 1, LABEL_COLS].head())
        raise ValueError("Таргет не корректный")
    print()

    # Перевод one-hot -> multiclass делаем через idxmax
    y = y_onehot.idxmax(axis=1)
    print("y (мультикласс) построен из one-hot: OK\n")

    # Класс-дистрибуция — базовая диагностика сложности задачи и будущих компромиссов метрик/валидации
    class_counts = y.value_counts()
    class_share = (class_counts / len(df)).sort_values(ascending=False)

    print("Распределение классов (кол-во):")
    print(class_counts)
    print("\nРаспределение классов (доля):")
    print(class_share.apply(pct))
    print()

    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Дисбаланс (max/min): {imbalance_ratio:.2f}x\n")

    # Группы критичности считаем через one-hot
    print("Распределение по группам критичности (гипотеза):")
    sev_counts = {}
    for grp, cols in SEVERITY_GROUPS.items():
        sev_counts[grp] = int(y_onehot[cols].sum().sum())
    sev_share = {k: pct(v / len(df)) for k, v in sev_counts.items()}
    print("Кол-во:", sev_counts)
    print("Доля  :", sev_share)
    print()

    # Признаки
    feature_cols = [c for c in df.columns if c not in LABEL_COLS]
    X = df[feature_cols]

    # Нечисловые признаки
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X[c])]
    print(f"Признаков: {len(feature_cols)}")
    print(f"Нечисловые признаки: {non_numeric}\n")

    # Константные признаки сразу фиксируем
    nunique = X.nunique(dropna=False).sort_values()
    const = nunique[nunique <= 1]
    print(f"Константные признаки: {list(const.index)}\n")

    # Вывод
    print("Сводка по признакам (выбранные перцентили):")
    desc = X.describe(percentiles=[0.01, 0.5, 0.99]).T
    print(desc[["mean", "std", "min", "1%", "50%", "99%", "max"]].head(12))

    # Корреляции
    print("Топ пар признаков по |corr|:")
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().sort_values(ascending=False)

    print(pairs.head(12))
    print()
    high_corr_n = int((pairs > 0.98).sum())
    print(f"Пар с |corr| > 0.98: {high_corr_n}\n")

    # Быстрая оценка склонности к выбросам (правило IQR) — не как строгий детектор,
    # а как сигнал: где могут понадобиться робастные лоссы/скейлинг/лог-преобразования
    out_rows = []
    for c in feature_cols:
        s = X[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            rate = 0.0
        else:
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            rate = float(((s < lo) | (s > hi)).mean())
        out_rows.append((c, rate, float(s.min()), float(s.max())))

    out_df = (
        pd.DataFrame(out_rows, columns=["feature", "iqr_outlier_rate", "min", "max"])
        .sort_values("iqr_outlier_rate", ascending=False)
    )

    print("Топ-10 признаков по доле IQR-выбросов:")
    print(out_df.head(10).to_string(index=False))
    print()

    print("ШАГ 1 ГОТОВ")

if __name__ == "__main__":
    main()
