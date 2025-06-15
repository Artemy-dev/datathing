import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stat
from statsmodels.tsa.seasonal import seasonal_decompose

def get_summary_report(df: pd.DataFrame, column: str) -> None:
    """
    Генерирует автоматический отчет по данным в указанной колонке, включая поиск аномальных значений.
    """
    print(f"\n=== Автоматический анализ данных для '{column}' ===\n")
    mean_value = df[column].mean()
    median_value = df[column].median()
    std_value = df[column].std()
    min_value = df[column].min()
    max_value = df[column].max()
    print(f"Среднее значение: {mean_value:.2f}")
    print(f"Медиана: {median_value:.2f}")
    print(f"Стандартное отклонение: {std_value:.2f}")
    print(f"Минимальное значение: {min_value}")
    print(f"Максимальное значение: {max_value}")
    lower_bound_3sigma = mean_value - 3 * std_value
    upper_bound_3sigma = mean_value + 3 * std_value
    outliers_3sigma = df[(df[column] < lower_bound_3sigma) | (df[column] > upper_bound_3sigma)]
    if not outliers_3sigma.empty:
        print("\nВыявлены аномальные значения по правилу 3σ:")
        print(outliers_3sigma)
    else:
        print("\nАномальные значения по правилу 3σ не обнаружены.")
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound_iqr = Q1 - 1.5 * IQR
    upper_bound_iqr = Q3 + 1.5 * IQR
    outliers_iqr = df[(df[column] < lower_bound_iqr) | (df[column] > upper_bound_iqr)]
    if not outliers_iqr.empty:
        print("Выявлены аномальные значения по правилу IQR:")
        print(outliers_iqr)
    else:
        print("Аномальные значения по правилу IQR не обнаружены.")
    if mean_value > median_value:
        print("\nВывод: распределение имеет правостороннюю асимметрию (более высокие значения).")
    elif mean_value < median_value:
        print("\nВывод: распределение имеет левостороннюю асимметрию (чаще встречаются низкие значения).")
    else:
        print("\nВывод: распределение симметрично.")

def get_scan(df: pd.DataFrame) -> None:
    """
    Выводит основную информацию о DataFrame.
    """
    print("\n=== Информация о данных ===")
    print(df.info())

def get_stats(df: pd.DataFrame) -> None:
    """
    Выводит описательную статистику для числовых колонок.
    """
    print("\n=== Базовая статистика ===")
    print(round(df.describe()))

def get_unique_counts(df: pd.DataFrame) -> None:
    """
    Считает количество уникальных значений в каждой колонке DataFrame.
    """
    print("\n=== Уникальные значения ===")
    print(df.nunique())

def get_unique_values(df: pd.DataFrame, column: str) -> None:
    """
    Выводит уникальные значения указанной колонки.
    """
    print(f"\n=== Уникальные значения в колонке '{column}' ===")
    print(df[column].unique())

def get_missing_values(df: pd.DataFrame) -> None:
    """
    Анализирует пропущенные значения в DataFrame.
    """
    print("\n=== Пропуски ===")
    print(df.isnull().sum())

def get_corr_matrix(df: pd.DataFrame) -> None:
    """
    Вычисляет и выводит матрицу корреляций для числовых колонок.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    print("\n=== Матрица корреляций ===")
    print(correlation_matrix.round(3))

def plot_distribution(df: pd.DataFrame, columns: list) -> None:
    """
    Визуализирует распределение числовых данных через гистограммы и boxplot.
    """
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(columns, 1):
        plt.subplot(2, 3, i)
        sns.histplot(df[col], kde=True, bins=15)
        plt.title(f'Распределение: {col}')
        plt.xlabel("")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 5))
    for i, col in enumerate(columns, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot: {col}')
    plt.tight_layout()
    plt.show()

def plot_corr_heatmap(df: pd.DataFrame) -> None:
    """
    Визуализирует матрицу корреляций через тепловую карту.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Тепловая карта корреляций")
    plt.show()

def plot_pairplot(df: pd.DataFrame) -> None:
    """
    Строит pairplot для всех числовых колонок DataFrame.
    """
    sns.pairplot(df)
    plt.show()

def plot_decomposition_time_series(df: pd.DataFrame, column: str, period: int = 7, model: str = "additive") -> None:
    """
    Строит декомпозицию временного ряда для указанной колонки с использованием
    выбранной модели декомпозиции и периода сезонности.
    """
    result = seasonal_decompose(df[column], model=model, period=period)
    fig = result.plot()
    plt.suptitle(f"Декомпозиция временного ряда ({model} модель)")
    for ax in fig.axes:
        ax.set_xticks(np.arange(0, len(df)))
        ax.set_xticklabels(np.arange(1, len(df) + 1))
    plt.tight_layout()
    plt.show()

def detect_sigma_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Выявляет выбросы по правилу 3σ (трех сигм) и добавляет колонку 'аномалия_3σ'.
    """
    mean_value = df[column].mean()
    std_value = df[column].std()
    df["аномалия_3σ"] = (df[column] < mean_value - 3 * std_value) | (df[column] > mean_value + 3 * std_value)
    print("\n=== Поиск выбросов по правилу трех сигм ===")
    print(df)
    return df

def detect_iqr_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Выявляет выбросы методом межквартильного размаха (IQR) и добавляет колонку 'аномалия_IQR'.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df["аномалия_IQR"] = (df[column] < lower_bound) | (df[column] > upper_bound)
    print("\n=== Поиск выбросов по IQR ===")
    print(df)
    return df

def test_shapiro(df: pd.DataFrame, column: str) -> None:
    """
    Выполняет тест Шапиро-Уилка для проверки нормальности распределения.
    """
    statistic, pvalue = stat.shapiro(df[column])
    print("\n=== Тест Шапиро-Уилка ===")
    print(f"Статистика теста: {statistic:.3f}")
    print(f"p-value: {pvalue:.3f}")
    if pvalue > 0.05:
        print("Данные нормальны, можно применять T-тест")
    else:
        print("Данные несимметричны или содержат выбросы, лучше использовать U-тест Манна-Уитни")

def test_ttest(df: pd.DataFrame, column: str, group_col: str) -> None:
    """
    Выполняет T-тест для проверки значимых различий между двумя группами.
    """
    group_A = df[df[group_col] == "A"][column]
    group_B = df[df[group_col] == "B"][column]
    t_stat, p_value = stat.ttest_ind(group_A, group_B, equal_var=False)
    print("\n=== T-тест ===")
    print(f"Статистика теста: {t_stat:.3f}")
    print(f"p-value: {p_value:.3f}")
    if p_value < 0.05:
        print("Различия между группами значимы!")
    else:
        print("Различия незначительны, данные могут быть случайными.")

def test_mannwhitneyu(df: pd.DataFrame, column: str, group_col: str) -> None:
    """
    Выполняет U-тест Манна-Уитни для проверки различий между двумя группами, если данные не нормальны.
    """
    group_A = df[df[group_col] == "A"][column]
    group_B = df[df[group_col] == "B"][column]
    u_stat, p_value_u = stat.mannwhitneyu(group_A, group_B)
    print("\n=== U-тест Манна-Уитни ===")
    print(f"Статистика теста: {u_stat:.3f}")
    print(f"p-value: {p_value_u:.3f}")
    if p_value_u < 0.05:
        print("Различия между группами значимы!")
    else:
        print("Различия незначительны, данные могут быть случайными.")

def test_chi2(df: pd.DataFrame, *columns: str) -> None:
    """
    Выполняет хи-квадрат тест (χ²) для проверки зависимости между несколькими категориальными переменными.
    """
    table = df[list(columns)].values
    chi2_stat, p_value, _, _ = stat.chi2_contingency(table)
    print("\n=== Хи-квадрат тест ===")
    print(f"Статистика теста: {chi2_stat:.3f}")
    print(f"p-value: {p_value:.3f}")
    if p_value < 0.05:
        print("Есть статистически значимая связь между переменными!")
    else:
        print("Связь между переменными незначительна.")