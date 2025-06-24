# 🚀 [datathing](https://pypi.org/project/datathing/) — Automated Data Analysis & Visualization  
Python module for automatic analysis, visualization, and statistical tests on tabular data (pandas.DataFrame).  
(Python-модуль для автоматического анализа, визуализации и статистических тестов над табличными данными.)

---

## 💻 Supported Platforms

- ✅ Windows  
- ✅ macOS  
- ✅ Linux  

---

## 📦 Features / Возможности

| Функция | Описание |
|--------|----------|
| `get_summary_report(df, column)` | Автоматический отчет по колонке: среднее, медиана, выбросы (IQR и 3σ), асимметрия |
| `get_scan(df)` | Краткая информация о структуре DataFrame |
| `get_stats(df)` | Базовая описательная статистика по числовым колонкам |
| `get_unique_counts(df)` | Количество уникальных значений в каждой колонке |
| `get_unique_values(df, column)` | Все уникальные значения в выбранной колонке |
| `get_missing_values(df)` | Подсчет пропущенных значений по колонкам |
| `get_corr_matrix(df)` | Расчет корреляций между числовыми колонками |
| `plot_distribution(df, columns)` | Визуализация распределения (гистограммы и boxplot) |
| `plot_corr_heatmap(df)` | Тепловая карта корреляций |
| `plot_pairplot(df)` | Парные графики для числовых переменных |
| `plot_decomposition_time_series(df, column, period=7, model="additive")` | Декомпозиция временного ряда (тренд, сезонность, остаток) |
| `detect_sigma_outliers(df, column)` | Поиск выбросов по правилу трех сигм (3σ) |
| `detect_iqr_outliers(df, column)` | Поиск выбросов по IQR (межквартильному размаху) |
| `test_shapiro(df, column)` | Тест Шапиро-Уилка на нормальность распределения |
| `test_ttest(df, column, group_col)` | T-тест для сравнения двух групп |
| `test_mannwhitneyu(df, column, group_col)` | U-тест Манна-Уитни — для ненормальных распределений |
| `test_chi2(df, *columns)` | Хи-квадрат тест на зависимость категориальных переменных |

---

## ❓ Why this project? / Зачем этот проект?

This module helps automate exploratory data analysis, reducing manual effort and improving data insight clarity.  
Модуль помогает автоматизировать разведочный анализ данных, снижая трудозатраты и повышая качество понимания данных.

---

## 🚀 Installation / Установка

```bash
pip install datathing
````

---

## ⚙️ Usage / Пример использования

```python
import pandas as pd
import datathing as dfg

df = pd.DataFrame({
    "возраст": [25, 30, 22, 40, 19, 120],
    "зарплата": [50000, 60000, 45000, 80000, 30000, 200000],
    "пол": ["M", "F", "M", "F", "M", "F"]
})

dfg.get_summary_report(df, "возраст")
```

```
=== Автоматический анализ данных для 'возраст' ===

Среднее значение: 42.67
Медиана: 27.50
Стандартное отклонение: 38.59
Минимальное значение: 19
Максимальное значение: 120

Аномальные значения по правилу 3σ не обнаружены.
Выявлены аномальные значения по правилу IQR:
   возраст  зарплата пол
5      120    200000   F

Вывод: распределение имеет правостороннюю асимметрию (более высокие значения).
```
---

## 📁 Project Structure / Структура проекта

```
├── datathing
├── datathing.egg-info
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## 👤 Author / Автор

**Artem Grachev**<br>
Python/Golang Developer | ML & DevOps Enthusiast<br>
Telegram: [@Artemy\_Develop](https://t.me/Artemy_Develop)<br>
GitHub: [Artemy-dev](https://github.com/Artemy-dev)

---

## 🌍 SEO Keywords

* data analysis python
* exploratory data analysis
* pandas data visualization
* statistical tests python
* anomaly detection python
* time series decomposition
* automatic EDA tool
* data science python library
* data profiling pandas

---

![PyPI version](https://img.shields.io/pypi/v/datathing)
