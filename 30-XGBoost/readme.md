# XGBoost

Команда: 30, Малышев Дмитрий, Боровиков Артём

## Содержание каталога

- `XGBoost.pdf` — презентация
- `src/01_numeric_example.py` — пошаговый численный пример (один шаг XGBoost вручную, сверка с библиотекой)
- `src/02_compare.py` — сравнение XGBoost с LightGBM, sklearn GradientBoosting, Random Forest и логистической регрессией (результат в `results.json`)
- `src/03_xgboost_example.py` — пример на `breast_cancer`: early stopping, важность признаков

## Запуск кода

Python 3.10+

```bash
pip install numpy scikit-learn xgboost lightgbm
cd 30-XGBoost/src
python 01_numeric_example.py
python 02_compare.py        # ~1.5 минуты (sklearn GB и Random Forest на 30000×50 медленные)
python 03_xgboost_example.py
```