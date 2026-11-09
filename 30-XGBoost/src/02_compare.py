"""Сравнение XGBoost с альтернативами. Результат -> results.json."""
import json, time
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

def models():
    d = {
        "XGBoost": xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                                     subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=0),
        "sklearn GB": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    }
    if lgb: d["LightGBM"] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                               n_jobs=-1, random_state=0, verbose=-1)
    return d

def run(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    out = {}
    for name, m in models().items():
        t = time.perf_counter(); m.fit(Xtr, ytr); dt = time.perf_counter() - t
        out[name] = {"acc": round(float(m.score(Xte, yte)), 4), "time": round(dt, 2)}
    return out

Xb, yb = load_breast_cancer(return_X_y=True)
Xs, ys = make_classification(30000, 50, n_informative=15, n_redundant=10, flip_y=0.05, random_state=0)
res = {"breast_cancer": run(Xb, yb), "synthetic_30000x50": run(Xs, ys)}
for k, v in res.items():
    print(k)
    for n, r in v.items(): print(f"  {n:14s} acc={r['acc']:.4f}  fit={r['time']:.2f}s")
json.dump(res, open("results.json", "w"), ensure_ascii=False, indent=1)
