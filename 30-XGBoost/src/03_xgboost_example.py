"""Базовый пример: XGBoost на breast_cancer с early stopping и важностью признаков."""
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.2, random_state=0, stratify=ytr)

model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3,
                          reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                          early_stopping_rounds=30, eval_metric="logloss", random_state=0)
model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)

print("лучшая итерация:", model.best_iteration)
print("accuracy: %.4f" % accuracy_score(yte, model.predict(Xte)))
print("ROC-AUC : %.4f" % roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
imp = sorted(zip(X.columns, model.feature_importances_), key=lambda p: -p[1])[:5]
for n, v in imp: print(f"{n:25s} {v:.3f}")
