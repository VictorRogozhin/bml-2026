"""Пошаговый численный пример одного шага XGBoost (вручную) + проверка библиотекой."""
import numpy as np
import xgboost as xgb

x = np.array([1, 2, 3, 4.]); y = np.array([1, 2, 6, 7.])
lam, gamma, eta = 1.0, 0.0, 0.5

pred = np.full(4, y.mean())            # F0 = среднее = 4
g, h = pred - y, np.ones(4)            # l = 1/2 (y - F)^2  ->  g = F - y, h = 1
print("F0 =", pred, "\ng  =", g, "\nh  =", h)

def score(G, H): return G**2 / (H + lam)
best = None
for t in (1.5, 2.5, 3.5):              # кандидаты порогов
    L = x < t
    GL, HL, GR, HR = g[L].sum(), h[L].sum(), g[~L].sum(), h[~L].sum()
    gain = 0.5 * (score(GL, HL) + score(GR, HR) - score(GL + GR, HL + HR)) - gamma
    print(f"x < {t}: GL={GL:+.0f} HL={HL:.0f} GR={GR:+.0f} HR={HR:.0f} Gain={gain:.3f}")
    if best is None or gain > best[0]: best = (gain, t, GL, HL, GR, HR)

_, t, GL, HL, GR, HR = best
wL, wR = -GL / (HL + lam), -GR / (HR + lam)
print(f"\nлучший порог {t}; w_L={wL:.3f}, w_R={wR:.3f}")
new = pred + eta * np.where(x < t, wL, wR)
print("F1 =", new.round(3))
print("MSE: %.3f -> %.3f" % (((y - pred)**2).mean(), ((y - new)**2).mean()))

# проверка библиотекой
m = xgb.XGBRegressor(n_estimators=1, max_depth=1, learning_rate=eta, reg_lambda=lam,
                     min_child_weight=0, base_score=y.mean(), tree_method="exact")
m.fit(x.reshape(-1, 1), y)
print("xgboost F1 =", m.predict(x.reshape(-1, 1)).round(3))
