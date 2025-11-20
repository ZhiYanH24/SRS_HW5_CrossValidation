import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("mushroom_overload.csv")
X = df[['stem-height','stem-width']]
Y = df['cap-diameter']
X_train, X_test, Y_train, Y_test = train_test_split(test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
print(model.intercept_)
print(model.coef_)
Y_pred = model.predict(X_test)
predictions = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
print(predictions.head())
r2 = r2_score(Y_test, Y_pred)
print(f"R-squared score: {r2}")

from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
loo_errors = []
for train_idx, val_idx in loo.split(X):
    X_train_loo, X_val_loo = X.iloc[train_idx], X.iloc[val_idx]
    Y_train_loo, Y_val_loo = Y.iloc[train_idx], Y.iloc[val_idx]
    model_loo = LinearRegression()
    model_loo.fit(X_train_loo, Y_train_loo)
    pred = model_loo.predict(X_val_loo)
    loo_errors.append((Y_val_loo.values[0] - pred[0]) ** 2)
loo_mse = np.mean(loo_errors)
loo_rmse = np.sqrt(loo_mse)
print(f"LOOCV MSE: {loo_mse:.4f}")
print(f"LOOCV RMSE: {loo_rmse:.4f}")

from sklearn.model_selection import KFold

kf = KFold(n_splits = 24, shuffle = True, random_state = 42)
kf = LeaveOneOut()
kf_errors = []
for train_idx, val_idx in kf.split(X):
    X_train_kf, X_val_kf = X.iloc[train_idx], X.iloc[val_idx]
    Y_train_kf, Y_val_kf = Y.iloc[train_idx], Y.iloc[val_idx]
    model_kf = LinearRegression()
    model_kf.fit(X_train_kf, Y_train_kf)
    pred = model_kf.predict(X_val_kf)
    kf_errors.append(mean_squared_error(Y_val_kf, pred))
kf_mse = np.mean(kf_errors)
kf_rmse = np.sqrt(kf_mse)
print(f"kfCV MSE: {kf_mse:.4f}")
print(f"kfCV RMSE: {kf_rmse:.4f}")