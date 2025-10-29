
"""
<Project Title>
Capstone Group 03
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load & clean data
def load_and_clean(filepath):
    df = pd.read_csv(filepath)

    # 1) Drop duplicates
    df = df.drop_duplicates()

    # 2) Parse date and engineer calendar features
    #    (dates are like '20141013T000000' -> keep first 8 chars)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'].astype(str).str.slice(0, 8),
                                    format='%Y%m%d', errors='coerce')
        df['sale_year'] = df['date'].dt.year
        df['sale_month'] = df['date'].dt.month

    # 3) Compute house_age using actual sale year (fallback to max sale year or 2025)
    if 'yr_built' in df.columns:
        if 'sale_year' in df.columns and df['sale_year'].notna().any():
            fallback_year = int(df['sale_year'].dropna().max())
        else:
            fallback_year = 2025
        df['house_age'] = (fallback_year - df['yr_built']).clip(lower=0)

    # 4) Trim obvious extremes (document these in your report)
    if 'price' in df.columns:
        df = df[df['price'].between(75_000, 3_000_000)]
    if 'bedrooms' in df.columns:
        df = df[df['bedrooms'] <= 10]      # drop 33-bedroom anomaly
    if 'bathrooms' in df.columns:
        df = df[df['bathrooms'] <= 6]      # trim extreme baths
    if 'floors' in df.columns:
        df = df[df['floors'] <= 3.5]       # trim extreme floors

    # 5) Create a log target (useful for a linear model variant)
    if 'price' in df.columns and 'price_log' not in df.columns:
        df['price_log'] = np.log1p(df['price'])

    return df

# 2. Split data into train/test
def split_data(df, target, test_size=0.3, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 3. Evaluate model
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"RMSE": rmse, "R2": r2}

# 4. Cross-validation helper
def cross_validate_model(model, X_train, y_train, cv_splits=5, scoring="r2"):
    """Perform k-fold cross-validation and return mean ± std scores."""
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv)
    return {"CV_Mean": float(np.mean(cv_scores)), "CV_Std": float(np.std(cv_scores))}
# 5. Ridge Regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

def build_ridge_pipeline(alphas=None, cv=5):
    if alphas is None:
        alphas = np.logspace(-3, 3, 31)  # 0.001 .. 1000
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas, cv=cv))
    ])

def run_ridge_regression(df, target="price", test_size=0.3, random_state=42,
                         alphas=None, cv_splits=5, scoring="r2"):
    # 1) Split
    X_train, X_test, y_train, y_test = split_data(df, target, test_size, random_state)

    # 2) Numeric-only features
    used_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_num = X_train[used_features].copy()
    X_test_num  = X_test[used_features].copy()

    # 3) Fit
    model = build_ridge_pipeline(alphas=alphas, cv=cv_splits)
    model.fit(X_train_num, y_train)

    # 4) Metrics
    test_metrics = evaluate_model(model, X_test_num, y_test)
    cv_stats = cross_validate_model(model, X_train_num, y_train, cv_splits=cv_splits, scoring=scoring)

    # 5) Package
    return {
        "model": model,
        "alpha": float(model.named_steps["ridge"].alpha_),
        "used_features": used_features,
        "test_metrics": test_metrics,
        "cv_stats": cv_stats,
        "X_test": X_test_num,
        "y_test": y_test
    }

def run_ridge_regression_logtarget(df, price_col="price", log_col="price_log",
                                   test_size=0.3, random_state=42,
                                   alphas=None, cv_splits=5):
    if log_col not in df.columns:
        df = df.copy()
        df[log_col] = np.log1p(df[price_col])

    res = run_ridge_regression(df, target=log_col, test_size=test_size,
                               random_state=random_state, alphas=alphas,
                               cv_splits=cv_splits, scoring="r2")

    y_true_log = res["y_test"]
    y_pred_log = res["model"].predict(res["X_test"])
    y_true_price = np.expm1(y_true_log.to_numpy())
    y_pred_price = np.expm1(y_pred_log)

    rmse_price = mean_squared_error(y_true_price, y_pred_price, squared=False)
    r2_price   = r2_score(y_true_price, y_pred_price)

    res.update({"rmse_price": float(rmse_price), "r2_price": float(r2_price)})
    return res
