import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import warnings

warnings.filterwarnings('ignore')

# خواندن داده
try:
    abalone_df = pd.read_csv('abalone.data.csv')
except:
    abalone_df = pd.read_csv('abalone.data.csv', header=None)
    abalone_df.columns = [
        "Sex", "Length", "Diameter", "Height",
        "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"
    ]

print("توزیع متغیر هدف (Rings):")
print(abalone_df["Rings"].value_counts().sort_index())

# مهندسی ویژگی‌های پیشرفته
abalone_df['Weight_ratio'] = abalone_df['Whole weight'] / (abalone_df['Shucked weight'] + 1e-6)
abalone_df['Volume_approx'] = abalone_df['Length'] * abalone_df['Diameter'] * abalone_df['Height']
abalone_df['Density'] = abalone_df['Whole weight'] / (abalone_df['Volume_approx'] + 1e-6)
abalone_df['Shell_ratio'] = abalone_df['Shell weight'] / (abalone_df['Whole weight'] + 1e-6)

# ایجاد ویژگی‌های چندجمله‌ای درجه دوم
numeric_cols = ["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
                "Viscera weight", "Shell weight"]
for col in numeric_cols:
    abalone_df[f'{col}_squared'] = abalone_df[col] ** 2

# آماده‌سازی داده
X = abalone_df.drop(columns=["Rings"])
y = abalone_df["Rings"]

# تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=abalone_df["Sex"])

# پیش‌پردازش
cat_features = ["Sex"]
num_features = [c for c in X.columns if c not in cat_features]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), cat_features),
        ("num", StandardScaler(), num_features),
    ]
)


# تابع ارزیابی مدل
def evaluate_model(model, X_tr, y_tr, X_te, y_te, label="model", cv_eval=False):
    if cv_eval:
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='r2')
        print(f"CV R² Scores: {[f'{s:.3f}' for s in cv_scores]}")
        print(f"CV R² Mean: {cv_scores.mean():.3f}")

    model.fit(X_tr, y_tr)
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)

    r2_tr = r2_score(y_tr, y_pred_tr)
    r2_te = r2_score(y_te, y_pred_te)
    mae_te = mean_absolute_error(y_te, y_pred_te)
    rmse_te = np.sqrt(mean_squared_error(y_te, y_pred_te))

    print(f"=== {label} ===")
    print(f"Train R²: {r2_tr:.3f} ({r2_tr * 100:.1f}%)")
    print(f"Test R²: {r2_te:.3f} ({r2_te * 100:.1f}%)")
    print(f"Test MAE: {mae_te:.3f}")
    print(f"Test RMSE: {rmse_te:.3f}\n")

    return {"r2_te": r2_te, "mae_te": mae_te, "rmse_te": rmse_te, "model": model}


# 1. Gradient Boosting با تنظیم‌های ضد اورفیت
gb = Pipeline([
    ("prep", preprocess),
    ("model", GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_split=10,
        random_state=42
    ))
])
res_gb = evaluate_model(gb, X_train, y_train, X_test, y_test, "GradientBoosting", cv_eval=True)

# 2. XGBoost با تنظیم‌های ضد اورفیت
xgb = Pipeline([
    ("prep", preprocess),
    ("model", XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    ))
])
res_xgb = evaluate_model(xgb, X_train, y_train, X_test, y_test, "XGBoost (Regularized)")

# 3. RandomForest با تنظیم‌های ضد اورفیت
rf = Pipeline([
    ("prep", preprocess),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=0.7,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ))
])
res_rf = evaluate_model(rf, X_train, y_train, X_test, y_test, "RandomForest (Regularized)")

# 4. Ensemble از بهترین مدل‌ها
from sklearn.ensemble import VotingRegressor

# ایجاد یک ensemble از مدل‌های مختلف
ensemble = Pipeline([
    ("prep", preprocess),
    ("model", VotingRegressor([
        ('gb', GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                                         subsample=0.8, min_samples_split=10, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=10,
                                     min_samples_leaf=4, max_features=0.7, random_state=42))
    ]))
])
res_ensemble = evaluate_model(ensemble, X_train, y_train, X_test, y_test, "Ensemble")

# مقایسه مدل‌ها
print("=" * 50)
print("مقایسه نهایی مدل‌ها:")
print("=" * 50)

results = [
    {"Model": "GradientBoosting", "R²": f"{res_gb['r2_te']:.3f}", "R² (%)": f"{res_gb['r2_te'] * 100:.1f}%",
     "MAE": f"{res_gb['mae_te']:.3f}", "RMSE": f"{res_gb['rmse_te']:.3f}"},
    {"Model": "XGBoost", "R²": f"{res_xgb['r2_te']:.3f}", "R² (%)": f"{res_xgb['r2_te'] * 100:.1f}%",
     "MAE": f"{res_xgb['mae_te']:.3f}", "RMSE": f"{res_xgb['rmse_te']:.3f}"},
    {"Model": "RandomForest", "R²": f"{res_rf['r2_te']:.3f}", "R² (%)": f"{res_rf['r2_te'] * 100:.1f}%",
     "MAE": f"{res_rf['mae_te']:.3f}", "RMSE": f"{res_rf['rmse_te']:.3f}"},
    {"Model": "Ensemble", "R²": f"{res_ensemble['r2_te']:.3f}", "R² (%)": f"{res_ensemble['r2_te'] * 100:.1f}%",
     "MAE": f"{res_ensemble['mae_te']:.3f}", "RMSE": f"{res_ensemble['rmse_te']:.3f}"}
]

results_df = pd.DataFrame(results)
print(results_df.sort_values("R²", ascending=False))

# انتخاب بهترین مدل
best_model_name = results_df.sort_values("R²", ascending=False).iloc[0]["Model"]
best_model = None

if best_model_name == "GradientBoosting":
    best_model = gb
elif best_model_name == "XGBoost":
    best_model = xgb
elif best_model_name == "RandomForest":
    best_model = rf
else:
    best_model = ensemble

print(f"\nبهترین مدل: {best_model_name}")

# آنالیز خطاها
y_pred = best_model.predict(X_test)
residuals = y_test - y_pred

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')

plt.tight_layout()
plt.show()

# بررسی اهمیت ویژگی‌ها
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    try:
        feature_names = best_model.named_steps['prep'].get_feature_names_out()
        importances = best_model.named_steps['model'].feature_importances_

        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp.head(15), x='importance', y='feature')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"نمایش اهمیت ویژگی‌ها امکان‌پذیر نیست: {e}")

# ذخیره بهترین مدل
# import joblib
#
# joblib.dump(best_model, "improved_abalone_model.joblib")
# print("مدل بهبود یافته ذخیره شد.")