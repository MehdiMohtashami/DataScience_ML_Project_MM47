# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

# 2. Load and explore data
df = pd.read_csv('Concrete_Data.csv')

print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)

# 3. بررسی نام ستون‌ها و اصلاح اگر لازم باشد
# حذف فاصله‌های اضافی از نام ستون‌ها
df.columns = df.columns.str.strip()
print("\nColumn Names after stripping:", df.columns.tolist())

# 3. Basic EDA with plots
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

for i, col in enumerate(df.columns[:-1]):  # همه به جز تارگت
    axes[i].scatter(df[col], df['Concrete compressive strength'], alpha=0.6)
    axes[i].set_title(f'{col} vs Strength')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Strength')

plt.tight_layout()
plt.savefig('All_Aspect.png')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig('Correlation_Matrix.png')
plt.show()

# 4. Preprocessing
X = df.drop('Concrete compressive strength', axis=1)
y = df['Concrete compressive strength']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'SVR': SVR(),
    'K-Neighbors': KNeighborsRegressor()
}

# 6. Train and evaluate models
results = {}
for name, model in models.items():
    # Train model
    if name in ['SVR', 'K-Neighbors', 'Ridge', 'Lasso', 'ElasticNet']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {
        'R2 Score': r2,
        'MSE': mse,
        'MAE': mae,
        'Model': model
    }

# 7. Compare results
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.sort_values(by='R2 Score', ascending=False, inplace=True)
print(results_df[['R2 Score', 'MSE', 'MAE']])

# 8. Visualize model comparison
plt.figure(figsize=(12, 6))
results_df['R2 Score'].plot(kind='bar', color='skyblue')
plt.title('R2 Score Comparison of Regression Models')
plt.ylabel('R2 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Score_Comparison_Regression_Models.png')
plt.show()

# 9. Save best model
best_model_name = results_df.index[0]
best_model = results[best_model_name]['Model']
print(best_model)
# اگر مدل برتر نیاز به اسکیل دیتا داشته، باید اسکیلر رو هم سیو کنیم
# if best_model_name in ['SVR', 'K-Neighbors', 'Ridge', 'Lasso', 'ElasticNet','XGBoost']:
#     joblib.dump(best_model, 'best_model_scaled.pkl')
#     joblib.dump(scaler, 'scaler.pkl')
# else:
#     joblib.dump(best_model, 'best_model.pkl')

print(f"\nBest model: {best_model_name}")
print(f"R2 Score: {results_df['R2 Score'].iloc[0]:.4f}")

# 10. Cross-validation برای مدل برتر
if best_model_name in ['SVR', 'K-Neighbors', 'Ridge', 'Lasso', 'ElasticNet', 'XGBoost']:
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
else:
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')

print(f"\nCross-validation R2 scores for {best_model_name}: {cv_scores}")
print(f"Mean CV R2 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")