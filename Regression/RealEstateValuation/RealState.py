# import pandas as pd
import seaborn as sns
import missingno as ms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# df = pd.read_csv('../../files/RealStateHouse/Real estate valuation data set.csv')

# print(df.head())
# print(df.columns)
# print(df.describe(include='all').to_string())
# print(df.info)
# print(df.isna().sum())
# print(df.shape)

# Histogram of all columns
# df.hist(figsize=(12, 10))
# plt.tight_layout()
# plt.show()
#
# # Correlation matrix
# corr = df.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()
#
# # Explore relationship between features and target variable
# sns.pairplot(df, y_vars=['Y house price of unit area'],
#              x_vars=df.columns.drop('Y house price of unit area'))
# plt.show()
#
# # Separate features and target variable
# X = df.drop('Y house price of unit area', axis=1)
# y = df['Y house price of unit area']
#
# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# # Standardize data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# models = {
#     'Linear Regression': LinearRegression(),
#     'Ridge': Ridge(alpha=1.0),
#     'Lasso': Lasso(alpha=0.1),
#     'Decision Tree': DecisionTreeRegressor(random_state=42),
#     'Random Forest': RandomForestRegressor(random_state=42),
#     'Gradient Boosting': GradientBoostingRegressor(random_state=42),
#     'XGBoost': XGBRegressor(random_state=42)
# }
#
# results = {}
# for name, model in models.items():
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
#
#     r2 = r2_score(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#
#     results[name] = {'R2': r2, 'RMSE': rmse}
#
#     print(f"{name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
#
# # Compare models
# results_df = pd.DataFrame(results).T
# results_df.sort_values(by='R2', ascending=False, inplace=True)
# print(results_df)
#
# # Best model based on R²
# best_model_name = results_df.index[0]
# best_model = models[best_model_name]
#
# # Hyperparameter tuning (example for Random Forest)
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint
#
# param_dist = {
#     'n_estimators': randint(100, 500),
#     'max_depth': randint(5, 30),
#     'min_samples_split': randint(2, 20),
#     'min_samples_leaf': randint(1, 10)
# }
#
# random_search = RandomizedSearchCV(
#     RandomForestRegressor(random_state=42),
#     param_distributions=param_dist,
#     n_iter=100,
#     cv=5,
#     random_state=42,
#     scoring='r2'
# )
# random_search.fit(X_train_scaled, y_train)
#
# best_model = random_search.best_estimator_
# print(f"Best parameters: {random_search.best_params_}")
# print(f"Best R²: {random_search.best_score_:.4f}")
#
#
# best_rf = RandomForestRegressor(
#     n_estimators=439,
#     max_depth=13,
#     min_samples_split=3,
#     min_samples_leaf=5,
#     random_state=42
# )
# best_rf.fit(X_train_scaled, y_train)
#
# # Evaluate on test data
# y_pred_tuned = best_rf.predict(X_test_scaled)
# r2_tuned = r2_score(y_test, y_pred_tuned)
# print(f"Tuned Random Forest R² on Test Set: {r2_tuned:.4f}")
#
# feature_importances = pd.Series(
#     best_rf.feature_importances_,
#     index=X.columns
# ).sort_values(ascending=False)
#
# print("Feature Importances:")
# print(feature_importances)
#
# # Visualize feature importance
# plt.figure(figsize=(10, 6))
# feature_importances.plot(kind='barh')
# plt.title('Feature Importances in Random Forest Model')
# plt.show()

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Separate features and target variable
X = df.drop('Y house price of unit area', axis=1)
y = df['Y house price of unit area']

# Split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize data (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model (default)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(rf_model, 'real_estate_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score

# Load model (optional - not needed if running in the same session)
rf_model = joblib.load('real_estate_rf_model.pkl')

# Predict on test data
y_pred = rf_model.predict(X_test_scaled)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred)**2))

print(f"Random Forest Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# 1. Model comparison plot (based on previous results)
models_comparison = {
    'Linear Regression': 0.6811,
    'Ridge': 0.6814,
    'Lasso': 0.6838,
    'Decision Tree': 0.6038,
    'Random Forest': 0.8059,
    'Gradient Boosting': 0.7964,
    'XGBoost': 0.7688
}

# Create DataFrame for visualization
results_df = pd.DataFrame(list(models_comparison.items()), columns=['Model', 'R2'])
results_df = results_df.sort_values('R2', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='R2', y='Model', data=results_df, palette='viridis')
plt.title('Comparison of Regression Models by R² Score', fontsize=14)
plt.xlabel('R² Score', fontsize=12)
plt.ylabel('')
plt.xlim(0, 1)
plt.axvline(x=0.8, color='red', linestyle='--', alpha=0.5)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('models_comparison.png', dpi=300)
plt.show()

# 2. Actual vs Predicted plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor=None)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.fill_between([y_test.min(), y_test.max()],
                 [y_test.min()-rmse, y_test.max()-rmse],
                 [y_test.min()+rmse, y_test.max()+rmse],
                 color='orange', alpha=0.1, label=f'±RMSE ({rmse:.2f})')
plt.title('Actual vs Predicted House Prices (Random Forest)', fontsize=14)
plt.xlabel('Actual Prices', fontsize=12)
plt.ylabel('Predicted Prices', fontsize=12)
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
plt.show()

# 3. Feature importance plot
feature_importances = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh', color='teal')
plt.title('Feature Importances in Random Forest Model', fontsize=14)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300)
plt.show()

# 4. Residual analysis plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Analysis', fontsize=14)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300)
plt.show()

# Load model and scaler
model = joblib.load('real_estate_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# New data (example)
new_data = pd.DataFrame({
    'X1 transaction date': [2013.333],
    'X2 house age': [10.5],
    'X3 distance to the nearest MRT station': [200.0],
    'X4 number of convenience stores': [5],
    'X5 latitude': [24.98],
    'X6 longitude': [121.54]
})

# Preprocess new data
new_data_scaled = scaler.transform(new_data)

# Predict price
predicted_price = model.predict(new_data_scaled)
print(f"Predicted house price: {predicted_price[0]:.2f} (10,000 TWD/Ping)")