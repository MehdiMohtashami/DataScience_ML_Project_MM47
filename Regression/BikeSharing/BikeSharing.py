# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
#
# # Load data
# hour_df = pd.read_csv('hour.csv')
# day_df = pd.read_csv('day.csv')
#
# # Initial data exploration
# print("Number of hourly data points:", hour_df.shape)
# print("Number of daily data points:", day_df.shape)
#
# # Set plot appearance
# plt.style.use('seaborn-v0_8')
# sns.set_palette("husl")
#
# # Distribution of rental bicycles
# fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#
# # Distribution of cnt
# sns.histplot(hour_df['cnt'], kde=True, ax=axes[0,0], color='blue')
# axes[0,0].set_title('Distribution of number of rental bicycles (hourly)')
#
# # Relationship between temperature and number of bicycles
# sns.scatterplot(x=hour_df['temp'], y=hour_df['cnt'], alpha=0.5, ax=axes[0,1], color='green')
# axes[0,1].set_title('Relationship between temperature and number of bicycles')
#
# # Number of bicycles by hour of the day
# sns.boxplot(x='hr', y='cnt', data=hour_df, ax=axes[1,0], color='yellow')
# axes[1,0].set_title('Number of bicycles by time of day')
#
# # Number of bicycles by season
# sns.boxplot(x='season', y='cnt', data=hour_df, ax=axes[1,1], color='Purple')
# axes[1,1].set_title('Number of bicycles by season')
#
# plt.tight_layout()
# plt.show()
#
# # Copy data for preprocessing
# df = hour_df.copy()
#
# # Convert dteday column to datetime
# df['dteday'] = pd.to_datetime(df['dteday'])
#
# # Extract new features from date
# df['year'] = df['dteday'].dt.year
# df['month'] = df['dteday'].dt.month
# df['day'] = df['dteday'].dt.day
#
# # Drop unnecessary columns
# df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1, inplace=True)
#
# # Check correlation
# plt.figure(figsize=(14, 10))
# corr = df.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
# plt.title('Feature correlation matrix')
# plt.show()
#
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# # Separate features and target variable
# X = df.drop('cnt', axis=1)
# y = df['cnt']
#
# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Standardize data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# import xgboost as xgb
#
# # Different models
# models = {
#     'Linear Regression': LinearRegression(),
#     'Ridge Regression': Ridge(alpha=1.0),
#     'Lasso Regression': Lasso(alpha=0.1),
#     'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#     'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
#     'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
#     'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
# }
#
# # Store results
# results = {}
#
# for name, model in models.items():
#     # Train model
#     model.fit(X_train_scaled, y_train)
#
#     # Predict
#     y_pred = model.predict(X_test_scaled)
#
#     # Calculate evaluation metrics
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#
#     # Store results
#     results[name] = {
#         'MSE': mse,
#         'RMSE': rmse,
#         'MAE': mae,
#         'R2': r2
#     }
#
#     print(f"{name}:")
#     print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}\n")
#
# # Compare results
# results_df = pd.DataFrame(results).T
# results_df = results_df.sort_values('R2', ascending=False)
# print(results_df)
#
# from sklearn.model_selection import GridSearchCV
#
# # Hyperparameter tuning for the best model
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }
#
# # Assume Random Forest is the best model
# best_model = RandomForestRegressor(random_state=42)
# grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
# grid_search.fit(X_train_scaled, y_train)
#
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)
#
# # Use the best model
# best_rf_model = grid_search.best_estimator_
# y_pred_best = best_rf_model.predict(X_test_scaled)
#
# # Evaluate final model
# final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
# final_r2 = r2_score(y_test, y_pred_best)
# print(f"Final model - RMSE: {final_rmse:.2f}, R2: {final_r2:.4f}")
#
# # Feature importance in the best model
# feature_importance = best_rf_model.feature_importances_
# feature_names = X.columns
#
# # Create DataFrame for feature importance
# importance_df = pd.DataFrame({
#     'feature': feature_names,
#     'importance': feature_importance
# }).sort_values('importance', ascending=False)
#
# # Plot feature importance
# plt.figure(figsize=(10, 8))
# sns.barplot(x='importance', y='feature', data=importance_df)
# plt.title('Feature importance in predicting the number of rental bicycles')
# plt.tight_layout()
# plt.show()

# import joblib
# import xgboost as xgb
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import pandas as pd
#
# # Load and preprocess data
# df = pd.read_csv('hour.csv')
# df['dteday'] = pd.to_datetime(df['dteday'])
# df['year'] = df['dteday'].dt.year
# df['month'] = df['dteday'].dt.month
# df['day'] = df['dteday'].dt.day
# df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1, inplace=True)
#
# # Separate features and target variable
# X = df.drop('cnt', axis=1)
# y = df['cnt']
#
# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Standardize data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Train XGBoost model (with best parameters)
# best_xgb_model = xgb.XGBRegressor(
#     n_estimators=100,
#     random_state=42,
#     max_depth=6,  # You can adjust these parameters
#     learning_rate=0.1
# )
# best_xgb_model.fit(X_train_scaled, y_train)
#
# # Save model and scaler
# joblib.dump(best_xgb_model, 'bike_sharing_xgboost_model.pkl')
# joblib.dump(scaler, 'bike_sharing_scaler.pkl')
#
# print("Model and scaler saved successfully!")
#
#
# # Code for loading and using the model in the future
# def load_and_predict(new_data):
#     # Load model and scaler
#     model = joblib.load('bike_sharing_xgboost_model.pkl')
#     scaler = joblib.load('bike_sharing_scaler.pkl')
#
#     # Preprocess new data (consistent with training data preprocessing)
#     new_data = new_data.copy()
#     new_data['dteday'] = pd.to_datetime(new_data['dteday'])
#     new_data['year'] = new_data['dteday'].dt.year
#     new_data['month'] = new_data['dteday'].dt.month
#     new_data['day'] = new_data['dteday'].dt.day
#     new_data.drop(['instant', 'dteday', 'casual', 'registered'], axis=1, inplace=True, errors='ignore')
#
#     # Standardize new data
#     new_data_scaled = scaler.transform(new_data)
#
#     # Predict
#     predictions = model.predict(new_data_scaled)
#     return predictions
#
#
# # Test loading and prediction
# try:
#     # Use a few samples from test data for testing
#     sample_data = X_test.head(3).copy()
#     predictions = load_and_predict(sample_data)
#     print("Sample predictions:", predictions)
#     print("Actual values:", y_test.head(3).values)
# except Exception as e:
#     print("Error in model testing:", e)