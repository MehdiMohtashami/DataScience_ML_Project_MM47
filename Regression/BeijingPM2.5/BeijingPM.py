# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from sklearn.impute import SimpleImputer
# import joblib
# import time
# from scipy import stats
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # 1. Load data
# print("Loading data...")
# data = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')
#
# # 2. Handle missing data
# print("Handling missing data...")
# data = data.dropna(subset=['pm2.5'])
# data = pd.get_dummies(data, columns=['cbwd'])
# data = data.drop(columns=['No'])
#
# # 3. Handle outliers with IQR method
# print("Handling outliers...")
#
#
# def handle_outliers(df, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#
#     df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
#     df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
#     return df
#
#
# # Apply to all numeric columns
# numeric_cols = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
# for col in numeric_cols:
#     data = handle_outliers(data, col)
#
# # 4. Better feature engineering
# print("Feature engineering...")
#
#
# def feature_engineering(df):
#     # Temporal features
#     df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
#     df['season'] = df['month'] % 12 // 3 + 1
#
#     # Interaction features
#     df['TEMP_DEWP'] = df['TEMP'] * df['DEWP']  # Interaction between temperature and dew point
#     df['PRES_Iws'] = df['PRES'] / df['Iws']  # Ratio of pressure to wind speed
#
#     # Statistical features
#     df['avg_last_3h'] = df['pm2.5'].rolling(window=3).mean().shift(1)  # Mean of the past 3 hours
#
#     return df.dropna()  # Remove rows created due to rolling calculations
#
#
# data = feature_engineering(data)
#
# # 5. Split data considering temporal nature
# print("Splitting data...")
# X = data.drop(columns=['pm2.5'])
# y = data['pm2.5']
#
# # Use 2014 as test data (most recent data)
# X_train = X[X['year'] < 2014]
# y_train = y[X['year'] < 2014]
# X_test = X[X['year'] == 2014]
# y_test = y[X['year'] == 2014]
#
# X_train = X_train.drop(columns=['year'])
# X_test = X_test.drop(columns=['year'])
#
# print(f"Number of training data points: {X_train.shape[0]}")
# print(f"Number of test data points: {X_test.shape[0]}")
#
# # 6. Define preprocessing pipeline
# print("Defining preprocessing pipeline...")
# numeric_features = ['month', 'day', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
#                     'hour_sin', 'hour_cos', 'season', 'TEMP_DEWP', 'PRES_Iws', 'avg_last_3h']
# categorical_features = list(set(X.columns) - set(numeric_features) - {'year'})
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='median')),
#             ('scaler', StandardScaler())]), numeric_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ])
#
# # 7. Define different models for comparison
# print("Defining models for comparison...")
# models = {
#     'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
#     'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1),
#     'LightGBM': LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=50, random_state=42, n_jobs=-1),
#     'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, random_state=42)
# }
#
# # 8. Train and evaluate models
# print("Training and evaluating models...")
# results = []
# best_model = None
# best_r2 = -np.inf
# model_pipelines = {}
#
# for name, model in models.items():
#     print(f"\nTraining model: {name}")
#     start_time = time.time()
#
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('model', model)
#     ])
#
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     training_time = time.time() - start_time
#
#     results.append({
#         'Model': name,
#         'RMSE': rmse,
#         'MAE': mae,
#         'R²': r2,
#         'Time (s)': training_time
#     })
#
#     # Store pipeline for later use
#     model_pipelines[name] = pipeline
#
#     if r2 > best_r2:
#         best_r2 = r2
#         best_model = pipeline
#         best_model_name = name
#
# # Display results
# results_df = pd.DataFrame(results)
# print("\nModel evaluation results:")
# print(results_df.sort_values(by='R²', ascending=False))
#
# # 9. Hyperparameter tuning for the best model (Gradient Boosting)
# if best_model_name == 'Gradient Boosting':
#     print("\nHyperparameter tuning for Gradient Boosting...")
#
#     # Parameters for tuning
#     param_grid = {
#         'model__n_estimators': [300, 500],
#         'model__learning_rate': [0.01, 0.05, 0.1],
#         'model__max_depth': [3, 5, 7],
#         'model__min_samples_split': [2, 5, 10],
#         'model__min_samples_leaf': [1, 2, 4]
#     }
#
#     # Use TimeSeriesSplit
#     tscv = TimeSeriesSplit(n_splits=3)  # Reduced to 3 for faster execution
#
#     grid_search = GridSearchCV(
#         estimator=best_model,
#         param_grid=param_grid,
#         cv=tscv,
#         scoring='r2',
#         n_jobs=-1,
#         verbose=1
#     )
#
#     grid_search.fit(X_train, y_train)
#
#     print(f"\nBest parameters: {grid_search.best_params_}")
#     print(f"Best R² score in validation: {grid_search.best_score_:.4f}")
#
#     # Update best model
#     best_model = grid_search.best_estimator_
#
#     # Final evaluation on test data
#     y_pred_final = best_model.predict(X_test)
#     final_r2 = r2_score(y_test, y_pred_final)
#     print(f"Final R² on test data: {final_r2:.4f}")

# 10. Save the best model
# print("\nSaving the best model...")
# joblib.dump(best_model, 'optimized_pm25_model.joblib')
# print(f"Best model ({best_model_name}) with accuracy {best_r2:.4f} saved.")
#
# # 11. Analyze feature importance
# print("\nAnalyzing feature importance...")
# if hasattr(best_model.named_steps['model'], 'feature_importances_'):
#     # Extract feature names
#     num_features = numeric_features
#     cat_features = list(
#         best_model.named_steps['preprocessor']
#         .named_transformers_['cat']
#         .get_feature_names_out(categorical_features)
#     )
#     feature_names = num_features + cat_features
#
#     importances = best_model.named_steps['model'].feature_importances_
#
#     # Create DataFrame for feature importance
#     feature_importance_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': importances
#     }).sort_values('Importance', ascending=False)
#
#     # Display top 15 important features
#     plt.figure(figsize=(14, 10))
#     sns.barplot(x='Importance', y='Feature',
#                 data=feature_importance_df.head(15),
#                 palette='viridis')
#     plt.title(f'Top 15 Features in {best_model_name} Model')
#     plt.tight_layout()
#     plt.savefig('feature_importances.png', dpi=300)
#     plt.show()
#
#     # Save feature importance table
#     feature_importance_df.to_csv('feature_importances.csv', index=False)
#
# # 12. Visual comparison of model performance
# print("\nCreating model comparison plot...")
# plt.figure(figsize=(12, 8))
# sns.barplot(x='R²', y='Model', data=results_df.sort_values('R²', ascending=True), palette='mako')
# plt.title('Model Accuracy Comparison (R²)')
# plt.xlim(0.8, 0.95)
# plt.xlabel('R² Score')
# plt.ylabel('Model')
# plt.tight_layout()
# plt.savefig('models_comparison.png', dpi=300)
# plt.show()
#
# # 13. Sample predictions and comparison with actual values
# print("\nSample predictions...")
# sample_data = X_test.sample(10, random_state=42)
# actual_values = y_test.loc[sample_data.index]
#
# # Predict with the best model
# predicted_values = best_model.predict(sample_data)
#
# # Predict with other models for comparison
# comparison_df = pd.DataFrame({
#     'Actual': actual_values,
#     best_model_name: predicted_values
# })
#
# for model_name, pipeline in model_pipelines.items():
#     if model_name != best_model_name:
#         comparison_df[model_name] = pipeline.predict(sample_data)
#
# print("\nModel prediction comparison for 10 random samples:")
# print(comparison_df)
#
# # 14. Save all models for later use
# print("\nSaving all models...")
# for model_name, pipeline in model_pipelines.items():
#     joblib.dump(pipeline, f'{model_name.replace(" ", "_")}_model.joblib')
# print("All models saved successfully.")
#
# # 15. Final report
# print("\nFinal report:")
# print(f"- Best model: {best_model_name} with R² = {best_r2:.4f}")
# print(f"- Number of features: {len(feature_names)}")
# print(f"- Number of training data points: {X_train.shape[0]}")
# print(f"- Number of test data points: {X_test.shape[0]}")
# print(f"- Most important feature: {feature_importance_df.iloc[0]['Feature']}")
#
# print("\nProject completed successfully!")