# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from joblib import dump
#
# # 1. Load data
# data = pd.read_csv('Daily_Demand_Forecasting_Orders.csv')  # Specify your file path
#
# # 2. Exploratory Data Analysis (EDA)
# def perform_eda(df):
#     print("Initial dataset information:")
#     print(df.info())
#
#     print("\nDescriptive statistics:")
#     print(df.describe())
#     print()
#
#     # Target variable distribution
#     plt.figure(figsize=(10, 6))
#     sns.histplot(df['Target (Total orders)'], kde=True)
#     plt.title('Target variable distribution (Total Orders)')
#     plt.savefig('TotalOrders.png')
#     plt.show()
#
#     # Correlation matrix
#     plt.figure(figsize=(12, 10))
#     corr_matrix = df.corr()
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
#     plt.title('Feature correlation matrix')
#     plt.savefig('correlation-matrix.png')
#     plt.show()
#
#     # Explore relationship between features and target variable
#     plt.figure(figsize=(15, 10))
#     for i, column in enumerate(df.columns[:-1], 1):
#         plt.subplot(3, 4, i)
#         sns.scatterplot(x=df[column], y=df['Target (Total orders)'])
#         plt.title(f'{column} vs Target')
#     plt.tight_layout()
#     plt.savefig('MultiTO.png')
#     plt.show()
#
# perform_eda(data)
#
# # 3. Preprocess data
# X = data.drop('Target (Total orders)', axis=1)
# y = data['Target (Total orders)']
#
# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# # Standardize features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # 4. Implement and evaluate different models
# models = {
#     'Linear Regression': LinearRegression(),
#     'Ridge Regression': Ridge(alpha=1.0),
#     'Lasso Regression': Lasso(alpha=0.1),
#     'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
#     'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
#     'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
#     'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#     'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
# }
#
# results = []
# for name, model in models.items():
#     # Train model
#     model.fit(X_train_scaled, y_train)
#
#     # Predict
#     y_pred = model.predict(X_test_scaled)
#
#     # Evaluate
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#
#     # Cross-validation
#     cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
#     cv_r2 = np.mean(cv_scores)
#
#     results.append({
#         'Model': name,
#         'R2_Score': r2,
#         'MAE': mae,
#         'MSE': mse,
#         'CV_R2': cv_r2
#     })
#
# # 5. Compare models
# results_df = pd.DataFrame(results).sort_values(by='R2_Score', ascending=False)
# print("\nModel evaluation results:")
# print(results_df)
#
# # 6. Select best model
# best_model_name = results_df.iloc[0]['Model']
# best_model = models[best_model_name]
# print(f"\nBest model: {best_model_name}")
#
# # 7. Save model with joblib
# dump(best_model, 'best_regression_model.joblib')
# dump(scaler, 'scaler.joblib')
# print("Model and scaler saved successfully.")
#
# # 8. Analyze results and visualize
# plt.figure(figsize=(12, 8))
# sns.barplot(x='R2_Score', y='Model', data=results_df, palette='viridis')
# plt.xlabel('R² Score')
# plt.ylabel('Model')
# plt.title('Comparing the performance of regression models')
# plt.xlim(0, 1)
# plt.savefig('Reg-Compering.png')
# plt.show()