# # import pandas as  pd
# # df_dataset= pd.read_csv('ENB2012_data.csv')
# # #For ENB2012_data.csv
# # print('#'*40,'For ENB2012_data.csv', '#'*40)
# # print(df_dataset.describe(include='all').to_string())
# # print(df_dataset.shape)
# # print(df_dataset.columns)
# # print(df_dataset.info)
# # print(df_dataset.dtypes)
# # print(df_dataset.isna().sum())
# # print(df_dataset.head(10).to_string())
# # print('='*90)
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# #
# # df = pd.read_csv('ENB2012_data.csv')
# #
# # # Distribution Y1 and Y2
# # plt.figure(figsize=(12, 5))
# # plt.subplot(1, 2, 1)
# # sns.histplot(df['Y1'], kde=True)
# # plt.title('Distribution of Heating Load (Y1)')
# # plt.subplot(1, 2, 2)
# # sns.histplot(df['Y2'], kde=True)
# # plt.title('Distribution of Cooling Load (Y2)')
# # plt.show()
# #
# # # Correlation of features
# # plt.figure(figsize=(10, 8))
# # sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# # plt.title('Correlation Heatmap')
# # plt.show()
# #
# # # Scatter plots For important features (based on high correlation)
# # plt.figure(figsize=(12, 10))
# # plt.subplot(2, 2, 1)
# # sns.scatterplot(x='X1', y='Y1', data=df)
# # plt.title('Relative Compactness vs Heating Load')
# # plt.subplot(2, 2, 2)
# # sns.scatterplot(x='X5', y='Y1', data=df)
# # plt.title('Overall Height vs Heating Load')
# # plt.subplot(2, 2, 3)
# # sns.scatterplot(x='X1', y='Y2', data=df)
# # plt.title('Relative Compactness vs Cooling Load')
# # plt.subplot(2, 2, 4)
# # sns.scatterplot(x='X5', y='Y2', data=df)
# # plt.title('Overall Height vs Cooling Load')
# # plt.show()
# #
# # # Boxplots to examine outliers and distributions based on categories (e.g. X6: Orientation)
# # plt.figure(figsize=(12, 5))
# # plt.subplot(1, 2, 1)
# # sns.boxplot(x='X6', y='Y1', data=df)
# # plt.title('Orientation vs Heating Load')
# # plt.subplot(1, 2, 2)
# # sns.boxplot(x='X6', y='Y2', data=df)
# # plt.title('Orientation vs Cooling Load')
# # plt.show()
# #
# # # Pairplot For an overview
# # sns.pairplot(df, vars=['X1', 'X2', 'X3', 'X4', 'X5', 'Y1', 'Y2'])
# # plt.show()
#
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
#
# # Loading the dataset
# df = pd.read_csv('ENB2012_data.csv')
#
# # Features and Target (for Y1 - heating load)
# X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
# y = df['Y1']
#
# #  train/test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # List Model
# models = {
#     'Linear Regression': LinearRegression(),
#     'Decision Tree': DecisionTreeRegressor(random_state=42),
#     'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
#     'Gradient Boosting': GradientBoostingRegressor(random_state=42),
#     'SVM Regression': SVR(),
#     'Neural Network': MLPRegressor(random_state=42, max_iter=1000)
# }
#
# # Training and evaluation
# results = []
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     r2 = r2_score(y_test, y_pred)
#     results.append([name, mae, rmse, r2])
#
# # Show in charts
# results_df = pd.DataFrame(results, columns=['Model', 'MAE', 'RMSE', 'R²'])
# print(results_df.sort_values(by='R²', ascending=False))
#
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# import joblib
#
# # load
# df = pd.read_csv('ENB2012_data.csv')
# X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
# y = df['Y1']  # یا Y2
#
# #  train/test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Random Forest
# model = RandomForestRegressor(random_state=42)
#
# # Feature Importance
# model.fit(X_train, y_train)
# importances = model.feature_importances_
# features = X.columns
# plt.figure(figsize=(10, 6))
# plt.barh(features, importances)
# plt.title('Feature Importance in Random Forest')
# plt.show()
#
# # Hyperparameter Tuning with GridSearchCV (5-fold CV)
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
# grid_search.fit(X_train, y_train)
#
# # best model
# best_model = grid_search.best_estimator_
# print(f'Best Parameters: {grid_search.best_params_}')
#
# #Evaluation with Cross-Validation (on the best model)
# cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
# print(f'Cross-Validation R² Scores: {cv_scores}')
# print(f'Mean CV R²: {cv_scores.mean():.4f}')
#
# # final test set
# y_pred = best_model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f'Final MAE: {mae:.4f}, R²: {r2:.4f}')
#
# # save Joblib
# joblib.dump(best_model, 'best_rf_model.joblib')
# print('Model saved as best_rf_model.joblib')
#
# # To load later: loaded model = joblib.load('best_rf_model.joblib')
#
# import streamlit as st
# import joblib
# import pandas as pd
#
# # loading
# model = joblib.load('best_rf_model.joblib')
#
# st.title('EnergyEfficiency Prediction')
# st.write('Enter building parameters to predict Heating Load (Y1)')
#
# # input
# x1 = st.number_input('Relative Compactness (X1)', min_value=0.62, max_value=0.98, value=0.76)
# x2 = st.number_input('Surface Area (X2)', min_value=514.5, max_value=808.5, value=671.7)
# x3 = st.number_input('Wall Area (X3)', min_value=245.0, max_value=416.5, value=318.5)
# x4 = st.number_input('Roof Area (X4)', min_value=110.25, max_value=220.5, value=176.6)
# x5 = st.number_input('Overall Height (X5)', min_value=3.5, max_value=7.0, value=5.25)
# x6 = st.number_input('Orientation (X6)', min_value=2, max_value=5, value=3, step=1)
# x7 = st.number_input('Glazing Area (X7)', min_value=0.0, max_value=0.4, value=0.23)
# x8 = st.number_input('Glazing Area Distribution (X8)', min_value=0, max_value=5, value=3, step=1)
#
# # Forecast
# if st.button('Predict'):
#     input_data = pd.DataFrame([[x1, x2, x3, x4, x5, x6, x7, x8]],
#                               columns=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])
#     prediction = model.predict(input_data)[0]
#     st.write(f'Predicted Heating Load (Y1): {prediction:.2f} kWh/m²')