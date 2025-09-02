# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import mean_absolute_error, mean_squared_error
# # from sklearn.linear_model import LinearRegression, Ridge, Lasso
# # from sklearn.tree import DecisionTreeRegressor
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# # from sklearn.pipeline import make_pipeline
# # from sklearn.cluster import KMeans
# # from sklearn.metrics import silhouette_score
# #
# # # 1. Load data
# # try:
# #     df = pd.read_csv('household_power_consumption.txt', sep=',', na_values='?')
# # except FileNotFoundError:
# #     print("File not found! Ensure the file path is correct.")
# #     exit()
# #
# # # Check columns and data
# # print("Dataset columns:")
# # print(df.columns)
# # print("\nFirst few rows of dataset:")
# # print(df.head(10))
# #
# # # Check for Date and Time columns
# # if 'Date' not in df.columns or 'Time' not in df.columns:
# #     print("Error: 'Date' or 'Time' columns not found!")
# #     exit()
# #
# # # 2. Fill NaNs in original data
# # df.fillna(df.mean(numeric_only=True), inplace=True)
# #
# # # 3. Create Datetime column
# # df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
# # df.set_index('Datetime', inplace=True)
# # df.drop(['Date', 'Time'], axis=1, inplace=True)
# #
# # # 4. Add Unmetered power
# # df['Unmetered'] = (df['Global_active_power'] * 1000 / 60) - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']
# #
# # # 5. Sampling (optional)
# # df = df.sample(frac=0.1, random_state=42)
# #
# # # 6. Resample to hourly
# # df_hourly = df.resample('h').mean()
# #
# # # 7. Fill NaNs in hourly data
# # df_hourly = df_hourly.ffill()  # Fix warning
# #
# # # 8. Add lag feature
# # df_hourly['Lag_1'] = df_hourly['Global_active_power'].shift(1)  # Power from previous hour
# # df_hourly = df_hourly.dropna()  # Remove NaNs from lag
# #
# # # Check hourly data
# # print("\nHourly data:")
# # print(df_hourly.head())
# # print("\nHourly data shape:", df_hourly.shape)
# # print("\nNaNs in hourly data:")
# # print(df_hourly.isna().sum())
# #
# # # 9. EDA: Plots
# # # Distributions
# # df_hourly.hist(bins=50, figsize=(12, 10))
# # plt.suptitle('Distributions of Features')
# # plt.show()
# #
# # # Correlation matrix
# # plt.figure(figsize=(10, 8))
# # sns.heatmap(df_hourly.corr(), annot=True, cmap='coolwarm')
# # plt.title('Correlation Matrix')
# # plt.show()
# #
# # # Time series plot for Global_active_power
# # plt.figure(figsize=(12, 6))
# # plt.plot(df_hourly.index, df_hourly['Global_active_power'], label='Global Active Power')
# # plt.xlabel('Time')
# # plt.ylabel('Power (kW)')
# # plt.title('Hourly Global Active Power Over Time')
# # plt.legend()
# # plt.show()
# #
# # # Scatter plot: Lag_1 vs Global_active_power
# # plt.figure(figsize=(8, 6))
# # sns.scatterplot(x='Lag_1', y='Global_active_power', data=df_hourly)
# # plt.title('Scatter: Lag_1 vs Active Power')
# # plt.show()
# #
# # # Boxplot for sub-metering
# # plt.figure(figsize=(10, 6))
# # sns.boxplot(data=df_hourly[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']])
# # plt.title('Boxplot of Sub-Meterings')
# # plt.show()
# #
# # # 10. Regression: Prepare data
# # # Drop Global_intensity and use Lag_1
# # X = df_hourly.drop(['Global_active_power', 'Global_intensity'], axis=1)
# # y = df_hourly['Global_active_power']
# #
# # # Scale features
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)
# # X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
# #
# # # Split chronological
# # split = int(0.8 * len(df_hourly))
# # X_train, X_test = X.iloc[:split], X.iloc[split:]
# # y_train, y_test = y.iloc[:split], y.iloc[split:]
# #
# # # Models
# # models = {
# #     'Linear Regression': LinearRegression(),
# #     'Polynomial Regression (deg=2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
# #     'Ridge': Ridge(alpha=1.0),
# #     'Lasso': Lasso(alpha=0.1, max_iter=10000),  # Increase max_iter for Lasso
# #     'Decision Tree': DecisionTreeRegressor(max_depth=5),
# #     'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
# # }
# #
# # # Test models
# # results = {}
# # for name, model in models.items():
# #     model.fit(X_train, y_train)
# #     y_pred = model.predict(X_test)
# #     mae = mean_absolute_error(y_test, y_pred)
# #     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# #     results[name] = {'MAE': mae, 'RMSE': rmse}
# #     print(f'{name}: MAE={mae:.4f}, RMSE={rmse:.4f}')
# #
# # # Find best model
# # best_model = min(results, key=lambda k: results[k]['RMSE'])
# # print(f'\nBest model: {best_model} with RMSE={results[best_model]["RMSE"]:.4f}')
# #
# # # 11. Clustering
# # features = df_hourly[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
# # scores = []
# # for k in range(2, 6):
# #     kmeans = KMeans(n_clusters=k, random_state=42)
# #     labels = kmeans.fit_predict(features)
# #     score = silhouette_score(features, labels)
# #     scores.append(score)
# #     print(f'K={k}, Silhouette Score={score:.4f}')
# #
# # best_k = np.argmax(scores) + 2
# # print(f'Best K: {best_k}')
# #
# # # Plot clusters
# # kmeans = KMeans(n_clusters=best_k, random_state=42)
# # df_hourly['Cluster'] = kmeans.fit_predict(features)
# # plt.figure(figsize=(12, 6))
# # sns.scatterplot(x=df_hourly.index, y='Global_active_power', hue='Cluster', data=df_hourly, palette='viridis')
# # plt.title('Clusters in Time Series')
# # plt.show()
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import joblib
#
# # 1. Load data
# try:
#     df = pd.read_csv('household_power_consumption.txt', sep=',', na_values='?')
# except FileNotFoundError:
#     print("File not found! Ensure the file path is correct.")
#     exit()
#
# # Check columns and data
# print("Dataset columns:")
# print(df.columns)
# print("\nFirst few rows of dataset:")
# print(df.head(10))
#
# # 2. Fill NaNs in original data
# df.fillna(df.mean(numeric_only=True), inplace=True)
#
# # 3. Create Datetime column
# df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
# df.set_index('Datetime', inplace=True)
# df.drop(['Date', 'Time'], axis=1, inplace=True)
#
# # 4. Add Unmetered power
# df['Unmetered'] = (df['Global_active_power'] * 1000 / 60) - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']
#
# # 5. Sampling (optional)
# df = df.sample(frac=0.1, random_state=42)
#
# # 6. Resample to hourly
# df_hourly = df.resample('h').mean()
#
# # 7. Fill NaNs in hourly data
# df_hourly = df_hourly.ffill()
#
# # 8. Add Lag_1
# df_hourly['Lag_1'] = df_hourly['Global_active_power'].shift(1)
# df_hourly = df_hourly.dropna()
#
# # 9. Clustering with K=4
# features = df_hourly[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
# kmeans = KMeans(n_clusters=4, random_state=42)
# df_hourly['Cluster'] = kmeans.fit_predict(features)
#
# # Save KMeans model
# joblib.dump(kmeans, 'kmeans_model.joblib')
# print("KMeans model with K=4 saved.")
#
# # Analyze clusters
# print("\nMean features in each cluster:")
# print(df_hourly.groupby('Cluster')[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean())
#
# # Plot clusters
# plt.figure(figsize=(12, 6))
# sns.scatterplot(x=df_hourly.index, y='Global_active_power', hue='Cluster', data=df_hourly, palette='viridis')
# plt.title('Clusters in Time Series (K=4)')
# plt.show()
#
# # Feature importance for clustering (examine variance of features in clusters)
# print("\nFeature variance in clusters:")
# for col in features.columns:
#     print(f"{col}: {df_hourly.groupby('Cluster')[col].var()}")
#
# # 10. Regression (excluding Unmetered and Lag_1 for more realistic results)
# X = df_hourly.drop(['Global_active_power', 'Global_intensity', 'Unmetered', 'Lag_1', 'Cluster'], axis=1)
# y = df_hourly['Global_active_power']
#
# # Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
#
# # Split chronological
# split = int(0.8 * len(df_hourly))
# X_train, X_test = X.iloc[:split], X.iloc[split:]
# y_train, y_test = y.iloc[:split], y.iloc[split:]
#
# # Models
# models = {
#     'Linear Regression': LinearRegression(),
#     'Ridge': Ridge(alpha=1.0),
#     'Lasso': Lasso(alpha=0.1, max_iter=10000),
#     'Decision Tree': DecisionTreeRegressor(max_depth=5),
#     'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
# }
#
# # Test models with Cross-validation
# results = {}
# for name, model in models.items():
#     # CV
#     scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
#     cv_rmse = -scores.mean()
#     # Train/test
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     results[name] = {'MAE': mae, 'RMSE': rmse, 'CV_RMSE': cv_rmse}
#     print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, CV_RMSE={cv_rmse:.4f}")
#
# # Find best model
# best_model = min(results, key=lambda k: results[k]['CV_RMSE'])
# print(f"\nBest model (based on CV_RMSE): {best_model} with CV_RMSE={results[best_model]['CV_RMSE']:.4f}")
#
# # Feature importance for Random Forest
# rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
# rf.fit(X_train, y_train)
# print("\nFeature importance for Random Forest:")
# for feature, importance in zip(X.columns, rf.feature_importances_):
#     print(f"{feature}: {importance:.4f}")
#
# # Save Random Forest model (or selected model)
# joblib.dump(rf, 'random_forest_model.joblib')
# print("Random Forest model saved.")