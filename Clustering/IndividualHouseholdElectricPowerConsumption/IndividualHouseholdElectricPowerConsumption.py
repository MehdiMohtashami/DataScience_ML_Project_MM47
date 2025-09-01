# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
#
# # ۱. لود داده
# try:
#     df = pd.read_csv('household_power_consumption.txt', sep=',', na_values='?')
# except FileNotFoundError:
#     print("فایل پیدا نشد! مطمئن شو که مسیر فایل درست باشه.")
#     exit()
#
# # چک کردن ستون‌ها و داده‌ها
# print("ستون‌های دیتاست:")
# print(df.columns)
# print("\nچند ردیف اول دیتاست:")
# print(df.head(10))
#
# # بررسی وجود ستون‌های Date و Time
# if 'Date' not in df.columns or 'Time' not in df.columns:
#     print("خطا: ستون‌های 'Date' یا 'Time' پیدا نشدن!")
#     exit()
#
# # ۲. پر کردن NaNهای داده اصلی
# df.fillna(df.mean(numeric_only=True), inplace=True)
#
# # ۳. ساخت ستون Datetime
# df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
# df.set_index('Datetime', inplace=True)
# df.drop(['Date', 'Time'], axis=1, inplace=True)
#
# # ۴. اضافه کردن Unmetered power
# df['Unmetered'] = (df['Global_active_power'] * 1000 / 60) - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']
#
# # ۵. نمونه‌گیری (اختیاری)
# df = df.sample(frac=0.1, random_state=42)
#
# # ۶. Resample به ساعتی
# df_hourly = df.resample('h').mean()
#
# # ۷. پر کردن NaNهای داده ساعتی
# df_hourly = df_hourly.ffill()  # اصلاح هشدار
#
# # ۸. اضافه کردن lag feature
# df_hourly['Lag_1'] = df_hourly['Global_active_power'].shift(1)  # توان ساعت قبل
# df_hourly = df_hourly.dropna()  # حذف NaNهای ناشی از lag
#
# # چک کردن داده‌های ساعتی
# print("\nداده‌های ساعتی:")
# print(df_hourly.head())
# print("\nابعاد داده ساعتی:", df_hourly.shape)
# print("\nNaNها در داده ساعتی:")
# print(df_hourly.isna().sum())
#
# # ۹. EDA: پلات‌ها
# # توزیع‌ها
# df_hourly.hist(bins=50, figsize=(12, 10))
# plt.suptitle('Distributions of Features')
# plt.show()
#
# # Correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(df_hourly.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()
#
# # Time series plot برای Global_active_power
# plt.figure(figsize=(12, 6))
# plt.plot(df_hourly.index, df_hourly['Global_active_power'], label='Global Active Power')
# plt.xlabel('Time')
# plt.ylabel('Power (kW)')
# plt.title('Hourly Global Active Power Over Time')
# plt.legend()
# plt.show()
#
# # Scatter plot: Lag_1 vs Global_active_power
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='Lag_1', y='Global_active_power', data=df_hourly)
# plt.title('Scatter: Lag_1 vs Active Power')
# plt.show()
#
# # Boxplot برای sub-metering
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df_hourly[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']])
# plt.title('Boxplot of Sub-Meterings')
# plt.show()
#
# # ۱۰. Regression: آماده‌سازی داده
# # حذف Global_intensity و استفاده از Lag_1
# X = df_hourly.drop(['Global_active_power', 'Global_intensity'], axis=1)
# y = df_hourly['Global_active_power']
#
# # مقیاس‌بندی ویژگی‌ها
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
#
# # Split chronological
# split = int(0.8 * len(df_hourly))
# X_train, X_test = X.iloc[:split], X.iloc[split:]
# y_train, y_test = y.iloc[:split], y.iloc[split:]
#
# # مدل‌ها
# models = {
#     'Linear Regression': LinearRegression(),
#     'Polynomial Regression (deg=2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
#     'Ridge': Ridge(alpha=1.0),
#     'Lasso': Lasso(alpha=0.1, max_iter=10000),  # افزایش max_iter برای Lasso
#     'Decision Tree': DecisionTreeRegressor(max_depth=5),
#     'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
# }
#
# # تست مدل‌ها
# results = {}
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     results[name] = {'MAE': mae, 'RMSE': rmse}
#     print(f'{name}: MAE={mae:.4f}, RMSE={rmse:.4f}')
#
# # پیدا کردن بهترین مدل
# best_model = min(results, key=lambda k: results[k]['RMSE'])
# print(f'\nبهترین مدل: {best_model} با RMSE={results[best_model]["RMSE"]:.4f}')
#
# # ۱۱. Clustering
# features = df_hourly[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
# scores = []
# for k in range(2, 6):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(features)
#     score = silhouette_score(features, labels)
#     scores.append(score)
#     print(f'K={k}, Silhouette Score={score:.4f}')
#
# best_k = np.argmax(scores) + 2
# print(f'بهترین K: {best_k}')
#
# # پلات clusters
# kmeans = KMeans(n_clusters=best_k, random_state=42)
# df_hourly['Cluster'] = kmeans.fit_predict(features)
# plt.figure(figsize=(12, 6))
# sns.scatterplot(x=df_hourly.index, y='Global_active_power', hue='Cluster', data=df_hourly, palette='viridis')
# plt.title('Clusters in Time Series')
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# ۱. لود داده
try:
    df = pd.read_csv('household_power_consumption.txt', sep=',', na_values='?')
except FileNotFoundError:
    print("فایل پیدا نشد! مطمئن شو که مسیر فایل درست باشه.")
    exit()

# چک کردن ستون‌ها و داده‌ها
print("ستون‌های دیتاست:")
print(df.columns)
print("\nچند ردیف اول دیتاست:")
print(df.head(10))

# ۲. پر کردن NaNهای داده اصلی
df.fillna(df.mean(numeric_only=True), inplace=True)

# ۳. ساخت ستون Datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('Datetime', inplace=True)
df.drop(['Date', 'Time'], axis=1, inplace=True)

# ۴. اضافه کردن Unmetered power
df['Unmetered'] = (df['Global_active_power'] * 1000 / 60) - df['Sub_metering_1'] - df['Sub_metering_2'] - df['Sub_metering_3']

# ۵. نمونه‌گیری (اختیاری)
df = df.sample(frac=0.1, random_state=42)

# ۶. Resample به ساعتی
df_hourly = df.resample('h').mean()

# ۷. پر کردن NaNهای داده ساعتی
df_hourly = df_hourly.ffill()

# ۸. اضافه کردن Lag_1
df_hourly['Lag_1'] = df_hourly['Global_active_power'].shift(1)
df_hourly = df_hourly.dropna()

# ۹. Clustering با K=4
features = df_hourly[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
kmeans = KMeans(n_clusters=4, random_state=42)
df_hourly['Cluster'] = kmeans.fit_predict(features)

# ذخیره مدل KMeans
joblib.dump(kmeans, 'kmeans_model.joblib')
print("مدل KMeans با K=4 ذخیره شد.")

# تحلیل خوشه‌ها
print("\nمیانگین ویژگی‌ها در هر خوشه:")
print(df_hourly.groupby('Cluster')[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean())

# پلات clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df_hourly.index, y='Global_active_power', hue='Cluster', data=df_hourly, palette='viridis')
plt.title('Clusters in Time Series (K=4)')
plt.show()

# Feature importance برای clustering (بررسی واریانس ویژگی‌ها در خوشه‌ها)
print("\nواریانس ویژگی‌ها در خوشه‌ها:")
for col in features.columns:
    print(f"{col}: {df_hourly.groupby('Cluster')[col].var()}")

# ۱۰. Regression (با حذف Unmetered و Lag_1 برای نتایج واقعی‌تر)
X = df_hourly.drop(['Global_active_power', 'Global_intensity', 'Unmetered', 'Lag_1', 'Cluster'], axis=1)
y = df_hourly['Global_active_power']

# مقیاس‌بندی ویژگی‌ها
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

# Split chronological
split = int(0.8 * len(df_hourly))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# مدل‌ها
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=10000),
    'Decision Tree': DecisionTreeRegressor(max_depth=5),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
}

# تست مدل‌ها با Cross-validation
results = {}
for name, model in models.items():
    # CV
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    cv_rmse = -scores.mean()
    # Train/test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {'MAE': mae, 'RMSE': rmse, 'CV_RMSE': cv_rmse}
    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, CV_RMSE={cv_rmse:.4f}")

# پیدا کردن بهترین مدل
best_model = min(results, key=lambda k: results[k]['CV_RMSE'])
print(f"\nبهترین مدل (بر اساس CV_RMSE): {best_model} با CV_RMSE={results[best_model]['CV_RMSE']:.4f}")

# Feature importance برای Random Forest
rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
print("\nFeature importance برای Random Forest:")
for feature, importance in zip(X.columns, rf.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# ذخیره مدل Random Forest (یا مدل انتخابی)
joblib.dump(rf, 'random_forest_model.joblib')
print("مدل Random Forest ذخیره شد.")