import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# خواندن داده‌ها
hour_df = pd.read_csv('hour.csv')
day_df = pd.read_csv('day.csv')

# بررسی اولیه داده‌ها
print("تعداد داده‌های hour:", hour_df.shape)
print("تعداد داده‌های day:", day_df.shape)

# تنظیم ظاهر نمودارها
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
# توزیع تعداد دوچرخه‌های اجاره‌ای
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# توزیع cnt
sns.histplot(hour_df['cnt'], kde=True, ax=axes[0,0], color='blue')
axes[0,0].set_title('Distribution of number of rental bicycles (hourly)')

# رابطه دما و تعداد دوچرخه
sns.scatterplot(x=hour_df['temp'], y=hour_df['cnt'], alpha=0.5, ax=axes[0,1], color='green')
axes[0,1].set_title('Relationship between temperature and number of bicycles')

# تعداد دوچرخه بر اساس ساعت روز
sns.boxplot(x='hr', y='cnt', data=hour_df, ax=axes[1,0], color='yellow')
axes[1,0].set_title('Number of bicycles by time of day')

# تعداد دوچرخه بر اساس فصل
sns.boxplot(x='season', y='cnt', data=hour_df, ax=axes[1,1],color='Purple')
axes[1,1].set_title('Number of bicycles by season')

plt.tight_layout()
plt.show()

# کپی از داده‌ها برای پیش‌پردازش
df = hour_df.copy()

# تبدیل ستون dteday به datetime
df['dteday'] = pd.to_datetime(df['dteday'])

# استخراج ویژگی‌های جدید از تاریخ
df['year'] = df['dteday'].dt.year
df['month'] = df['dteday'].dt.month
df['day'] = df['dteday'].dt.day

# حذف ستون‌های غیرضروری
df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1, inplace=True)

# بررسی همبستگی
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Feature correlation matrix')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# جدا کردن ویژگی‌ها و متغیر هدف
X = df.drop('cnt', axis=1)
y = df['cnt']

# تقسیم داده به آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# مدل‌های مختلف
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
}

# ذخیره نتایج
results = {}

for name, model in models.items():
    # آموزش مدل
    model.fit(X_train_scaled, y_train)

    # پیش‌بینی
    y_pred = model.predict(X_test_scaled)

    # محاسبه معیارهای ارزیابی
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # ذخیره نتایج
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    print(f"{name}:")
    print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}\n")

# مقایسه نتایج
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('R2', ascending=False)
print(results_df)

from sklearn.model_selection import GridSearchCV

# تنظیم هیپرپارامترها برای بهترین مدل
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# فرض می‌کنیم Random Forest بهترین بوده
best_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("بهترین پارامترها:", grid_search.best_params_)
print("بهترین امتیاز:", grid_search.best_score_)

# استفاده از بهترین مدل
best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test_scaled)

# ارزیابی مدل نهایی
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
final_r2 = r2_score(y_test, y_pred_best)
print(f"مدل نهایی - RMSE: {final_rmse:.2f}, R2: {final_r2:.4f}")

# اهمیت ویژگی‌ها در بهترین مدل
feature_importance = best_rf_model.feature_importances_
feature_names = X.columns

# ایجاد DataFrame برای اهمیت ویژگی‌ها
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# نمودار اهمیت ویژگی‌ها
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('The importance of features in predicting the number of rental bicycles')
plt.tight_layout()
plt.show()

# import joblib
# import xgboost as xgb
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import pandas as pd
#
# # خواندن و پیش‌پردازش داده‌ها
# df = pd.read_csv('hour.csv')
# df['dteday'] = pd.to_datetime(df['dteday'])
# df['year'] = df['dteday'].dt.year
# df['month'] = df['dteday'].dt.month
# df['day'] = df['dteday'].dt.day
# df.drop(['instant', 'dteday', 'casual', 'registered'], axis=1, inplace=True)
#
# # جدا کردن ویژگی‌ها و متغیر هدف
# X = df.drop('cnt', axis=1)
# y = df['cnt']
#
# # تقسیم داده به آموزش و آزمون
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # استانداردسازی داده‌ها
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # آموزش مدل XGBoost (با بهترین پارامترها)
# best_xgb_model = xgb.XGBRegressor(
#     n_estimators=100,
#     random_state=42,
#     max_depth=6,  # می‌توانید این پارامترها را تنظیم کنید
#     learning_rate=0.1
# )
# best_xgb_model.fit(X_train_scaled, y_train)
#
# # ذخیره مدل و اسکیلر
# joblib.dump(best_xgb_model, 'bike_sharing_xgboost_model.pkl')
# joblib.dump(scaler, 'bike_sharing_scaler.pkl')
#
# print("مدل و اسکیلر با موفقیت ذخیره شدند!")
#
#
# # کد برای بارگذاری و استفاده از مدل در آینده
# def load_and_predict(new_data):
#     # بارگذاری مدل و اسکیلر
#     model = joblib.load('bike_sharing_xgboost_model.pkl')
#     scaler = joblib.load('bike_sharing_scaler.pkl')
#
#     # پیش‌پردازش داده جدید (مطابق با روشی که روی داده آموزش انجام دادیم)
#     new_data = new_data.copy()
#     new_data['dteday'] = pd.to_datetime(new_data['dteday'])
#     new_data['year'] = new_data['dteday'].dt.year
#     new_data['month'] = new_data['dteday'].dt.month
#     new_data['day'] = new_data['dteday'].dt.day
#     new_data.drop(['instant', 'dteday', 'casual', 'registered'], axis=1, inplace=True, errors='ignore')
#
#     # استانداردسازی داده جدید
#     new_data_scaled = scaler.transform(new_data)
#
#     # پیش‌بینی
#     predictions = model.predict(new_data_scaled)
#     return predictions
#
#
# # تست بارگذاری و پیش‌بینی
# try:
#     # استفاده از چند نمونه از داده تست برای آزمایش
#     sample_data = X_test.head(3).copy()
#     predictions = load_and_predict(sample_data)
#     print("پیش‌بینی‌های نمونه:", predictions)
#     print("مقادیر واقعی:", y_test.head(3).values)
# except Exception as e:
#     print("خطا در آزمایش مدل:", e)