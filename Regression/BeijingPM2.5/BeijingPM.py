import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
import joblib
import time
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 1. بارگذاری داده‌ها
print("بارگذاری داده‌ها...")
data = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')

# 2. مدیریت داده‌های گمشده
print("مدیریت داده‌های گمشده...")
data = data.dropna(subset=['pm2.5'])
data = pd.get_dummies(data, columns=['cbwd'])
data = data.drop(columns=['No'])

# 3. مدیریت Outliers با روش IQR
print("مدیریت داده‌های پرت...")


def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR


    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df


# اعمال بر روی تمام ستون‌های عددی
numeric_cols = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
for col in numeric_cols:
    data = handle_outliers(data, col)

# 4. مهندسی ویژگی‌های بهتر
print("مهندسی ویژگی‌ها...")


def feature_engineering(df):
    # ویژگی‌های زمانی
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['season'] = df['month'] % 12 // 3 + 1

    # ویژگی‌های تعاملی
    df['TEMP_DEWP'] = df['TEMP'] * df['DEWP']  # تعامل دما و نقطه شبنم
    df['PRES_Iws'] = df['PRES'] / df['Iws']  # نسبت فشار به سرعت باد

    # ویژگی‌های آماری
    df['avg_last_3h'] = df['pm2.5'].rolling(window=3).mean().shift(1)  # میانگین 3 ساعت گذشته

    return df.dropna()  # حذف سطرهایی که به دلیل محاسبات rolling ایجاد شده‌اند


data = feature_engineering(data)

# 5. تقسیم داده با در نظر گرفتن ماهیت زمانی
print("تقسیم داده‌ها...")
X = data.drop(columns=['pm2.5'])
y = data['pm2.5']

# استفاده از سال 2014 به عنوان داده آزمون (جدیدترین داده‌ها)
X_train = X[X['year'] < 2014]
y_train = y[X['year'] < 2014]
X_test = X[X['year'] == 2014]
y_test = y[X['year'] == 2014]

X_train = X_train.drop(columns=['year'])
X_test = X_test.drop(columns=['year'])

print(f"تعداد داده‌های آموزش: {X_train.shape[0]}")
print(f"تعداد داده‌های آزمون: {X_test.shape[0]}")

# 6. تعریف خط لوله پیش‌پردازش
print("تعریف خط لوله پیش‌پردازش...")
numeric_features = ['month', 'day', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
                    'hour_sin', 'hour_cos', 'season', 'TEMP_DEWP', 'PRES_Iws', 'avg_last_3h']
categorical_features = list(set(X.columns) - set(numeric_features) - {'year'})

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 7. تعریف مدل‌های مختلف برای مقایسه
print("تعریف مدل‌ها برای مقایسه...")
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=50, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, random_state=42)
}

# 8. آموزش و ارزیابی مدل‌ها
print("آموزش و ارزیابی مدل‌ها...")
results = []
best_model = None
best_r2 = -np.inf
model_pipelines = {}

for name, model in models.items():
    print(f"\nآموزش مدل: {name}")
    start_time = time.time()

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    training_time = time.time() - start_time

    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Time (s)': training_time
    })

    # ذخیره خط لوله برای استفاده بعدی
    model_pipelines[name] = pipeline

    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline
        best_model_name = name

# نمایش نتایج
results_df = pd.DataFrame(results)
print("\nنتایج ارزیابی مدل‌ها:")
print(results_df.sort_values(by='R²', ascending=False))

# 9. تنظیم هیپرپارامترها برای بهترین مدل (Gradient Boosting)
if best_model_name == 'Gradient Boosting':
    print("\nتنظیم هیپرپارامترها برای Gradient Boosting...")

    # پارامترهای برای تنظیم
    param_grid = {
        'model__n_estimators': [300, 500],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }

    # استفاده از TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)  # کاهش به 3 برای سرعت اجرا

    grid_search = GridSearchCV(
        estimator=best_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nبهترین پارامترها: {grid_search.best_params_}")
    print(f"بهترین امتیاز R² در اعتبارسنجی: {grid_search.best_score_:.4f}")

    # به‌روزرسانی بهترین مدل
    best_model = grid_search.best_estimator_

    # ارزیابی نهایی روی داده آزمون
    y_pred_final = best_model.predict(X_test)
    final_r2 = r2_score(y_test, y_pred_final)
    print(f"R² نهایی روی داده آزمون: {final_r2:.4f}")

# 10. ذخیره بهترین مدل
# print("\nذخیره بهترین مدل...")
# joblib.dump(best_model, 'optimized_pm25_model.joblib')
# print(f"بهترین مدل ({best_model_name}) با دقت {best_r2:.4f} ذخیره شد.")
#
# # 11. تحلیل اهمیت ویژگی‌ها برای بهترین مدل
# print("\nتحلیل اهمیت ویژگی‌ها...")
# if hasattr(best_model.named_steps['model'], 'feature_importances_'):
#     # استخراج نام ویژگی‌ها
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
#     # ایجاد DataFrame برای اهمیت ویژگی‌ها
#     feature_importance_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': importances
#     }).sort_values('Importance', ascending=False)
#
#     # نمایش 15 ویژگی مهم
#     plt.figure(figsize=(14, 10))
#     sns.barplot(x='Importance', y='Feature',
#                 data=feature_importance_df.head(15),
#                 palette='viridis')
#     plt.title(f'۱۵ ویژگی مهم در مدل {best_model_name}')
#     plt.tight_layout()
#     plt.savefig('feature_importances.png', dpi=300)
#     plt.show()
#
#     # ذخیره جدول اهمیت ویژگی‌ها
#     feature_importance_df.to_csv('feature_importances.csv', index=False)
#
# # 12. مقایسه بصری عملکرد مدل‌ها
# print("\nایجاد نمودار مقایسه مدل‌ها...")
# plt.figure(figsize=(12, 8))
# sns.barplot(x='R²', y='Model', data=results_df.sort_values('R²', ascending=True), palette='mako')
# plt.title('مقایسه دقت مدل‌ها (R²)')
# plt.xlim(0.8, 0.95)
# plt.xlabel('ضریب تعیین (R²)')
# plt.ylabel('مدل')
# plt.tight_layout()
# plt.savefig('models_comparison.png', dpi=300)
# plt.show()
#
# # 13. پیش‌بینی نمونه و مقایسه با واقعیت
# print("\نمونه‌ای از پیش‌بینی‌ها...")
# sample_data = X_test.sample(10, random_state=42)
# actual_values = y_test.loc[sample_data.index]
#
# # پیش‌بینی با بهترین مدل
# predicted_values = best_model.predict(sample_data)
#
# # پیش‌بینی با سایر مدل‌ها برای مقایسه
# comparison_df = pd.DataFrame({
#     'Actual': actual_values,
#     best_model_name: predicted_values
# })
#
# for model_name, pipeline in model_pipelines.items():
#     if model_name != best_model_name:
#         comparison_df[model_name] = pipeline.predict(sample_data)
#
# print("\nمقایسه پیش‌بینی مدل‌ها برای 10 نمونه تصادفی:")
# print(comparison_df)
#
# # 14. ذخیره تمام مدل‌ها برای استفاده بعدی
# print("\nذخیره تمام مدل‌ها...")
# for model_name, pipeline in model_pipelines.items():
#     joblib.dump(pipeline, f'{model_name.replace(" ", "_")}_model.joblib')
# print("تمام مدل‌ها با موفقیت ذخیره شدند.")
#
# # 15. گزارش نهایی
# print("\nگزارش نهایی:")
# print(f"- بهترین مدل: {best_model_name} با دقت R² = {best_r2:.4f}")
# print(f"- تعداد ویژگی‌ها: {len(feature_names)}")
# print(f"- تعداد داده‌های آموزش: {X_train.shape[0]}")
# print(f"- تعداد داده‌های آزمون: {X_test.shape[0]}")
# print(f"- مهم‌ترین ویژگی: {feature_importance_df.iloc[0]['Feature']}")
#
# print("\nپروژه با موفقیت به پایان رسید!")

