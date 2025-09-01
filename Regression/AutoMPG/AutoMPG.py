import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# 1. بارگذاری داده‌ها از فایل CSV
df = pd.read_csv('auto-mpg.csv')


# 2. پیش‌پردازش داده‌ها
def preprocess_data(df):
    # بررسی و مدیریت مقادیر گمشده
    print("\nMissing values status before preprocessing:")
    print(df.isna().sum())

    # ستون‌های عددی
    numeric_cols = ['displacement', 'horsepower', 'weight', 'acceleration']

    # جایگزینی مقادیر گمشده با میانه
    for col in numeric_cols:
        if df[col].dtype == 'object':
            # تبدیل به عددی و مدیریت مقادیر نامعتبر
            df[col] = pd.to_numeric(df[col], errors='coerce')

        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

    # تبدیل origin به رشته
    origin_map = {1: 'American', 2: 'European', 3: 'Japanese'}
    df['origin'] = df['origin'].map(origin_map)

    # حذف ستون غیرضروری
    if 'car name' in df.columns:
        df.drop('car name', axis=1, inplace=True)
    if 'car_name' in df.columns:
        df.drop('car_name', axis=1, inplace=True)

    # بررسی نهایی مقادیر گمشده
    print("\nMissing values status after preprocessing:")
    print(df.isna().sum())

    return df


df = preprocess_data(df)


# 3. تحلیل اکتشافی داده‌ها (EDA) با عنوان‌های انگلیسی
def perform_eda(df):
    plt.figure(figsize=(18, 12))

    # توزیع MPG
    plt.subplot(2, 3, 1)
    sns.histplot(df['mpg'], kde=True, color='skyblue')
    plt.title('MPG Distribution', fontsize=14)

    # ماتریس همبستگی
    plt.subplot(2, 3, 2)
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix', fontsize=14)

    # رابطه وزن و مصرف سوخت
    plt.subplot(2, 3, 3)
    sns.scatterplot(x='weight', y='mpg', data=df, hue='origin', palette='viridis', alpha=0.8)
    plt.title('Vehicle Weight vs MPG', fontsize=14)

    # مصرف سوخت بر اساس سال ساخت
    plt.subplot(2, 3, 4)
    sns.boxplot(x='model year', y='mpg', data=df, palette='Set2')
    plt.title('MPG by Model Year', fontsize=14)

    # رابطه حجم موتور و مصرف سوخت
    plt.subplot(2, 3, 5)
    sns.regplot(x='displacement', y='mpg', data=df, scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'})
    plt.title('Engine Displacement vs MPG', fontsize=14)

    # رابطه اسب بخار و شتاب
    plt.subplot(2, 3, 6)
    sns.scatterplot(x='horsepower', y='acceleration', data=df, hue='cylinders', palette='plasma')
    plt.title('Horsepower vs Acceleration', fontsize=14)

    plt.tight_layout()
    plt.savefig('auto_eda.png', dpi=300)
    plt.show()

    # بررسی همبستگی با MPG
    plt.figure(figsize=(10, 6))
    corr_with_mpg = df.corr(numeric_only=True)['mpg'].drop('mpg')
    corr_with_mpg.sort_values().plot(kind='barh', color='teal')
    plt.title('Feature Correlation with MPG', fontsize=14)
    plt.xlabel('Correlation Coefficient')
    plt.savefig('mpg_correlation.png', dpi=300)
    plt.show()


perform_eda(df)

# 4. آماده‌سازی داده‌ها برای مدل‌سازی
X = df.drop('mpg', axis=1)
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# تعریف تبدیل‌گرهای ستون‌ها
numeric_features = ['displacement', 'horsepower', 'weight', 'acceleration']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['cylinders', 'model year', 'origin']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. پیاده‌سازی و ارزیابی مدل‌های مختلف
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, random_state=42),
    'Support Vector': SVR(C=1.0, epsilon=0.2)
}

results = []

for name, model in models.items():
    try:
        # ایجاد پاینل کامل
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # آموزش مدل
        full_pipeline.fit(X_train, y_train)

        # پیش‌بینی و ارزیابی
        y_pred = full_pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            'Model': name,
            'R2 Score': r2,
            'MAE': mae,
            'RMSE': rmse
        })

        print(f'{name} - R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    except Exception as e:
        print(f"Error in {name} model: {str(e)}")

# 6. مقایسه مدل‌ها
results_df = pd.DataFrame(results).sort_values('R2 Score', ascending=False)
print("\nModel evaluation results:")
print(results_df.to_string(index=False))
#
# # 7. انتخاب Ridge Regression به عنوان بهترین مدل (به صورت دستی)
# best_model_name = 'Ridge Regression'
# best_model = models[best_model_name]
#
# # آموزش مجدد مدل انتخابی
# final_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', best_model)
# ])
#
# final_pipeline.fit(X_train, y_train)
#
# # ذخیره مدل با joblib
# joblib.dump(final_pipeline, 'best_auto_mpg_model.joblib')
# print(f"\nSelected model ({best_model_name}) saved successfully.")
#
# # 8. مصورسازی نتایج با عنوان‌های انگلیسی
# plt.figure(figsize=(14, 8))
# sns.set_style("whitegrid")
# ax = sns.barplot(x='R2 Score', y='Model', data=results_df, palette='viridis')
# plt.title('Regression Models Comparison (R-Squared)', fontsize=16)
# plt.xlabel('R2 Score', fontsize=12)
# plt.ylabel('Model', fontsize=12)
# plt.xlim(0.7, 0.95)
#
# # افزودن مقادیر روی نمودار
# for p in ax.patches:
#     width = p.get_width()
#     plt.text(width + 0.005, p.get_y() + p.get_height() / 2,
#              '{:.4f}'.format(width),
#              ha='left', va='center', fontsize=10)
#
# plt.tight_layout()
# plt.savefig('model_comparison.png', dpi=300)
# plt.show()
#
# # 9. ارزیابی نهایی مدل انتخابی
# y_pred = final_pipeline.predict(X_test)
#
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.title('Actual vs Predicted Values', fontsize=14)
# plt.xlabel('Actual MPG', fontsize=12)
# plt.ylabel('Model Predictions', fontsize=12)
# plt.savefig('actual_vs_predicted.png', dpi=300)
# plt.show()
#
#
# # 10. مثال پیش‌بینی با مدل نهایی
# def predict_mpg(model, sample_data):
#     """پیش‌بینی مصرف سوخت برای داده‌های نمونه"""
#     prediction = model.predict(sample_data)
#     print(f"\nPredicted MPG: {prediction[0]:.2f}")
#     return prediction[0]
#
#
# # ایجاد یک نمونه دلخواه از داده‌ها
# sample_car = pd.DataFrame({
#     'cylinders': [6],
#     'displacement': [250.0],
#     'horsepower': [100.0],
#     'weight': [3500],
#     'acceleration': [15.0],
#     'model year': [75],
#     'origin': ['American']
# })
#
# # پیش‌بینی برای نمونه
# predicted_mpg = predict_mpg(final_pipeline, sample_car)
#
# # نمایش جزئیات نمونه و پیش‌بینی
# print("\nSample car details:")
# print(sample_car)
# print(f"Predicted fuel efficiency: {predicted_mpg:.2f} MPG")
#
# # 11. ذخیره تمام نتایج در یک گزارش
# with open('modeling_report.txt', 'w') as f:
#     f.write("AutoMPG Modeling Report\n")
#     f.write("=" * 50 + "\n\n")
#     f.write("Top performing models:\n")
#     f.write(results_df.head().to_string(index=False) + "\n\n")
#     f.write(f"Selected model: {best_model_name}\n")
#     f.write(f"Sample prediction: {predicted_mpg:.2f} MPG\n")
#     f.write("\nSample car details:\n")
#     f.write(sample_car.to_string(index=False))
#
# print("\nModeling report saved as 'modeling_report.txt'")