import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump

# 1. بارگذاری داده‌ها
data = pd.read_csv('Daily_Demand_Forecasting_Orders.csv')  # مسیر فایل خود را قرار دهید


# 2. تحلیل اکتشافی (EDA)
def perform_eda(df):
    print("اطلاعات اولیه دیتاست:")
    print(df.info())

    print("\nآمار توصیفی:")
    print(df.describe())
    print()

    # توزیع متغیر هدف
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Target (Total orders)'], kde=True)
    plt.title('Target variable distribution (Total Orders)')
    plt.savefig('TotalOrders.png')
    plt.show()

    # ماتریس همبستگی
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature correlation matrix')
    plt.savefig('correlation-matrix.png')
    plt.show()

    # بررسی رابطه ویژگی‌ها با متغیر هدف
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns[:-1], 1):
        plt.subplot(3, 4, i)
        sns.scatterplot(x=df[column], y=df['Target (Total orders)'])
        plt.title(f'{column} vs Target')
    plt.tight_layout()
    plt.savefig('MultiTO.png')
    plt.show()


perform_eda(data)

# 3. پیش‌پردازش داده‌ها
X = data.drop('Target (Total orders)', axis=1)
y = data['Target (Total orders)']

# تقسیم داده به آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# استانداردسازی ویژگی‌ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. پیاده‌سازی و ارزیابی مدل‌های مختلف
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = []
for name, model in models.items():
    # آموزش مدل
    model.fit(X_train_scaled, y_train)

    # پیش‌بینی
    y_pred = model.predict(X_test_scaled)

    # ارزیابی
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # اعتبارسنجی متقاطع
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_r2 = np.mean(cv_scores)

    results.append({
        'Model': name,
        'R2_Score': r2,
        'MAE': mae,
        'MSE': mse,
        'CV_R2': cv_r2
    })

# 5. مقایسه مدل‌ها
results_df = pd.DataFrame(results).sort_values(by='R2_Score', ascending=False)
print("\nنتایج ارزیابی مدل‌ها:")
print(results_df)

# 6. انتخاب بهترین مدل
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
print(f"\nبهترین مدل: {best_model_name}")

# 7. ذخیره مدل با joblib
dump(best_model, 'best_regression_model.joblib')
dump(scaler, 'scaler.joblib')
print("مدل و اسکیلر با موفقیت ذخیره شدند.")

# 8. تحلیل نتایج و مصورسازی
plt.figure(figsize=(12, 8))
sns.barplot(x='R2_Score', y='Model', data=results_df, palette='viridis')
plt.xlabel('R² Score')
plt.ylabel('Model')
plt.title('Comparing the performance of regression models')
plt.xlim(0, 1)
plt.savefig('Reg-Compering.png')
plt.show()