# # import pandas as  pd
# # df_dataset= pd.read_csv('data_akbilgic.csv')
# # #For data_akbilgic.csv
# # print('#'*40,'For data_akbilgic.csv', '#'*40)
# # print(df_dataset.describe(include='all').to_string())
# # print(df_dataset.shape)
# # print(df_dataset.columns)
# # print(df_dataset.info)
# # print(df_dataset.dtypes)
# # print(df_dataset.isna().sum())
# # print(df_dataset.head(10).to_string())
# # print('='*90)
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
#
# # مدل‌های Regression
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
#
# # مدل‌های Classification
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
#
# # بارگذاری داده‌ها
# df = pd.read_csv('data_akbilgic.csv')
#
# # بررسی اولیه داده‌ها
# print("Dataset Shape:", df.shape)
# print("\nMissing Values:")
# print(df.isnull().sum())
# print("\nData Types:")
# print(df.dtypes)
#
# # تبدیل تاریخ به فرمت مناسب (اگر لازم باشد)
# df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y')
#
# # بررسی همبستگی
# plt.figure(figsize=(12, 8))
# correlation_matrix = df.corr(numeric_only=True)
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.tight_layout()
# plt.show()
#
# # تحلیل ویژگی‌ها
# plt.figure(figsize=(15, 10))
# for i, column in enumerate(df.columns[1:], 1):
#     plt.subplot(3, 4, i)
#     sns.histplot(df[column], kde=True)
#     plt.title(f'Distribution of {column}')
# plt.tight_layout()
# plt.show()
#
# # تعریف ویژگی‌ها و هدف
# X = df.drop(['date', 'ISE', 'ISE.1'], axis=1)  # حذف تاریخ و یکی از هدف‌ها
# y_regression = df['ISE.1']  # برای Regression
#
# # برای Classification: ایجاد هدف جدید (1 اگر بازدهی مثبت، 0 اگر منفی)
# y_classification = (df['ISE.1'] > 0).astype(int)
#
# # تقسیم داده‌ها
# X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
# _, _, y_train_cls, y_test_cls = train_test_split(X, y_classification, test_size=0.2, random_state=42)
#
# # استانداردسازی
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # مقایسه مدل‌های Regression
# regression_models = {
#     'Linear Regression': LinearRegression(),
#     'Ridge Regression': Ridge(alpha=1.0),
#     'Lasso Regression': Lasso(alpha=0.1),
#     'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#     'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
#     'SVR': SVR(kernel='rbf'),
#     'K-Neighbors': KNeighborsRegressor(n_neighbors=5)
# }
#
# print("=" * 60)
# print("REGRESSION MODELS COMPARISON")
# print("=" * 60)
#
# regression_results = {}
# for name, model in regression_models.items():
#     model.fit(X_train_scaled, y_train_reg)
#     y_pred = model.predict(X_test_scaled)
#
#     mse = mean_squared_error(y_test_reg, y_pred)
#     r2 = r2_score(y_test_reg, y_pred)
#
#     regression_results[name] = {'MSE': mse, 'R2': r2}
#
#     print(f"{name:20s} | MSE: {mse:.6f} | R2 Score: {r2:.4f}")
#
# # مقایسه مدل‌های Classification
# classification_models = {
#     'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
#     'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
#     'SVC': SVC(kernel='rbf', random_state=42),
#     'K-Neighbors Classifier': KNeighborsClassifier(n_neighbors=5)
# }
#
# print("\n" + "=" * 60)
# print("CLASSIFICATION MODELS COMPARISON")
# print("=" * 60)
#
# classification_results = {}
# for name, model in classification_models.items():
#     model.fit(X_train_scaled, y_train_cls)
#     y_pred = model.predict(X_test_scaled)
#
#     accuracy = accuracy_score(y_test_cls, y_pred)
#
#     classification_results[name] = {'Accuracy': accuracy}
#
#     print(f"{name:30s} | Accuracy: {accuracy:.4f}")
#
#     # نمایش گزارش دقیق‌تر برای بهترین مدل
#     if accuracy == max([result['Accuracy'] for result in classification_results.values()]):
#         best_cls_report = classification_report(y_test_cls, y_pred)
#         print(f"Classification Report for {name}:")
#         print(best_cls_report)
#
# # پیدا کردن بهترین مدل Regression
# best_reg_model = min(regression_results.items(), key=lambda x: x[1]['MSE'])
# print(f"\nBest Regression Model: {best_reg_model[0]} with MSE: {best_reg_model[1]['MSE']:.6f}")
#
# # نمایش اهمیت ویژگی‌ها برای مدل‌های ensemble
# if hasattr(regression_models['Random Forest'], 'feature_importances_'):
#     feature_importance = regression_models['Random Forest'].feature_importances_
#     feature_names = X.columns
#
#     plt.figure(figsize=(10, 6))
#     indices = np.argsort(feature_importance)[::-1]
#     plt.title('Feature Importance - Random Forest')
#     plt.bar(range(len(feature_importance)), feature_importance[indices])
#     plt.xticks(range(len(feature_importance)), feature_names[indices], rotation=45)
#     plt.tight_layout()
#     plt.show()
#
# # پیش‌بینی با بهترین مدل
# best_model = regression_models[best_reg_model[0]]
# predictions = best_model.predict(X_test_scaled)
#
# # نمایش نتایج پیش‌بینی
# results_df = pd.DataFrame({
#     'Actual': y_test_reg.values,
#     'Predicted': predictions,
#     'Absolute Error': np.abs(y_test_reg.values - predictions)
# })
#
# print("\nSample Predictions:")
# print(results_df.head(10).to_string())
#
# # نمایش scatter plot پیش‌بینی‌ها vs مقادیر واقعی
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test_reg, predictions, alpha=0.6)
# plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs Predicted Values')
# plt.tight_layout()
# plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import joblib
import time
import os

# حذف فایل‌های joblib قبلی
print("Removing previous joblib files...")
joblib_files = ['optimized_final_model.pkl', 'final_scaler.pkl', 'feature_names.pkl',
                'ensemble_model.pkl', 'ensemble_scaler.pkl', 'ensemble_features.pkl']
for file in joblib_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed {file}")

# بارگذاری داده‌ها
df = pd.read_csv('data_akbilgic.csv')

# تبدیل تاریخ به فرمت مناسب
df['date'] = pd.to_datetime(df['date'], format='%d-%b-%y')

print("Creating optimized features based on importance analysis...")

# 1. استفاده از ویژگی‌های مهم و حذف ویژگی‌های کم‌اهمیت
X_optimized = df[['EM', 'EU']].copy()

# 2. ایجاد ویژگی تعاملی
X_optimized['EM_EU_interaction'] = X_optimized['EM'] * X_optimized['EU']

# 3. ایجاد ویژگی‌های lag (تأخیری) فقط برای ویژگی‌های مهم
X_optimized['EM_lag1'] = X_optimized['EM'].shift(1)
X_optimized['EM_lag2'] = X_optimized['EM'].shift(2)
X_optimized['EU_lag2'] = X_optimized['EU'].shift(2)  # فقط lag2 برای EU

# 4. ایجاد moving averages
X_optimized['EM_MA5'] = X_optimized['EM'].rolling(window=5).mean()
X_optimized['EU_MA5'] = X_optimized['EU'].rolling(window=5).mean()

# 5. ایجاد ویژگی‌های ترند جدید
X_optimized['EM_trend'] = X_optimized['EM'] - X_optimized['EM_lag1']
X_optimized['EU_trend'] = X_optimized['EU'] - X_optimized['EU_lag2']

# حذف مقادیر NaN ناشی از lag و moving average
X_optimized = X_optimized.dropna()

# تطبیق هدف با ویژگی‌های جدید
y_optimized = (df['ISE.1'] > 0).astype(int)
y_optimized = y_optimized.iloc[len(y_optimized) - len(X_optimized):]

# تنظیم مجدد ایندکس‌ها
X_optimized = X_optimized.reset_index(drop=True)
y_optimized = y_optimized.reset_index(drop=True)

print(f"Optimized dataset shape: {X_optimized.shape}")
print(f"Features: {list(X_optimized.columns)}")

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X_optimized, y_optimized, test_size=0.2, random_state=42, stratify=y_optimized
)

# استانداردسازی
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ایجاد مدل Ensemble
print("\n" + "=" * 60)
print("CREATING ENSEMBLE MODEL")
print("=" * 60)

# تعریف مدل‌های پایه
base_models = {
    'Logistic Regression': LogisticRegression(C=100, solver='liblinear', random_state=42),
    'SVC': SVC(C=10, kernel='linear', probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# ایجاد Ensemble Model
ensemble_model = VotingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    voting='soft',
    n_jobs=-1
)

# آموزش و ارزیابی مدل Ensemble
print("Training and evaluating ensemble model...")

# Cross-Validation
cv_scores = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Ensemble CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")

# آموزش نهایی
ensemble_model.fit(X_train_scaled, y_train)

# پیش‌بینی
y_pred_ensemble = ensemble_model.predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

print(f"Ensemble Test Accuracy: {ensemble_accuracy:.4f}")
print("\nEnsemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble))

# مقایسه با مدل‌های تکی
print("\n" + "=" * 60)
print("COMPARISON WITH BASE MODELS")
print("=" * 60)

base_results = {}
for name, model in base_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    base_results[name] = accuracy
    print(f"{name:25s}: {accuracy:.4f}")

print(f"{'Ensemble':25s}: {ensemble_accuracy:.4f}")

# تحلیل اهمیت ویژگی‌ها برای Ensemble
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

perm_importance = permutation_importance(ensemble_model, X_test_scaled, y_test,
                                        n_repeats=10, random_state=42, n_jobs=-1)

feature_importance_df = pd.DataFrame({
    'feature': X_optimized.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("Feature Importance Scores:")
print(feature_importance_df.to_string(index=False))

# Visualization
plt.figure(figsize=(18, 12))

# نمودار اهمیت ویژگی‌ها
plt.subplot(2, 3, 1)
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance_df)))
bars = plt.barh(feature_importance_df['feature'], feature_importance_df['importance_mean'],
                xerr=feature_importance_df['importance_std'], color=colors)
plt.xlabel('Permutation Importance')
plt.title('Feature Importance (Permutation)')
plt.gca().invert_yaxis()

# نمودار مقایسه مدل‌ها
plt.subplot(2, 3, 2)
models = list(base_results.keys()) + ['Ensemble']
accuracies = list(base_results.values()) + [ensemble_accuracy]
colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
bars = plt.bar(models, accuracies, color=colors)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim(0.7, 0.9)
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{accuracy:.3f}', ha='center', va='bottom')
plt.xticks(rotation=45)

# نمودار confusion matrix
plt.subplot(2, 3, 3)
cm = confusion_matrix(y_test, y_pred_ensemble)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Ensemble')

# نمودار cross-validation scores
plt.subplot(2, 3, 4)
plt.plot(range(1, 6), cv_scores, 'o-', label='CV Scores')
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Ensemble Cross-Validation Scores')
plt.legend()
plt.ylim(0.7, 0.9)

# نمودار مقایسه دقت Ensemble با بهترین مدل تکی
plt.subplot(2, 3, 5)
best_single_model = max(base_results.items(), key=lambda x: x[1])
comparison_labels = [f'Best Single ({best_single_model[0]})', 'Ensemble']
comparison_accuracies = [best_single_model[1], ensemble_accuracy]
colors = ['lightblue', 'gold']
bars = plt.bar(comparison_labels, comparison_accuracies, color=colors)
plt.ylabel('Accuracy')
plt.title('Ensemble vs Best Single Model')
plt.ylim(0.7, 0.9)
for bar, accuracy in zip(bars, comparison_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{accuracy:.3f}', ha='center', va='bottom')

# نمودار ویژگی‌های مهم
plt.subplot(2, 3, 6)
top_features = feature_importance_df.head(5)
plt.barh(top_features['feature'], top_features['importance_mean'], color='lightgreen')
plt.xlabel('Importance')
plt.title('Top 5 Most Important Features')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# ذخیره مدل Ensemble و scaler
print("\nSaving the ensemble model and scaler...")
joblib.dump(ensemble_model, 'ensemble_model.pkl')
joblib.dump(scaler, 'ensemble_scaler.pkl')
joblib.dump(list(X_optimized.columns), 'ensemble_features.pkl')
print("Ensemble model, scaler, and feature names saved successfully!")

# نمایش اطلاعات نهایی
print("\n" + "=" * 60)
print("FINAL ENSEMBLE MODEL INFORMATION")
print("=" * 60)
print(f"Best single model: {best_single_model[0]} with accuracy: {best_single_model[1]:.4f}")
print(f"Ensemble model accuracy: {ensemble_accuracy:.4f}")
print(f"Improvement: {((ensemble_accuracy - best_single_model[1]) / best_single_model[1] * 100):.2f}%")
print(f"Number of features: {X_optimized.shape[1]}")
print(f"Top 3 features: {list(feature_importance_df['feature'].head(3))}")

# پیش‌بینی نمونه‌ای
print("\nSample Predictions with Ensemble Model:")
sample_indices = np.random.choice(len(X_test), 15, replace=False)
sample_X = X_test.iloc[sample_indices]
sample_y_true = y_test.iloc[sample_indices]
sample_y_pred = ensemble_model.predict(scaler.transform(sample_X))

sample_results = pd.DataFrame({
    'Actual': sample_y_true.values,
    'Predicted': sample_y_pred,
    'Correct': sample_y_true.values == sample_y_pred
})

print(sample_results.to_string(index=False))
print(f"\nSample Accuracy: {sample_results['Correct'].mean():.2%}")

# نمایش احتمال پیش‌بینی برای چند نمونه
print("\nPrediction Probabilities for Sample Cases:")
sample_probas = ensemble_model.predict_proba(scaler.transform(sample_X.head(5)))
for i, (true_val, pred_val) in enumerate(zip(sample_y_true.head(5), sample_y_pred.head(5))):
    print(f"Sample {i+1}: Actual={true_val}, Predicted={pred_val}, "
          f"Probability=[{sample_probas[i][0]:.3f}, {sample_probas[i][1]:.3f}]")