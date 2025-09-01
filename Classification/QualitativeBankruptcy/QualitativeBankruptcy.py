import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('Qualitative_Bankruptcy.csv')

# بررسی اولیه
print("Data shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nClass distribution:\n", df['Class'].value_counts())

# تبدیل داده‌های کیفی به عددی برای بصری‌سازی
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

# ماتریس همبستگی
plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig('CorrelationMatrix.png')
plt.show()

# توزیع کلاس‌ها
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.savefig('Class_Distribution.png')
plt.show()

# توزیع هر ویژگی نسبت به کلاس
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features = df.columns[:-1]

for i, feature in enumerate(features):
    row, col = i // 3, i % 3
    sns.countplot(x=feature, hue='Class', data=df, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} vs Class')

plt.tight_layout()
plt.savefig('Other_Feature.png')
plt.show()

# Encoding داده‌ها
from sklearn.preprocessing import OneHotEncoder

# برای ویژگی‌ها
X = df.drop('Class', axis=1)
y = df['Class']

# One-Hot Encoding
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = encoder.fit_transform(X)

# برای تارگت
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f'{name} Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred))
    print('-' * 50)

# مقایسه نتایج
plt.figure(figsize=(12, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.xticks(rotation=45)
plt.title('Model Comparison')
plt.ylabel('Accuracy')
plt.savefig('Model_Comparison.png')
plt.show()

from sklearn.model_selection import GridSearchCV

# انتخاب بهترین مدل (مثلاً Random Forest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# استفاده از بهترین مدل
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Final Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score

#اعتبارسنجی متقابل برای اطمینان
cv_scores = cross_val_score(best_model, X_encoded, y_encoded, cv=10)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# استفاده از train/test split متفاوت
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=123, stratify=y_encoded
)

best_model.fit(X_train2, y_train2)
y_pred2 = best_model.predict(X_test2)
print(f"Accuracy with different split: {accuracy_score(y_test2, y_pred2):.4f}")
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# ایجاد پوشه برای ذخیره مدل‌ها
os.makedirs('models', exist_ok=True)

# Load data
df = pd.read_csv('Qualitative_Bankruptcy.csv')

# جدا کردن ویژگی‌ها و تارگت
X = df.drop('Class', axis=1)
y = df['Class']

# One-Hot Encoding برای ویژگی‌ها
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Label Encoding برای تارگت
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# آموزش مدل Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# آموزش مدل KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

# ذخیره مدل‌ها و encoderها با joblib
joblib.dump(logistic_model, 'models/logistic_regression_model.pkl')
joblib.dump(knn_model, 'models/knn_model.pkl')
joblib.dump(encoder, 'models/onehot_encoder.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')

print("مدل‌ها و encoderها با موفقیت ذخیره شدند!")
print("فایل‌های ایجاد شده:")
print("- models/logistic_regression_model.pkl")
print("- models/knn_model.pkl")
print("- models/onehot_encoder.pkl")
print("- models/label_encoder.pkl")


# تست مدل‌های ذخیره شده
def test_saved_models():
    # بارگذاری مدل‌ها
    logistic_loaded = joblib.load('models/logistic_regression_model.pkl')
    knn_loaded = joblib.load('models/knn_model.pkl')
    encoder_loaded = joblib.load('models/onehot_encoder.pkl')
    label_encoder_loaded = joblib.load('models/label_encoder.pkl')

    # پیش‌بینی با داده تست
    logistic_pred = logistic_loaded.predict(X_test)
    knn_pred = knn_loaded.predict(X_test)

    # دقت مدل‌ها
    logistic_accuracy = np.mean(logistic_pred == y_test)
    knn_accuracy = np.mean(knn_pred == y_test)

    print(f"\nدقت مدل Logistic Regression بعد از بارگذاری: {logistic_accuracy:.4f}")
    print(f"دقت مدل KNN بعد از بارگذاری: {knn_accuracy:.4f}")


# تست مدل‌های ذخیره شده
test_saved_models()


# تابع برای پیش‌بینی با داده جدید
def predict_new_data(model_type, new_data):
    """
    تابع برای پیش‌بینی با داده جدید

    Parameters:
    model_type: 'logistic' یا 'knn'
    new_data: داده جدید به صورت DataFrame با همان ساختار داده اصلی
    """
    # بارگذاری مدل و encoderها
    if model_type == 'logistic':
        model = joblib.load('models/logistic_regression_model.pkl')
    elif model_type == 'knn':
        model = joblib.load('models/knn_model.pkl')
    else:
        raise ValueError("model_type باید 'logistic' یا 'knn' باشد")

    encoder = joblib.load('models/onehot_encoder.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')

    # تبدیل داده جدید
    new_data_encoded = encoder.transform(new_data)

    # پیش‌بینی
    prediction_encoded = model.predict(new_data_encoded)
    prediction = label_encoder.inverse_transform(prediction_encoded)

    return prediction


# مثال استفاده از تابع پیش‌بینی
print("\nمثال استفاده از تابع پیش‌بینی:")
sample_data = pd.DataFrame({
    'Industrial Risk': ['P'],
    'Management Risk': ['P'],
    'Financial Flexibility': ['P'],
    'Credibility': ['P'],
    'Competitiveness': ['P'],
    'Operating Risk': ['P']
})

logistic_prediction = predict_new_data('logistic', sample_data)
knn_prediction = predict_new_data('knn', sample_data)

print(f"پیش‌بینی Logistic Regression: {logistic_prediction[0]}")
print(f"پیش‌بینی KNN: {knn_prediction[0]}")

import joblib
import pandas as pd

# بارگذاری مدل‌ها
logistic_model = joblib.load('models/logistic_regression_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')
encoder = joblib.load('models/onehot_encoder.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# داده جدید برای پیش‌بینی
new_data = pd.DataFrame({
    'Industrial Risk': ['N'],
    'Management Risk': ['N'],
    'Financial Flexibility': ['N'],
    'Credibility': ['N'],
    'Competitiveness': ['N'],
    'Operating Risk': ['N']
})

# تبدیل داده جدید
new_data_encoded = encoder.transform(new_data)

# پیش‌بینی
logistic_pred = logistic_model.predict(new_data_encoded)
knn_pred = knn_model.predict(new_data_encoded)

# تبدیل به label اصلی
logistic_result = label_encoder.inverse_transform(logistic_pred)
knn_result = label_encoder.inverse_transform(knn_pred)

print(f"Logistic Regression Prediction: {logistic_result[0]}")
print(f"KNN Prediction: {knn_result[0]}")
