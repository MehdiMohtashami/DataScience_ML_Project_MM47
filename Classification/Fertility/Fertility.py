# # import pandas as  pd
# # df_dataset= pd.read_csv('fertility_Diagnosis.csv')
# # #For fertility_Diagnosis.csv
# # print('#'*40,'For fertility_Diagnosis.csv', '#'*40)
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
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
#
# # Load data
# df = pd.read_csv('fertility_Diagnosis.csv')
#
# # Encode target variable
# le = LabelEncoder()
# df['Output'] = le.fit_transform(df['Output'])  # N=0, O=1
#
# # Separate features and target
# X = df.drop('Output', axis=1)
# y = df['Output']
#
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Scale features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Initialize models
# models = {
#     'Logistic Regression': LogisticRegression(),
#     'KNN': KNeighborsClassifier(),
#     'Decision Tree': DecisionTreeClassifier(),
#     'Random Forest': RandomForestClassifier(),
#     'SVM': SVC(),
#     'XGBoost': XGBClassifier()
# }
#
# # Compare model performance
# results = {}
# for name, model in models.items():
#     cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
#     results[name] = cv_scores.mean()
#     print(f"{name}: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
#
# # Plot comparison
# plt.figure(figsize=(10, 6))
# plt.bar(results.keys(), results.values())
# plt.title('Model Comparison')
# plt.xticks(rotation=45)
# plt.ylabel('Accuracy')
# plt.tight_layout()
# plt.show()
#
# # Correlation heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title('Feature Correlation Matrix')
# plt.show()
#
# # Distribution of target variable
# plt.figure(figsize=(6, 4))
# sns.countplot(x='Output', data=df)
# plt.title('Distribution of Output (N=0, O=1)')
# plt.show()
#
# # Pairplot for selected features
# sns.pairplot(df[['Age', 'Number of hours spent sitting per day', 'Output']],
#              hue='Output')
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
df = pd.read_csv('fertility_Diagnosis.csv')

# Encode target variable
le = LabelEncoder()
df['Output'] = le.fit_transform(df['Output'])  # N=0, O=1

X = df.drop('Output', axis=1)
y = df['Output']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Detailed Cross-Validation Evaluation
print("=" * 60)
print("DETAILED CROSS-VALIDATION RESULTS")
print("=" * 60)

models = {
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='accuracy')
    print(f"{name}:")
    print(f"  Mean Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"  All CV scores: {[f'{score:.3f}' for score in cv_scores]}")
    print()

# 2. Hyperparameter Tuning for both models
print("=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

# SVM Tuning
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}

grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train_scaled, y_train)

print("SVM Best Parameters:", grid_svm.best_params_)
print("SVM Best CV Score:", grid_svm.best_score_)

# Random Forest Tuning
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                      param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train_scaled, y_train)

print("Random Forest Best Parameters:", grid_rf.best_params_)
print("Random Forest Best CV Score:", grid_rf.best_score_)

# 3. Final Model Evaluation
print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

# Train best models
best_svm = grid_svm.best_estimator_
best_rf = grid_rf.best_estimator_

best_svm.fit(X_train_scaled, y_train)
best_rf.fit(X_train_scaled, y_train)

# Test predictions
y_pred_svm = best_svm.predict(X_test_scaled)
y_pred_rf = best_rf.predict(X_test_scaled)

# Evaluate both models
print("SVM Test Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Random Forest Test Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# 4. Feature Importance Analysis
print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Get feature importances from Random Forest
feature_importances = best_rf.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("Feature Importances:")
print(importance_df.to_string(index=False))

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances from Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Save the best model
final_model = best_svm if accuracy_score(y_test, y_pred_svm) >= accuracy_score(y_test, y_pred_rf) else best_rf

joblib.dump(final_model, 'best_fertility_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print(f"\nBest model saved: {type(final_model).__name__}")

# 6. Confusion Matrix Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# SVM Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('SVM Confusion Matrix')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('Random Forest Confusion Matrix')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()