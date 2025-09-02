# import pandas as  pd
# df_dataset= pd.read_csv('data_banknote_authentication.csv')
# #For data_banknote_authentication.csv
# print('#'*40,'For data_banknote_authentication.csv', '#'*40)
# print(df_dataset.describe(include='all').to_string())
# print(df_dataset.shape)
# print(df_dataset.columns)
# print(df_dataset.info)
# print(df_dataset.dtypes)
# print(df_dataset.isna().sum())
# print(df_dataset.head(10).to_string())
# print('='*90)
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv('data_banknote_authentication.csv')

# 1. EXPLORATORY DATA ANALYSIS
print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# Check basic info
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDataset Description:")
print(df.describe())

# Check class distribution
print("\nClass Distribution:")
print(df['Class'].value_counts())
print(f"\nClass Ratio: {df['Class'].value_counts(normalize=True)}")

# Visualizations
plt.figure(figsize=(15, 10))

# Pairplot to see relationships between features
sns.pairplot(df, hue='Class', diag_kind='hist')
plt.suptitle('Pairplot of Features by Class', y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Boxplots for each feature by class
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
features = df.columns[:-1]

for i, feature in enumerate(features):
    row, col = i // 2, i % 2
    sns.boxplot(x='Class', y=feature, data=df, ax=axes[row, col])
    axes[row, col].set_title(f'Boxplot of {feature} by Class')

plt.tight_layout()
plt.show()

# 2. DATA PREPROCESSING
print("\n" + "=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

# Split features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data split and scaled successfully!")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 3. MODEL TRAINING AND COMPARISON
print("\n" + "=" * 50)
print("MODEL TRAINING AND COMPARISON")
print("=" * 50)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate models
results = {}
for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()

    # Store results
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_mean,
        'model': model
    }

    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  CV Mean Accuracy: {cv_mean:.4f}")
    print("-" * 30)

# 4. FIND BEST MODEL
print("\n" + "=" * 50)
print("MODEL COMPARISON RESULTS")
print("=" * 50)

# Create results dataframe
results_df = pd.DataFrame({
    'Model': [name for name in results.keys()],
    'Accuracy': [results[name]['accuracy'] for name in results.keys()],
    'CV Mean': [results[name]['cv_mean'] for name in results.keys()]
})

# Sort by accuracy
results_df = results_df.sort_values('Accuracy', ascending=False)

print("Models sorted by Accuracy:")
print(results_df.to_string(index=False))

# Plot accuracy comparison
plt.figure(figsize=(12, 6))
sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlim(0.9, 1.0)
plt.tight_layout()
plt.show()

# 5. DETAILED EVALUATION OF BEST MODEL
best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

print("\n" + "=" * 50)
print(f"DETAILED EVALUATION OF BEST MODEL: {best_model_name}")
print("=" * 50)

# Make predictions with best model
y_pred = best_model.predict(X_test_scaled)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Forged (0)', 'Genuine (1)'],
            yticklabels=['Forged (0)', 'Genuine (1)'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=['Forged (0)', 'Genuine (1)']))

# Feature importance (if available)
if hasattr(best_model, 'feature_importance_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.show()

    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Based on our analysis, the best model for this dataset is: {best_model_name}")
print(f"With an accuracy of: {results[best_model_name]['accuracy']:.4f}")

# # Hyperparameter tuning for superior models
# from sklearn.model_selection import GridSearchCV
#
# # Hyperparameter tuning for SVM
# param_grid_svm = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 'auto', 0.1, 0.01],
#     'kernel': ['rbf', 'linear']
# }
#
# grid_svm = GridSearchCV(SVC(random_state=42), param_grid_svm,
#                        cv=5, scoring='accuracy', n_jobs=-1)
# grid_svm.fit(X_train_scaled, y_train)
#
# print("Best SVM Parameters:", grid_svm.best_params_)
# print("Best SVM Score:", grid_svm.best_score_)
#
# # Hyperparameter tuning for KNN
# param_grid_knn = {
#     'n_neighbors': range(3, 15),
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }
#
# grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn,
#                        cv=5, scoring='accuracy', n_jobs=-1)
# grid_knn.fit(X_train_scaled, y_train)
#
# print("Best KNN Parameters:", grid_knn.best_params_)
# print("Best KNN Score:", grid_knn.best_score_)

# Feature Importance Analysis for Ensemble Models
best_rf = RandomForestClassifier(random_state=42)
best_rf.fit(X_train_scaled, y_train)

# Getting the importance of features
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance (Random Forest):")
print(feature_importance)

# Feature importance chart
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()

# Stronger Validation with Stratified K-Fold
from sklearn.model_selection import StratifiedKFold

# Using 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Evaluating models with advanced CV
models_advanced = {
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models_advanced.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train,
                               cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"{name} - CV Scores: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Plotting learning curves to check overfitting
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Plotting the learning curve for SVM
plot_learning_curve(SVC(kernel='rbf', C=10, random_state=42),
                    "Learning Curve (SVM)",
                    X_train_scaled, y_train, cv=5)
plt.show()
#
# import joblib
#
# # Save the best model
# best_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
# best_model.fit(X_train_scaled, y_train)
#
# # Save model and scaler
# joblib.dump(best_model, 'banknote_svm_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
#
# print("Model and scaler saved successfully!")