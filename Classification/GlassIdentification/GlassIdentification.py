# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier
# import joblib
# from imblearn.over_sampling import SMOTE
#
# # 1. Loading data
# df = pd.read_csv('glass.csv')
#
# #2. Quick data review
# print("First 5 rows:")
# print(df.head())
# print("\nDataset info:")
# print(df.info())
# print("\nSummary statistics:")
# print(df.describe())
# print("\nTarget variable distribution:")
# print(df['Type of Glass'].value_counts())
#
# # 3. Visualizing the distribution of classes (Target Variable)
# plt.figure(figsize=(10, 6))
# sns.countplot(x='Type of Glass', data=df, hue='Type of Glass', palette='viridis', legend=False)
# plt.title('Distribution of Glass Types')
# plt.xlabel('Glass Type')
# plt.ylabel('Count')
# plt.show()
#
# #4. Checking correlation between features
# plt.figure(figsize=(12, 10))
# correlation_matrix = df.drop('Id number', axis=1).corr()  # حذف ستون Id
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix of Features')
# plt.show()
#
# # 5. Visualizing feature distribution with Boxplot (for outlier detection)
# features = df.drop(['Id number', 'Type of Glass'], axis=1).columns
# plt.figure(figsize=(16, 20))
# for i, feature in enumerate(features, 1):
#     plt.subplot(4, 3, i)
#     sns.boxplot(x='Type of Glass', y=feature, data=df)
#     plt.title(f'Boxplot of {feature} by Glass Type')
# plt.tight_layout()
# plt.show()
#
# #1. Separate features and labels
# X = df.drop(['Id number', 'Type of Glass'], axis=1)  # حذف ستون Id و Target
# y = df['Type of Glass']
#
# #2. Splitting data into Train and Test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# # The stratify parameter is to maintain the proportion of classes in train and test
#
# # 3. Data standardization (essential for many models such as SVM and KNN)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # A list to store results
# models = []
# models.append(('Logistic Regression', LogisticRegression(max_iter=1000)))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('SVM Linear', SVC(kernel='linear')))
# models.append(('SVM RBF', SVC(kernel='rbf')))
# models.append(('Decision Tree', DecisionTreeClassifier()))
# models.append(('Random Forest', RandomForestClassifier()))
# models.append(('Naive Bayes', GaussianNB()))
#
# # Evaluate each model using Cross-Validation
# results = []
# names = []
# for name, model in models:
#     cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
#     results.append(cv_scores)
#     names.append(name)
#     print(f'{name}: Mean Accuracy = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}')
#
# # Visually compare the accuracy of models with Boxplot
# plt.figure(figsize=(15, 8))
# plt.boxplot(results, tick_labels=names)  # Change labels to tick labels to fix warning
# plt.title('Model Comparison - Cross Validation Scores')
# plt.xticks(rotation=45)
# plt.ylabel('Accuracy')
# plt.show()
#
# # Training the best model (Random Forest)
# best_model = RandomForestClassifier(class_weight='balanced', random_state=42)
# best_model.fit(X_train_scaled, y_train)
#
# # پیش‌بینی روی داده تست
# y_pred = best_model.predict(X_test_scaled)
#
# # ارزیابی مدل
# print("Accuracy on test set:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
#
# # Confusion Matrix for more details
# plt.figure(figsize=(8, 6))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.show()
#
# # Tuning Random Forest hyperparameters
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# rf = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train_scaled, y_train)
#
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validation score:", grid_search.best_score_)
#
# # Use the best model
# best_rf_model = grid_search.best_estimator_
# y_pred_tuned = best_rf_model.predict(X_test_scaled)
# print("Tuned Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
#
# # Using SMOTE to deal with data imbalance
# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
#
# # Training the model with balanced data
# best_model_smote = RandomForestClassifier(**grid_search.best_params_, random_state=42)
# best_model_smote.fit(X_train_res, y_train_res)
# y_pred_smote = best_model_smote.predict(X_test_scaled)
# print("SMOTE Model Accuracy:", accuracy_score(y_test, y_pred_smote))
#
# # Using XGBoost by encoding labels
# le = LabelEncoder()
# y_train_encoded = le.fit_transform(y_train)
# y_test_encoded = le.transform(y_test)
#
# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
# xgb_model.fit(X_train_scaled, y_train_encoded)
# y_pred_xgb = xgb_model.predict(X_test_scaled)
# print("XGBoost Accuracy:", accuracy_score(y_test_encoded, y_pred_xgb))
#
# # Save the best model and scaler
# joblib.dump(best_rf_model, 'best_glass_classification_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# print("Model and scaler saved successfully!")