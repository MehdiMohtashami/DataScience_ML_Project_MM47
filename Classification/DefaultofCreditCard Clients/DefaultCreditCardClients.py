# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve
# import joblib
# import warnings
# warnings.filterwarnings('ignore')
#
# # Load data
# df = pd.read_csv('default of credit card clients.csv')
#
# # Initial exploration
# print("Dataset shape:", df.shape)
# print("\nMissing values:\n", df.isnull().sum())
# print("\nData types:\n", df.dtypes)
#
# # Examine distribution of target variable
# plt.figure(figsize=(10, 6))
# sns.countplot(x='default payment next month', data=df)
# plt.title('Distribution of Target Variable')
# plt.show()
#
# # Examine correlation
# plt.figure(figsize=(20, 15))
# correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.show()
#
# # Drop ID column as it is not needed
# df = df.drop('ID', axis=1)
#
# # Check and correct values in categorical columns
# print("Unique values in EDUCATION:", df['EDUCATION'].unique())
# print("Unique values in MARRIAGE:", df['MARRIAGE'].unique())
#
# # Correct out-of-range values (based on data dictionary)
# df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)  # Convert all invalid values to 4 (others)
# df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)  # Convert 0 to 3 (others)
#
# # Separating features and target
# X = df.drop('default payment next month', axis=1)
# y = df['default payment next month']
#
# # Standardize numerical data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Split data into train and test
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
#                                                     random_state=42, stratify=y)
#
# # Define different models
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
#     'Random Forest': RandomForestClassifier(random_state=42),
#     'Gradient Boosting': GradientBoostingClassifier(random_state=42),
#     'SVM': SVC(probability=True, random_state=42)
# }
#
# # Evaluate models with cross-validation
# results = {}
# for name, model in models.items():
#     cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
#     results[name] = cv_scores
#     print(f"{name}: AUC = {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
#
# # Visualize model comparison
# plt.figure(figsize=(12, 8))
# plt.boxplot(results.values(), labels=results.keys())
# plt.title('Comparison of Algorithm Performance')
# plt.ylabel('ROC AUC Score')
# plt.xticks(rotation=45)
# plt.show()
#
# # Select the best model based on results
# best_model = GradientBoostingClassifier(random_state=42)
#
# # Hyperparameter tuning
# param_grid = {
#     'n_estimators': [100, 200],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 4, 5],
#     'subsample': [0.8, 0.9, 1.0]
# }
#
# grid_search = GridSearchCV(best_model, param_grid, cv=5,
#                           scoring='roc_auc', n_jobs=-1, verbose=1)
# grid_search.fit(X_train, y_train)
#
# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))
#
# # Train final model with best parameters
# final_model = grid_search.best_estimator_
# final_model.fit(X_train, y_train)
#
# # Predict on test data
# y_pred = final_model.predict(X_test)
# y_pred_proba = final_model.predict_proba(X_test)[:, 1]
#
# # Calculate evaluation metrics
# print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
# print("ROC AUC: {:.4f}".format(roc_auc_score(y_test, y_pred_proba)))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
#
# # Confusion matrix
# plt.figure(figsize=(8, 6))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()
#
# # ROC curve
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc_score(y_test, y_pred_proba))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend(loc="lower right")
# plt.show()
#
# joblib.dump(final_model, 'best_credit_card_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
#
# print("Model saved successfully!")