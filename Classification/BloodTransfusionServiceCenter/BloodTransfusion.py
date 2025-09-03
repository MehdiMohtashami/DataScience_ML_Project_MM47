# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.preprocessing import PowerTransformer
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
# import joblib
# import json
#
# # Load data
# df = pd.read_csv('transfusion.data.csv')
#
# # Feature Engineering
# df['donation_ratio'] = df['Frequency'] / (df['Time'] + 1)
# df['avg_donation_amount'] = df['Monetary'] / (df['Frequency'] + 1)
# df['recency_ratio'] = df['Recency'] / (df['Time'] + 1)
#
# # Prepare data
# X = df.drop('donated blood', axis=1)
# y = df['donated blood']
#
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                     random_state=42,
#                                                     stratify=y)
#
# # Feature scaling
# scaler = PowerTransformer(method='yeo-johnson')
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Train and evaluate Logistic Regression
# print("=== Logistic Regression ===")
# lr_model = LogisticRegression(class_weight='balanced', C=0.1, solver='liblinear', random_state=42)
# lr_model.fit(X_train_scaled, y_train)
#
# # Cross-validation
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
# print(f"CV ROC AUC: {lr_cv_scores.mean():.4f} (± {lr_cv_scores.std():.4f})")
#
# # Test evaluation
# y_pred_lr = lr_model.predict(X_test_scaled)
# y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
#
# print(f"Test Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
# print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")
# print(f"Test F1 Score: {f1_score(y_test, y_pred_lr):.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_lr))
#
# # Train and evaluate SVM
# print("\n=== SVM ===")
# svm_model = SVC(class_weight='balanced', C=1.0, kernel='rbf',
#                 gamma='scale', probability=True, random_state=42)
# svm_model.fit(X_train_scaled, y_train)
#
# # Cross-validation
# svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
# print(f"CV ROC AUC: {svm_cv_scores.mean():.4f} (± {svm_cv_scores.std():.4f})")
#
# # Test evaluation
# y_pred_svm = svm_model.predict(X_test_scaled)
# y_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]
#
# print(f"Test Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
# print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba_svm):.4f}")
# print(f"Test F1 Score: {f1_score(y_test, y_pred_svm):.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_svm))
#
# # Save models and scaler
# print("\n=== Saving Models ===")
# joblib.dump(lr_model, 'logistic_regression_model.pkl')
# joblib.dump(svm_model, 'svm_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
#
# # Save feature names
# feature_info = {
#     'feature_names': X.columns.tolist(),
#     'target_name': 'donated blood'
# }
#
# with open('feature_info.json', 'w') as f:
#     json.dump(feature_info, f)
#
# print("Models, scaler, and feature information saved successfully!")
#
#
# # Create a function for prediction
# def predict_donation(model_name, recency, frequency, monetary, time):
#     """
#     Predict blood donation probability
#
#     Parameters:
#     model_name: 'lr' for Logistic Regression, 'svm' for SVM
#     recency: Months since last donation
#     frequency: Total number of donations
#     monetary: Total blood donated in c.c.
#     time: Months since first donation
#     """
#     # Calculate derived features
#     donation_ratio = frequency / (time + 1)
#     avg_donation_amount = monetary / (frequency + 1)
#     recency_ratio = recency / (time + 1)
#
#     # Create feature array
#     features = np.array([[recency, frequency, monetary, time,
#                           donation_ratio, avg_donation_amount, recency_ratio]])
#
#     # Load scaler and transform features
#     scaler = joblib.load('scaler.pkl')
#     features_scaled = scaler.transform(features)
#
#     # Load model and predict
#     if model_name == 'lr':
#         model = joblib.load('logistic_regression_model.pkl')
#     else:
#         model = joblib.load('svm_model.pkl')
#
#     probability = model.predict_proba(features_scaled)[0, 1]
#     prediction = model.predict(features_scaled)[0]
#
#     return {
#         'prediction': int(prediction),
#         'probability': float(probability),
#         'will_donate': bool(prediction)
#     }
#
#
# # Test the prediction function
# print("\n=== Testing Prediction Function ===")
# test_prediction = predict_donation('lr', 2, 50, 12500, 98)
# print(f"Test prediction: {test_prediction}")
#
# # Create comparison plot
# plt.figure(figsize=(10, 6))
# models = ['Logistic Regression', 'SVM']
# cv_scores = [lr_cv_scores.mean(), svm_cv_scores.mean()]
# cv_std = [lr_cv_scores.std(), svm_cv_scores.std()]
#
# plt.bar(models, cv_scores, yerr=cv_std, capsize=5, alpha=0.7)
# plt.ylabel('ROC AUC Score')
# plt.title('Model Comparison (5-Fold Cross Validation)')
# plt.ylim(0.6, 0.8)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig('model_comparison.png')
# plt.show()
#
# # Feature importance for Logistic Regression
# if hasattr(lr_model, 'coef_'):
#     plt.figure(figsize=(10, 6))
#     feature_importance = pd.DataFrame({
#         'feature': X.columns,
#         'importance': np.abs(lr_model.coef_[0])
#     }).sort_values('importance', ascending=False)
#
#     sns.barplot(data=feature_importance, x='importance', y='feature')
#     plt.title('Feature Importance - Logistic Regression')
#     plt.tight_layout()
#     plt.savefig('feature_importance.png')
#     plt.show()