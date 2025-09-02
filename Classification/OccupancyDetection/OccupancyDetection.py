# import pandas as  pd
# df_dataset= pd.read_csv('datatest.csv')
# df_dataset2= pd.read_csv('datatest2.csv')
# df_datatraining= pd.read_csv('datatraining.csv')
# #For dataset.csv
# print('#'*20,'For dataset.csv', '#'*20)
# print(df_dataset.describe(include='all').to_string())
# print(df_dataset.shape)
# print(df_dataset.columns)
# print(df_dataset.info)
# print(df_dataset.dtypes)
# print(df_dataset.isna().sum())
# print(df_dataset.head(10).to_string())
# print('='*70)
# # For dataset2.csv
# print('#'*20,'For dataset2.csv', '#'*20)
# print(df_dataset2.describe(include='all').to_string())
# print(df_dataset2.shape)
# print(df_dataset2.columns)
# print(df_dataset2.info)
# print(df_dataset2.dtypes)
# print(df_dataset2.isna().sum())
# print(df_dataset2.head(10).to_string())
# print('='*70)
# #For datatraining.cs
# print('#'*20,'For datatraining.csv', '#'*20)
# print(df_datatraining.describe(include='all').to_string())
# print(df_datatraining.shape)
# print(df_datatraining.columns)
# print(df_datatraining.info)
# print(df_datatraining.dtypes)
# print(df_datatraining.isna().sum())
# print(df_datatraining.head(10).to_string())
# print('='*70)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# from sklearn.model_selection import cross_val_score
#
# # 1. Load and combine datasets
# df1 = pd.read_csv('datatraining.csv')
# df2 = pd.read_csv('datatest.csv')
# df3 = pd.read_csv('datatest2.csv')
#
# # Add a source column to track original dataset
# df1['source'] = 'training'
# df2['source'] = 'test1'
# df3['source'] = 'test2'
#
# # Combine datasets
# df_combined = pd.concat([df1, df2, df3], ignore_index=True)
#
# # 2. Preprocess date column
# df_combined['date'] = pd.to_datetime(df_combined['date'], format='%m/%d/%Y %H:%M')
# df_combined['hour'] = df_combined['date'].dt.hour
# df_combined['day_of_week'] = df_combined['date'].dt.dayofweek
# df_combined['month'] = df_combined['date'].dt.month
#
# # Drop the original date column
# df_combined.drop('date', axis=1, inplace=True)
#
# # 3. EDA with plots
# plt.figure(figsize=(15, 10))
#
# # Distribution of Occupancy
# plt.subplot(2, 3, 1)
# sns.countplot(x='Occupancy', data=df_combined)
# plt.title('Distribution of Occupancy')
#
# # Correlation heatmap
# plt.subplot(2, 3, 2)
# numeric_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'hour', 'day_of_week', 'month']
# sns.heatmap(df_combined[numeric_cols + ['Occupancy']].corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
#
# # Boxplots for important features
# plt.subplot(2, 3, 3)
# sns.boxplot(x='Occupancy', y='Light', data=df_combined)
# plt.title('Light vs Occupancy')
#
# plt.subplot(2, 3, 4)
# sns.boxplot(x='Occupancy', y='CO2', data=df_combined)
# plt.title('CO2 vs Occupancy')
#
# plt.subplot(2, 3, 5)
# sns.boxplot(x='Occupancy', y='Temperature', data=df_combined)
# plt.title('Temperature vs Occupancy')
#
# plt.tight_layout()
# plt.show()
#
# # 4. Prepare data for modeling
# X = df_combined[numeric_cols]
# y = df_combined['Occupancy']
#
# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # Scale the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # 5. Implement and compare different models
# models = {
#     'Logistic Regression': LogisticRegression(),
#     'Decision Tree': DecisionTreeClassifier(),
#     'Random Forest': RandomForestClassifier(),
#     'SVM': SVC(),
#     'K-Nearest Neighbors': KNeighborsClassifier(),
#     'Gradient Boosting': GradientBoostingClassifier(),
#     'XGBoost': XGBClassifier()
# }
#
# results = {}
# for name, model in models.items():
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
#     accuracy = accuracy_score(y_test, y_pred)
#     results[name] = accuracy
#     print(f'{name} Accuracy: {accuracy:.4f}')
#     print(classification_report(y_test, y_pred))
#     print('-' * 50)
#
# # 6. Plot model comparison
# plt.figure(figsize=(10, 6))
# models_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
# sns.barplot(x='Accuracy', y='Model', data=models_df.sort_values('Accuracy', ascending=False))
# plt.title('Model Comparison - Accuracy Scores')
# plt.xlim(0, 1)
# plt.show()
#
# # 7. Feature importance from best model (Random Forest)
# best_model = RandomForestClassifier()
# best_model.fit(X_train_scaled, y_train)
# feature_importance = pd.DataFrame({
#     'feature': numeric_cols,
#     'importance': best_model.feature_importances_
# }).sort_values('importance', ascending=False)
#
# plt.figure(figsize=(10, 6))
# sns.barplot(x='importance', y='feature', data=feature_importance)
# plt.title('Feature Importance (Random Forest)')
# plt.show()
#
# # 8. Cross-validation for best model
# cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
# print(f'Cross-Validation Scores: {cv_scores}')
# print(f'Mean CV Accuracy: {cv_scores.mean():.4f}')
#
# import joblib
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
#
# # Load and preprocess data (same as before)
# df1 = pd.read_csv('datatraining.csv')
# df2 = pd.read_csv('datatest.csv')
# df3 = pd.read_csv('datatest2.csv')
#
# df1['source'] = 'training'
# df2['source'] = 'test1'
# df3['source'] = 'test2'
#
# df_combined = pd.concat([df1, df2, df3], ignore_index=True)
#
# # Preprocess date column
# df_combined['date'] = pd.to_datetime(df_combined['date'], format='%m/%d/%Y %H:%M')
# df_combined['hour'] = df_combined['date'].dt.hour
# df_combined['day_of_week'] = df_combined['date'].dt.dayofweek
# df_combined['month'] = df_combined['date'].dt.month
# df_combined.drop('date', axis=1, inplace=True)
#
# # Prepare features and target
# numeric_cols = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'hour', 'day_of_week', 'month']
# X = df_combined[numeric_cols]
# y = df_combined['Occupancy']
#
# # Scale the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Initialize all models
# models = {
#     'Logistic_Regression': LogisticRegression(),
#     'Decision_Tree': DecisionTreeClassifier(),
#     'Random_Forest': RandomForestClassifier(),
#     'SVM': SVC(probability=True),
#     'KNN': KNeighborsClassifier(),
#     'Gradient_Boosting': GradientBoostingClassifier(),
#     'XGBoost': XGBClassifier()
# }
#
# # Train and save all models
# for model_name, model in models.items():
#     print(f"Training {model_name}...")
#     model.fit(X_scaled, y)
#
#     # Save model
#     joblib.dump(model, f'{model_name}_model.pkl')
#     print(f"{model_name} saved successfully!")
#
# # Save the scaler
# joblib.dump(scaler, 'standard_scaler.pkl')
# print("Scaler saved successfully!")
#
# print("\nAll models and scaler have been saved successfully!")
#
# import joblib
# import pandas as pd
#
#
# def load_model(model_name):
#     """Load the desired model"""
#     return joblib.load(f'{model_name}_model.pkl')
#
#
# def load_scaler():
#     """Load Scaler"""
#     return joblib.load('standard_scaler.pkl')
#
#
# # مثال استفاده از مدل‌های مختلف برای پیش‌بینی
# def predict_occupancy(new_data, model_name='XGBoost'):
#     """
#     Predict occupancy with the given model
#
#     Parameters:
#     new_data (DataFrame): New data to predict
#     model_name (str): Model name ('XGBoost', 'Random_Forest', etc.)
#
#     Returns:
#     prediction: Prediction (0 or 1)
#     probability: Prediction probability
#     """
#     # Load model and scaler
#     model = load_model(model_name)
#     scaler = load_scaler()
#
#     new_data_scaled = scaler.transform(new_data)
#
#     prediction = model.predict(new_data_scaled)
#     probability = model.predict_proba(new_data_scaled)
#
#     return prediction[0], probability[0]
#
#
# # New data example for testing
# new_data_example = pd.DataFrame({
#     'Temperature': [23.0],
#     'Humidity': [26.0],
#     'Light': [500.0],
#     'CO2': [700.0],
#     'HumidityRatio': [0.0048],
#     'hour': [14],
#     'day_of_week': [2],
#     'month': [2]
# })
#
# model_names = ['Logistic_Regression', 'Decision_Tree', 'Random_Forest',
#                'SVM', 'KNN', 'Gradient_Boosting', 'XGBoost']
#
# print("Testing all models with sample data:")
# print("=" * 50)
#
# for model_name in model_names:
#     try:
#         prediction, probability = predict_occupancy(new_data_example, model_name)
#         status = "Occupied" if prediction == 1 else "Not Occupied"
#         confidence = probability[prediction]
#
#         print(f"{model_name:20}: {status:15} (Confidence: {confidence:.2%})")
#     except Exception as e:
#         print(f"{model_name:20}: Error - {str(e)}")