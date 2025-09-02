# # Basic Libraries
# import pandas as pd
# import numpy as np
#
# # Visualization Libraries
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Data Preprocessing
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
#
# # Classification Models
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
#
# # Evaluation Metrics
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
#
# # Ignore Warnings
# import warnings
# warnings.filterwarnings('ignore')
#
# # Load the dataset
# df = pd.read_csv('hepatitis.csv')
#
# # 1. Check the balance of the Target Variable
# plt.figure(figsize=(6, 4))
# sns.countplot(x='Class', data=df)
# plt.title('Distribution of Target Variable (Class)')
# plt.xlabel('Class (1:Die, 2:Live)')
# plt.ylabel('Count')
# plt.show()
#
# # 2. Check Correlation Matrix (for numeric variables)
# # First we separate the list of numeric columns
# numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
# # Remove the Target column from the list of attributes
# numeric_features.remove('Class')
#
# plt.figure(figsize=(16, 12))
# sns.heatmap(df[numeric_features + ['Class']].corr(), annot=True, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.show()
#
# # 3. Distribution of Numeric Features
# df[numeric_features].hist(bins=15, figsize=(15, 10), layout=(5, 4))
# plt.suptitle('Distribution of Numeric Features')
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()
#
# # 4. Boxplots to check for outliers
# plt.figure(figsize=(15, 10))
# for i, col in enumerate(numeric_features):
#     plt.subplot(5, 4, i+1)
#     sns.boxplot(y=df[col])
#     plt.title(col)
# plt.tight_layout()
# plt.show()
#
# # Separate features and target
# X = df.drop('Class', axis=1)
# y = df['Class']
#
# # Split data to Train and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratify برای حفظ比例 توزیع Class
#
# # Define which features are categorical and which are numeric
# # (According to the Data Dictionary, even though they are numbers, some are categorical)
# categorical_features = ['Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia', 'Liver Big', 'Liver Firm', 'Spleen Palpable', 'Spiders', 'Ascites', 'Varices', 'Histology']
# numeric_features = ['Age', 'Bilirubin', 'Alk Phosphate', 'Sgot', 'Albumin', 'Protime']
#
# # Create preprocessors for numeric and categorical data
# # For numeric data: Fill Missing Values with median and then Scale.
# # For categorical data: Fill Missing Values with the most frequent value (mode). Since their value is numeric, there is no need for One-Hot Encoding.
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')), # or KNNImputer(n_neighbors=5)
#     ('scaler', StandardScaler()) # یا MinMaxScaler
# ])
#
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
# # Since the values are already numeric, no encoder is needed.
# ])
#
# # Combine preprocessors using ColumnTransformer
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])
#
# # Transform the data
# X_train_preprocessed = preprocessor.fit_transform(X_train)
# X_test_preprocessed = preprocessor.transform(X_test)
#
# # (Optional) If we want to see the processed data in a DataFrame
# # Since ColumnTransformer destroys the column names, we recreate them
# feature_names = numeric_features + categorical_features
# X_train_processed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
# print(X_train_processed_df.head())
# print("\nMissing values in processed training set:", np.isnan(X_train_preprocessed).sum())
#
# # Define a list of models to evaluate
# models = {
#     'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
#     'K-Nearest Neighbors': KNeighborsClassifier(),
#     'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
#     'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
#     'Gradient Boosting': GradientBoostingClassifier(random_state=42),
#     'SVM': SVC(random_state=42, class_weight='balanced')
# }
#
# # Evaluate each model using Cross-Validation on the training set
# results = {}
# for name, model in models.items():
#     # Create a full pipeline including the preprocessor and the model
#     pipeline = Pipeline(steps=[('preprocessor', preprocessor),
#                                ('classifier', model)])
#
#     # Perform cross-validation (5-fold)
#     cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
#     results[name] = cv_scores
#     print(f"{name:20} | Cross-Val Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
#
# # Visualize the results for comparison
# results_df = pd.DataFrame(results)
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=results_df)
# plt.title('Comparison of Classification Algorithm Accuracy (5-Fold CV)')
# plt.ylabel('Accuracy')
# plt.xticks(rotation=45)
# plt.show()
#
# # Define the best model pipeline
# best_model = Pipeline(steps=[('preprocessor', preprocessor),
#                              ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))])
#
# # Define hyperparameter grid for tuning
# param_grid = {
#     'classifier__n_estimators': [50, 100, 200],
#     'classifier__max_depth': [None, 10, 20],
#     'classifier__min_samples_split': [2, 5, 10]
# }
#
# # Perform Grid Search CV
# grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
# grid_search.fit(X_train, y_train)
#
# # Print the best parameters and score
# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
#
# # Get the best model
# tuned_model = grid_search.best_estimator_
#
# # Final Prediction on the Test Set
# y_pred = tuned_model.predict(X_test)
# y_pred_proba = tuned_model.predict_proba(X_test)[:, 1] # برای محاسبه ROC-AUC
#
# # 1. Accuracy and Classification Report
# print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
#
# # 2. Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap='Blues')
# plt.title('Confusion Matrix')
# plt.show()
#
# # 3. ROC Curve and AUC (if the model has probability)
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=2) # pos_label=2 means the "live" class is considered a positive class.
# roc_auc = roc_auc_score(y_test, y_pred_proba)
#
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()
#
# # --- Train and Save the BEST Models ---
#
# # Let's train both KNN and Tuned Random Forest on the FULL training data and save them.
#
# # 1. Create the preprocessor and fit it on the FULL training data
# # (This is important to have all the parameters fitted on the maximum amount of data)
# preprocessor.fit(X_train, y_train)
#
# # 2. Define the best models with their parameters
# best_knn = KNeighborsClassifier()
# best_rf = RandomForestClassifier(random_state=42, class_weight='balanced',
#                                  n_estimators=100, max_depth=None, min_samples_split=10)
#
# # 3. Create full pipelines for each model
# pipeline_knn = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', best_knn)
# ])
#
# pipeline_rf = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', best_rf)
# ])
#
# # 4. Train them on the FULL training data
# print("Training KNN on full training data...")
# pipeline_knn.fit(X_train, y_train)
#
# print("Training Tuned Random Forest on full training data...")
# pipeline_rf.fit(X_train, y_train)
#
# # 5. Save the models using joblib
# import joblib
#
# joblib.dump(pipeline_knn, 'hepatitis_knn_model.pkl')
# print("KNN model saved as 'hepatitis_knn_model.pkl'")
#
# joblib.dump(pipeline_rf, 'hepatitis_rf_model.pkl')
# print("Random Forest model saved as 'hepatitis_rf_model.pkl'")
#
# # (Optional) Save the preprocessor separately as well, in case you need it for new models
# joblib.dump(preprocessor, 'hepatitis_preprocessor.pkl')
# print("Preprocessor saved as 'hepatitis_preprocessor.pkl'")
#
# print("\nBoth models are ready for use in a UI or any other application!")
#
# import joblib
# import pandas as pd
#
# # Load the model
# loaded_model = joblib.load('hepatitis_knn_model.pkl') # یا 'hepatitis_rf_model.pkl'
#
# # Create a sample new patient data (must have all 19 features in the correct order)
# # This order of columns should be exactly the same as the original data: 'Age', 'Sex', 'Steroid', ...
# new_patient_data = pd.DataFrame([{
#     'Age': 45,
#     'Sex': 1,
#     'Steroid': 2,
#     'Antivirals': 2,
#     'Fatigue': 1,
#     'Malaise': 2,
#     'Anorexia': 1,
#     'Liver Big': 2,
#     'Liver Firm': 1,
#     'Spleen Palpable': 2,
#     'Spiders': 1,
#     'Ascites': 1,
#     'Varices': 1,
#     'Bilirubin': 0.8,
#     'Alk Phosphate': 95.0,
#     'Sgot': 40.0,
#     'Albumin': 4.0,
#     'Protime': 70.0,
#     'Histology': 1
# }])
#
# # Make a prediction
# prediction = loaded_model.predict(new_patient_data)
# prediction_proba = loaded_model.predict_proba(new_patient_data)
#
# # Convert prediction to a meaningful result
# result = "LIVE" if prediction[0] == 2 else "DIE"
# confidence = prediction_proba[0].max() * 100
#
# print(f"The model predicts: {result}")
# print(f"Confidence: {confidence:.2f}%")