# # Basic Libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Preprocessing
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
#
# # Classification Models
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
#
# # Metrics
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
#
# # Miscellaneous
# import warnings
# warnings.filterwarnings('ignore')
#
# # Load the dataset
# df = pd.read_csv('processed.cleveland.csv')
#
# # Convert the problem to binary classification (0 vs. >0)
# # This is a common practice with this dataset
# df['target'] = (df['num'] > 0).astype(int)
#
# # Drop the original 'num' column
# df = df.drop('num', axis=1)
#
# # Handle missing values (简单处理)
# # Since there are only a few missing values, we can use median/mode imputation
# # But we'll do it properly in the pipeline later
# print("Missing values:\n", df.isna().sum())
#
# # Set style for plots
# sns.set_style("whitegrid")
# # plt.figure(figsize=(12, 8))
#
# # 1. Check the distribution of the target variable
# # plt.subplot(2, 3, 1)
# sns.countplot(x='target', data=df)
# plt.title('Distribution of Target Variable')
# # plt.savefig('Distribution_TargetVariable.png')
# plt.show()
#
# # 2. Age distribution by target
# # plt.subplot(2, 3, 2)
# sns.histplot(data=df, x='age', hue='target', kde=True)
# plt.title('Age Distribution by Target')
# # plt.savefig('Age_Distribution_Target.png')
# plt.show()
#
# # 3. Cholesterol distribution by target
# # plt.subplot(2, 3, 3)
# sns.histplot(data=df, x='chol', hue='target', kde=True)
# plt.title('Cholesterol Distribution by Target')
# # plt.savefig('Cholesterol_Distribution_Target.png')
# plt.show()
#
# # 4. Correlation heatmap
# # plt.subplot(2, 3, 4)
# # corr_matrix = df.corr()
# # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
# # plt.title('Correlation Heatmap')
# # plt.savefig('CorrelationHeatmap.png')
# # plt.show()
#
# # 5. Chest pain type vs target
# # plt.subplot(2, 3, 5)
# sns.countplot(x='cp', hue='target', data=df)
# plt.title('Chest Pain Type vs Target')
# # plt.savefig('ChestPainType_Target.png')
# plt.show()
#
# # 6. Thalach (max heart rate) by target
# # plt.subplot(2, 3, 6)
# sns.boxplot(x='target', y='thalach', data=df)
# plt.title('Max Heart Rate by Target')
# # plt.savefig('MaxHeart_RateTarget.png')
# plt.show()
#
# # Separate features and target
# X = df.drop('target', axis=1)
# y = df['target']
#
# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # Define numeric and categorical features
# numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
# categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
#
# # Create preprocessing pipelines
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])
#
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# # Combine preprocessing steps
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])
#
# # Define models to test
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
#     'Decision Tree': DecisionTreeClassifier(random_state=42),
#     'Random Forest': RandomForestClassifier(random_state=42),
#     'SVM': SVC(random_state=42),
#     'K-NN': KNeighborsClassifier(),
#     'Gradient Boosting': GradientBoostingClassifier(random_state=42)
# }
#
# # Evaluate each model
# results = {}
# for name, model in models.items():
#     # Create pipeline
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('classifier', model)
#     ])
#
#     # Train and evaluate
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#
#     # Cross-validation for more robust evaluation
#     cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
#
#     results[name] = {
#         'model': pipeline,
#         'accuracy': accuracy,
#         'cv_mean': cv_scores.mean(),
#         'cv_std': cv_scores.std()
#     }
#
#     print(f"{name}:")
#     print(f"  Accuracy: {accuracy:.4f}")
#     print(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
#     print(classification_report(y_test, y_pred))
#     print("-" * 50)
#
# # Based on results, let's say Random Forest was the best
# # Now let's tune its hyperparameters
#
# # Define parameter grid
# param_grid = {
#     'classifier__n_estimators': [100, 200, 300],
#     'classifier__max_depth': [None, 10, 20, 30],
#     'classifier__min_samples_split': [2, 5, 10]
# }
#
# # Create the pipeline
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', RandomForestClassifier(random_state=42))
# ])
#
# # Grid search
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train, y_train)
#
# # Best model
# best_model = grid_search.best_estimator_
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
#
# print(f"Best parameters: {best_params}")
# print(f"Best cross-validation score: {best_score:.4f}")
#
# # Evaluate on a test set
# y_pred = best_model.predict(X_test)
# print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
# print(classification_report(y_test, y_pred))
#
# # Get feature names after preprocessing
# preprocessor.fit(X)
# feature_names = (numeric_features +
#                  list(preprocessor.named_transformers_['cat']
#                      .named_steps['encoder']
#                      .get_feature_names_out(categorical_features)))
#
# # Get feature importances from the best model
# importances = best_model.named_steps['classifier'].feature_importances_
#
# # Create a DataFrame for visualization
# feature_importance_df = pd.DataFrame({
#     'feature': feature_names,
#     'importance': importances
# }).sort_values('importance', ascending=False)
#
# # Plot feature importances
# plt.figure(figsize=(12, 8))
# sns.barplot(x='importance', y='feature', data=feature_importance_df)
# plt.title('Feature Importances')
# plt.tight_layout()
# plt.show()
#
# import joblib
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
#
# # Training and storing models
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
#     'SVM': SVC(random_state=42, probability=True),
#     'K-NN': KNeighborsClassifier()
# }
#
# # Create a simple preprocessor
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numeric_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ])
#
# for model_name, model in models.items():
#     # Create pipeline
#     pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('classifier', model)
#     ])
#
#     # Model training
#     pipeline.fit(X_train, y_train)
#
#     # Save model
#     filename = f'{model_name.replace(" ", "_").lower()}_model.joblib'
#     joblib.dump(pipeline, filename)
#     print(f'{model_name} saved as {filename}')
#
#     # Model evaluation
#     accuracy = pipeline.score(X_test, y_test)
#     print(f'{model_name} Accuracy: {accuracy:.4f}')