# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn.svm import SVC
# # from sklearn.metrics import accuracy_score, f1_score
# #
# # try:
# #     df = pd.read_csv('bupa.data.csv')
# # except FileNotFoundError:
# #     url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data'
# #     df = pd.read_csv(url, header=None)
# #     df.columns = ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks', 'Selector']
# #
# # X = df.drop('Selector', axis=1)  #  Selector ( Drinks)
# # y = df['Selector']  # target: Selector
# #
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #
# # scaler = StandardScaler()
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)
# #
# # clf_models = {
# #     'Logistic Regression': LogisticRegression(max_iter=200),
# #     'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
# #     'KNN': KNeighborsClassifier(n_neighbors=5),
# #     'SVC': SVC(kernel='rbf')
# # }
# #
# # for name, model in clf_models.items():
# #     model.fit(X_train_scaled, y_train)
# #     y_pred = model.predict(X_test_scaled)
# #     acc = accuracy_score(y_test, y_pred)
# #     f1 = f1_score(y_test, y_pred, average='weighted')
# #     print(f'{name}: Accuracy = {acc:.2f}, F1 Score = {f1:.2f}')
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from imblearn.over_sampling import SMOTE
#
# try:
#     df = pd.read_csv('bupa.data.csv')
# except FileNotFoundError:
#     url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data'
#     df = pd.read_csv(url, header=None)
#     df.columns = ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks', 'Selector']
#
# def remove_outliers_iqr(df, columns):
#     for col in columns:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR
#         df = df[(df[col] >= lower) & (df[col] <= upper)]
#     return df
#
# columns = ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks']
# df_clean = remove_outliers_iqr(df, columns)
# print(f"Shape after IQR: {df_clean.shape}")
#
# # Feature Engineering
# df_clean['Sgpt_Sgot_ratio'] = df_clean['Sgpt'] / (df_clean['Sgot'] + 1e-5)
# df_clean['Gammagt_Alkphos_ratio'] = df_clean['Gammagt'] / (df_clean['Alkphos'] + 1e-5)
# df_clean['Sgpt_Gammagt'] = df_clean['Sgpt'] * df_clean['Gammagt']
# features = ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks', 'Sgpt_Sgot_ratio', 'Gammagt_Alkphos_ratio', 'Sgpt_Gammagt']
#
# # تبدیل Selector به [0, 1]
# le = LabelEncoder()
# df_clean['Selector_encoded'] = le.fit_transform(df_clean['Selector'])
# X = df_clean[features]
# y = df_clean['Selector_encoded']
# print("Selector_encoded distribution:\n", y.value_counts())
#
# corr_matrix = df_clean.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.savefig('correlation_matrix.png')
# plt.show()
#
# # Feature Selection
# selector = SelectKBest(score_func=f_classif, k=8)
# X_selected = selector.fit_transform(X, y)
# selected_features = [features[i] for i in selector.get_support(indices=True)]
# print("Selected features:", selected_features)
#
# # Split
# X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.15, random_state=100, stratify=y)
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # SMOTE
# smote = SMOTE(random_state=42)
# X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
# print("After SMOTE, train Selector_encoded distribution:\n", pd.Series(y_train).value_counts())
#
# # Hyperparameter Tuning با Random Forest
# param_grid = {
#     'n_estimators': [300, 500, 700],
#     'max_depth': [10, 15, None],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }
# rf = RandomForestClassifier(random_state=42)
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# grid = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', error_score='raise')
# grid.fit(X_train_scaled, y_train)
# print("Best params:", grid.best_params_)
# print("Best CV score:", grid.best_score_)
#
# best_model = grid.best_estimator_
# y_pred = best_model.predict(X_test_scaled)
# acc = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"Final Model (Selector as Label): Accuracy = {acc:.2f}, F1 = {f1:.2f}")
#
# print("\n=== Final report for class project ===")
# print(f"Dataset: BUPA LiverDisorders (after removing outliers: {df_clean.shape[0]} sample)")
# print(f"Selected features: {selected_features}")
# print(f"Model: Random Forest with best parameters: {grid.best_params_}")
# print(f"Cross-Validation accuracy: {grid.best_score_:.2f}")
# print(f"Accuracy on Test Set: {acc:.2f}")
# print(f"F1 Score: {f1:.2f}")
# print("Note: Selector is used as a label which has no medical meaning in practice, but is suitable for class exercise.")
#
# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig('confusion_matrix.png')
# plt.show()
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from imblearn.over_sampling import SMOTE
# import joblib

# try:
#     df = pd.read_csv('bupa.data.csv')
# except FileNotFoundError:
#     url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data'
#     df = pd.read_csv(url, header=None)
#     df.columns = ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks', 'Selector']
#
# # Remove outliers with IQR
# def remove_outliers_iqr(df, columns):
#     for col in columns:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR
#         df = df[(df[col] >= lower) & (df[col] <= upper)]
#     return df
#
# columns = ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks']
# df_clean = remove_outliers_iqr(df, columns)
# print(f"Shape after IQR: {df_clean.shape}")
#
# # Feature Engineering
# df_clean['Sgpt_Sgot_ratio'] = df_clean['Sgpt'] / (df_clean['Sgot'] + 1e-5)
# df_clean['Gammagt_Alkphos_ratio'] = df_clean['Gammagt'] / (df_clean['Alkphos'] + 1e-5)
# df_clean['Sgpt_Gammagt'] = df_clean['Sgpt'] * df_clean['Gammagt']
# features = ['Mcv', 'Alkphos', 'Sgpt', 'Sgot', 'Gammagt', 'Drinks', 'Sgpt_Sgot_ratio', 'Gammagt_Alkphos_ratio', 'Sgpt_Gammagt']
#
# # Convert Selector to [0, 1]
# le = LabelEncoder()
# df_clean['Selector_encoded'] = le.fit_transform(df_clean['Selector'])
# X = df_clean[features]
# y = df_clean['Selector_encoded']
# print("Selector_encoded distribution:\n", y.value_counts())
#
# # EDA: Correlation Matrix
# corr_matrix = df_clean.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.savefig('correlation_matrix.png')
# plt.show()
#
# # Feature Selection
# selector = SelectKBest(score_func=f_classif, k=8)
# X_selected = selector.fit_transform(X, y)
# selected_features = [features[i] for i in selector.get_support(indices=True)]
# print("Selected features:", selected_features)
#
# # Split
# X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.15, random_state=100, stratify=y)
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # SMOTE
# smote = SMOTE(random_state=42)
# X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
# print("After SMOTE, train Selector_encoded distribution:\n", pd.Series(y_train).value_counts())
#
# # Hyperparameter Tuning با Random Forest
# param_grid = {
#     'n_estimators': [300, 500, 700],
#     'max_depth': [10, 15, None],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }
# rf = RandomForestClassifier(random_state=42)
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# grid = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', error_score='raise')
# grid.fit(X_train_scaled, y_train)
# print("Best params:", grid.best_params_)
# print("Best CV score:", grid.best_score_)
#
# best_model = grid.best_estimator_
# y_pred = best_model.predict(X_test_scaled)
# acc = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"Final Model (Selector as Label): Accuracy = {acc:.2f}, F1 = {f1:.2f}")
#
# Save model, scaler and selector
# joblib.dump(best_model, 'liver_disorders_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(selector, 'selector.pkl')
# joblib.dump(le, 'label_encoder.pkl')
# print("Model, scaler, selector and label encoder saved successfully!")
#
# # Report for class presentation
# print("\n=== Final report for class project ===")
# print(f"Dataset: BUPA Liver Disorders (after removing outliers: {df_clean.shape[0]} sample)")
# print(f"Selected features: {selected_features}")
# print(f"Model: Random Forest with best parameters: {grid.best_params_}")
# print(f"Cross-Validation accuracy: {grid.best_score_:.2f}")
# print(f"Accuracy on Test Set: {acc:.2f}")
# print(f"F1 Score: {f1:.2f}")
# print("Note: Selector is used as a label which has no medical meaning in practice, but is suitable for class practice.")
# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig('confusion_matrix.png')
# plt.show()
#
# # Sample code for loading and predicting
# print("\n=== Sample code for loading and predicting ===")