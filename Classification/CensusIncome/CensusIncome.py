# import warnings
# warnings.filterwarnings("ignore")
#
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import xgboost as xgb
# from sklearn.metrics import accuracy_score, classification_report
#
# # -----------------------
# # File paths and column names
# # -----------------------
# train_path = 'adult.data.csv'
# test_path  = 'adult.test.csv'
#
# cols = [
#     'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
#     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
#     'hours-per-week', 'native-country', 'income'
# ]
#
# # ------------ Read files ------------
# # Use names to ensure columns are properly named.
# # skipinitialspace removes extra spaces after commas.
# train_df = pd.read_csv(train_path, header=None, names=cols, na_values='?', skipinitialspace=True)
# test_df  = pd.read_csv(test_path,  header=None, names=cols, na_values='?', skipinitialspace=True, comment='|')
#
# # In some versions of adult.test, the first row may contain headers or extra values.
# # Initial cleanup: Remove rows where columns mistakenly contain column names
# for c in cols:
#     # Check if string matches column name (case-insensitive)
#     mask = test_df[c].astype(str).str.strip().str.lower() == c.lower()
#     if mask.any():
#         test_df = test_df[~mask]
#
# # Temporarily combine for consistent processing
# full_df = pd.concat([train_df, test_df], ignore_index=True)
#
# # Remove rows that are completely empty or all values are NaN
# full_df = full_df.dropna(how='all').reset_index(drop=True)
#
# # Some corrupt rows may contain text like 'age' or other headers in any column.
# # For safety: Remove rows where any column value exactly matches the column name.
# for c in cols:
#     full_df = full_df[full_df[c].astype(str).str.strip().str.lower() != c.lower()]
#
# full_df = full_df.reset_index(drop=True)
#
# # ------------- Convert numeric columns to numbers and remove corrupt rows -------------
# numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
# # Attempt to convert all numeric_cols to numbers; non-convertible values => NaN
# for nc in numeric_cols:
#     full_df[nc] = pd.to_numeric(full_df[nc], errors='coerce')
#
# # Optionally, you can impute NaN values; here we take the simplest approach: remove rows with NaN in numeric columns
# before = len(full_df)
# full_df = full_df.dropna(subset=numeric_cols).reset_index(drop=True)
# after = len(full_df)
# print(f"Dropped {before - after} rows because numeric columns contained invalid values.")
#
# # For categorical columns, fill NaN with mode (if any)
# categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
#                     'relationship', 'race', 'sex', 'native-country', 'income']
# for c in categorical_cols:
#     if full_df[c].isna().sum() > 0:
#         full_df[c] = full_df[c].fillna(full_df[c].mode()[0])
#
# # ----------------- Encoding: One LabelEncoder for each categorical column -----------------
# encoders = {}
# from sklearn.preprocessing import LabelEncoder
# for col in ['workclass', 'education', 'marital-status', 'occupation',
#             'relationship', 'race', 'sex', 'native-country']:
#     le = LabelEncoder()
#     full_df[col] = full_df[col].astype(str)
#     le.fit(full_df[col])
#     full_df[col] = le.transform(full_df[col])
#     encoders[col] = le
#
# # Encode income as well
# income_le = LabelEncoder()
# full_df['income'] = income_le.fit_transform(full_df['income'].astype(str))
# encoders['income'] = income_le
#
# # -------------- Split back into train/test with original lengths --------------
# train_len = sum(1 for _ in open(train_path))  # Number of rows in the original training file (including any unusual headers)
# # Instead of calculating file length, it's better to use the initial train_df length; some corrupt rows may have been removed.
# # For precision: Use the length of train_df as read initially, which we have:
# train_rows = train_df.shape[0]
# # Now, considering some rows from test_df were removed, split based on the original train_df length:
# train_processed = full_df.iloc[:train_rows].reset_index(drop=True)
# test_processed  = full_df.iloc[train_rows:].reset_index(drop=True)
#
# # --------- Prepare feature matrix and target ---------
# feature_names = [
#     'age', 'workclass', 'fnlwgt', 'education', 'education-num',
#     'marital-status', 'occupation', 'relationship', 'race', 'sex',
#     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
# ]
#
# # Ensure all columns exist
# missing = [c for c in feature_names if c not in train_processed.columns]
# if missing:
#     raise ValueError("Missing features in processed dataframe: " + ", ".join(missing))
#
# X_train = train_processed[feature_names].astype(float)
# y_train = train_processed['income'].astype(int)
# X_test  = test_processed[feature_names].astype(float)
# y_test  = test_processed['income'].astype(int)
#
# # ---------- Standardization ----------
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)
#
# # ---------- Train model ----------
# model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
# model.fit(X_train_scaled, y_train)
#
# # ---------- Evaluation ----------
# y_pred = model.predict(X_test_scaled)
# print("Accuracy on test:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#
# # ---------- Save files ----------
# joblib.dump(model, 'best_xgboost_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(encoders, 'encoders.pkl')
# joblib.dump(feature_names, 'feature_names.pkl')
#
# print("Saved: best_xgboost_model.pkl, scaler.pkl, encoders.pkl, feature_names.pkl")