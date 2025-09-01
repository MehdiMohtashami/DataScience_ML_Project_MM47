# # import pandas as  pd
# # df_dataset= pd.read_csv('Autism-Adult-Data.csv')
# # #For Autism-Adult-Data.csv
# # print('#'*20,'For Autism-Adult-Data.csv', '#'*20)
# # print(df_dataset.describe(include='all').to_string())
# # print(df_dataset.shape)
# # print(df_dataset.columns)
# # print(df_dataset.info)
# # print(df_dataset.dtypes)
# # print(df_dataset.isna().sum())
# # print(df_dataset.head(10).to_string())
# # print('='*70)
# # Import necessary libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # Load the dataset
# df = pd.read_csv('Autism-Adult-Data.csv')
#
# # Initial data analysis
# print("=" * 50)
# print("INITIAL DATA ANALYSIS")
# print("=" * 50)
# print(f"Dataset Shape: {df.shape}")
# print("\nData Types:")
# print(df.dtypes)
# print("\nMissing Values:")
# print(df.isna().sum())
# print("\nFirst 10 Rows:")
# print(df.head(10))
#
# # Handle missing values
# print("\n" + "=" * 50)
# print("HANDLING MISSING VALUES")
# print("=" * 50)
#
# # Fill age with median
# age_median = df['age'].median()
# df['age'].fillna(age_median, inplace=True)
# print(f"Filled {df['age'].isna().sum()} missing values in 'age' with median: {age_median}")
#
# # Fill categorical columns with 'Unknown'
# df['ethnicity'].fillna('Unknown', inplace=True)
# df['relation'].fillna('Unknown', inplace=True)
# print(f"Filled missing values in 'ethnicity' and 'relation' with 'Unknown'")
#
# # Check if any missing values remain
# print(f"Remaining missing values: {df.isna().sum().sum()}")
#
# # Data Preprocessing
# print("\n" + "=" * 50)
# print("DATA PREPROCESSING")
# print("=" * 50)
#
# # Remove unnecessary columns (age_desc has only one value)
# df.drop('age_desc', axis=1, inplace=True)
# print("Removed 'age_desc' column")
#
# # Convert binary categorical columns to numerical
# binary_cols = ['gender', 'jundice', 'austim', 'used_app_before', 'Class/ASD']
# le = LabelEncoder()
# for col in binary_cols:
#     df[col] = le.fit_transform(df[col])
#     print(f"Converted {col} to numerical values")
#
# # One-Hot Encoding for other categorical columns
# categorical_cols = ['ethnicity', 'contry_of_res', 'relation']
# df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# print(f"Applied One-Hot Encoding to: {categorical_cols}")
#
# print(f"Final dataset shape: {df.shape}")
#
# # Exploratory Data Analysis (EDA)
# print("\n" + "=" * 50)
# print("EXPLORATORY DATA ANALYSIS (EDA)")
# print("=" * 50)
#
# # Set style for plots
# sns.set_style("whitegrid")
# plt.figure(figsize=(12, 8))
#
# # 1. Age Distribution
# plt.subplot(2, 2, 1)
# sns.histplot(df['age'], bins=30, kde=True)
# plt.title('Age Distribution')
# plt.xlabel('Age')
#
# # 2. Class Distribution
# plt.subplot(2, 2, 2)
# class_counts = df['Class/ASD'].value_counts()
# plt.pie(class_counts, labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90)
# plt.title('Class Distribution (ASD)')
#
# # 3. Correlation Heatmap (first 20 features for readability)
# plt.subplot(2, 2, 3)
# corr_matrix = df.corr().abs()
# sns.heatmap(corr_matrix.iloc[:20, :20], cmap='coolwarm', center=0)
# plt.title('Correlation Matrix (First 20 Features)')
#
# # 4. Boxplot of Age by ASD Class
# plt.subplot(2, 2, 4)
# sns.boxplot(x=df['Class/ASD'], y=df['age'])
# plt.title('Age Distribution by ASD Class')
# plt.xticks([0, 1], ['No', 'Yes'])
#
# plt.tight_layout()
# plt.show()
#
# # Additional EDA - Question Scores Analysis
# plt.figure(figsize=(15, 10))
# question_cols = [f'A{i}_Score' for i in range(1, 11)]
# asd_yes = df[df['Class/ASD'] == 1]
# asd_no = df[df['Class/ASD'] == 0]
#
# for i, col in enumerate(question_cols, 1):
#     plt.subplot(3, 4, i)
#     yes_mean = asd_yes[col].mean()
#     no_mean = asd_no[col].mean()
#     plt.bar(['No ASD', 'ASD'], [no_mean, yes_mean])
#     plt.title(f'{col} Mean Score')
#     plt.ylabel('Mean Score')
#
# plt.tight_layout()
# plt.show()
#
# # Prepare data for modeling
# X = df.drop('Class/ASD', axis=1)
# y = df['Class/ASD']
#
# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # Standardize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
# print(f"Class distribution in training set: {np.bincount(y_train)}")
# print(f"Class distribution in test set: {np.bincount(y_test)}")
#
# # Model Training and Evaluation
# print("\n" + "=" * 50)
# print("MODEL TRAINING AND EVALUATION")
# print("=" * 50)
#
# models = {
#     "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced'),
#     "K-Nearest Neighbors": KNeighborsClassifier(),
#     "Support Vector Machine": SVC(random_state=42, class_weight='balanced'),
#     "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
#     "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
#     "Gradient Boosting": GradientBoostingClassifier(random_state=42),
#     "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
# }
#
# results = {}
# for name, model in models.items():
#     # Train the model
#     model.fit(X_train_scaled, y_train)
#
#     # Make predictions
#     y_pred = model.predict(X_test_scaled)
#
#     # Calculate accuracy
#     acc = accuracy_score(y_test, y_pred)
#     results[name] = acc
#
#     # Print results
#     print(f"\n{name}")
#     print("-" * len(name))
#     print(f"Accuracy: {acc:.4f}")
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))
#
#     # Plot confusion matrix
#     plt.figure(figsize=(5, 4))
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['No ASD', 'ASD'],
#                 yticklabels=['No ASD', 'ASD'])
#     plt.title(f'Confusion Matrix - {name}')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.show()
#
# # Compare model performance
# print("\n" + "=" * 50)
# print("MODEL COMPARISON")
# print("=" * 50)
#
# # Create a DataFrame for results comparison
# results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
# results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
#
# print("Models ranked by accuracy:")
# print(results_df)
#
# # Plot model comparison
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')
# plt.title('Model Comparison - Accuracy Scores')
# plt.xlim(0, 1)
# plt.xlabel('Accuracy')
# plt.tight_layout()
# plt.show()
#
# # Feature importance for the best tree-based model
# best_model_name = results_df.iloc[0]['Model']
# best_model = models[best_model_name]
#
# if hasattr(best_model, 'feature_importances_'):
#     print(f"\nFeature Importance from {best_model_name}:")
#     feature_importance = pd.DataFrame({
#         'feature': X.columns,
#         'importance': best_model.feature_importances_
#     })
#     feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
#
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='importance', y='feature', data=feature_importance, palette='magma')
#     plt.title(f'Top 10 Feature Importance - {best_model_name}')
#     plt.tight_layout()
#     plt.show()
#
#     print(feature_importance)
#
# print("\n" + "=" * 50)
# print("CONCLUSION")
# print("=" * 50)
# print(f"Best performing model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")
#
# # ادامه کد بعد از بخش مدل‌سازی
#
# import joblib
# import os
#
# # ایجاد پوشه برای ذخیره مدل‌ها
# if not os.path.exists('models'):
#     os.makedirs('models')
#
# # ذخیره همه مدل‌ها
# for name, model in models.items():
#     joblib.dump(model, f'models/{name.replace(" ", "_").lower()}_model.joblib')
#     print(f"مدل {name} ذخیره شد.")
#
# # ذخیره scaler
# joblib.dump(scaler, 'models/scaler.joblib')
# print("Scaler ذخیره شد.")
#
# # ذخیره لیست ویژگی‌ها (برای استفاده در آینده)
# feature_names = list(X.columns)
# joblib.dump(feature_names, 'models/feature_names.joblib')
# print("لیست ویژگی‌ها ذخیره شد.")
#
# # ایجاد یک دیکشنری برای نگهداری اطلاعات همه مدل‌ها
# model_info = {
#     'models': {name: model for name, model in models.items()},
#     'scaler': scaler,
#     'feature_names': feature_names,
#     'results': results
# }
#
# # ذخیره همه اطلاعات در یک فایل
# joblib.dump(model_info, 'models/all_models_info.joblib')
# print("همه اطلاعات مدل‌ها در یک فایل ذخیره شد.")
#
# print("\n" + "=" * 50)
# print("مدل‌های ذخیره شده:")
# print("=" * 50)
# for name in models.keys():
#     print(f"- {name.replace('_', ' ').title()}")
#
# # کد برای بارگذاری مدل‌ها در آینده
# print("\n" + "=" * 50)
# print("نحوه بارگذاری مدل‌ها در آینده:")
# print("=" * 50)
# print("""
# # بارگذاری یک مدل خاص
# model = joblib.load('models/gradient_boosting_model.joblib')
#
# # بارگذاری scaler
# scaler = joblib.load('models/scaler.joblib')
#
# # بارگذاری همه اطلاعات
# all_models = joblib.load('models/all_models_info.joblib')
# gradient_boosting_model = all_models['models']['Gradient Boosting']
# """)
#
#
# # ایجاد یک تابع پیش‌بینی ساده برای استفاده آینده
# def predict_asd(new_data, model_name='Gradient Boosting'):
#     """
#     تابع برای پیش‌بینی داده‌های جدید
#     """
#     # بارگذاری مدل و scaler
#     model = joblib.load(f'models/{model_name.replace(" ", "_").lower()}_model.joblib')
#     scaler = joblib.load('models/scaler.joblib')
#     feature_names = joblib.load('models/feature_names.joblib')
#
#     # پیش‌پردازش داده‌های جدید
#     new_data_scaled = scaler.transform(new_data)
#
#     # پیش‌بینی
#     predictions = model.predict(new_data_scaled)
#     prediction_proba = model.predict_proba(new_data_scaled) if hasattr(model, "predict_proba") else None
#
#     return predictions, prediction_proba
#
#
# # تست تابع پیش‌بینی با یک نمونه از داده‌های تست
# sample_data = X_test.iloc[:1]
# prediction, probability = predict_asd(sample_data)
# print(f"\nتست پیش‌بینی روی یک نمونه: {prediction[0]} (کلاس واقعی: {y_test.iloc[0]})")
# if probability is not None:
#     print(f"احتمالات: {probability[0]}")
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
#     f1_score
# import joblib
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # بارگذاری داده اصلی
# df = pd.read_csv('Autism-Adult-Data.csv')
#
# # پیش‌پردازش داده‌ها (مشابه قبل)
# df['age'].fillna(df['age'].median(), inplace=True)
# df['ethnicity'].fillna('Unknown', inplace=True)
# df['relation'].fillna('Unknown', inplace=True)
# df.drop('age_desc', axis=1, inplace=True)
#
# # تبدیل متغیرهای کیفی به عددی
# from sklearn.preprocessing import LabelEncoder
#
# binary_cols = ['gender', 'jundice', 'austim', 'used_app_before', 'Class/ASD']
# le = LabelEncoder()
# for col in binary_cols:
#     df[col] = le.fit_transform(df[col])
#
# # One-Hot Encoding
# categorical_cols = ['ethnicity', 'contry_of_res', 'relation']
# df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
#
# # تقسیم داده‌ها
# X = df.drop('Class/ASD', axis=1)
# y = df['Class/ASD']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # استانداردسازی
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# print("=" * 60)
# print("بررسی جامع مدل‌های ذخیره شده")
# print("=" * 60)
#
# # بارگذاری مدل‌های ذخیره شده
# models_to_load = [
#     'logistic_regression_model.joblib',
#     'k-nearest_neighbors_model.joblib',
#     'support_vector_machine_model.joblib',
#     'decision_tree_model.joblib',
#     'random_forest_model.joblib',
#     'gradient_boosting_model.joblib',
#     'xgboost_model.joblib'
# ]
#
# models = {}
# for model_file in models_to_load:
#     try:
#         model_name = model_file.replace('_model.joblib', '').replace('_', ' ').title()
#         models[model_name] = joblib.load(f'models/{model_file}')
#         print(f"✅ مدل {model_name} با موفقیت بارگذاری شد")
#     except:
#         print(f"❌ خطا در بارگذاری مدل {model_file}")
#
# print("\n" + "=" * 60)
# print("ارزیابی مدل‌ها روی داده تست")
# print("=" * 60)
#
# results = {}
# for name, model in models.items():
#     # پیش‌بینی روی داده تست
#     y_pred = model.predict(X_test_scaled)
#
#     # محاسبه معیارهای ارزیابی
#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred)
#     rec = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#
#     results[name] = {
#         'accuracy': acc,
#         'precision': prec,
#         'recall': rec,
#         'f1_score': f1
#     }
#
#     print(f"\n{name}:")
#     print(f"  دقت: {acc:.4f}")
#     print(f"  precision: {prec:.4f}")
#     print(f"  recall: {rec:.4f}")
#     print(f"  F1-score: {f1:.4f}")
#     print("  " + "-" * 30)
#
# # ایجاد DataFrame برای مقایسه نتایج
# results_df = pd.DataFrame.from_dict(results, orient='index')
# results_df = results_df.sort_values('accuracy', ascending=False)
#
# print("\n" + "=" * 60)
# print("مقایسه مدل‌ها بر اساس دقت")
# print("=" * 60)
# print(results_df[['accuracy', 'f1_score']].round(4))
#
# # بررسی overfitting با مقایسه عملکرد روی داده train و test
# print("\n" + "=" * 60)
# print("بررسی Overfitting (اختلاف عملکرد train/test)")
# print("=" * 60)
#
# overfitting_results = {}
# for name, model in models.items():
#     # پیش‌بینی روی داده train
#     y_train_pred = model.predict(X_train_scaled)
#     train_acc = accuracy_score(y_train, y_train_pred)
#
#     # پیش‌بینی روی داده test
#     y_test_pred = model.predict(X_test_scaled)
#     test_acc = accuracy_score(y_test, y_test_pred)
#
#     # اختلاف performance
#     acc_diff = train_acc - test_acc
#
#     overfitting_results[name] = {
#         'train_accuracy': train_acc,
#         'test_accuracy': test_acc,
#         'accuracy_difference': acc_diff
#     }
#
#     print(f"{name}:")
#     print(f"  دقت آموزش: {train_acc:.4f}")
#     print(f"  دقت تست: {test_acc:.4f}")
#     print(f"  اختلاف: {acc_diff:.4f} {'(ممکن است overfit باشد)' if acc_diff > 0.05 else ''}")
#     print("  " + "-" * 40)
#
# # اعتبارسنجی متقابل برای اطمینان بیشتر
# print("\n" + "=" * 60)
# print("اعتبارسنجی متقابل (Cross-Validation)")
# print("=" * 60)
#
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
# for name, model in models.items():
#     cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
#     print(f"{name}:")
#     print(f"  دقت CV: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
#     print(f"  مقادیر CV: {[f'{score:.4f}' for score in cv_scores]}")
#
# # بررسی اهمیت ویژگی‌ها برای مدل‌های درختی
# print("\n" + "=" * 60)
# print("بررسی اهمیت ویژگی‌ها در مدل‌های درختی")
# print("=" * 60)
#
# tree_based_models = {
#     'Decision Tree': models.get('Decision Tree'),
#     'Random Forest': models.get('Random Forest'),
#     'Gradient Boosting': models.get('Gradient Boosting'),
#     'Xgboost': models.get('Xgboost')
# }
#
# for name, model in tree_based_models.items():
#     if model and hasattr(model, 'feature_importances_'):
#         print(f"\n{name} - 10 ویژگی مهم:")
#         feature_importance = pd.DataFrame({
#             'feature': X.columns,
#             'importance': model.feature_importances_
#         })
#         feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
#
#         plt.figure(figsize=(10, 6))
#         sns.barplot(x='importance', y='feature', data=feature_importance)
#         plt.title(f'ویژگی‌های مهم - {name}')
#         plt.tight_layout()
#         plt.show()
#
#         print(feature_importance[['feature', 'importance']].to_string(index=False))
#
# # پیشنهاد نهایی
# print("\n" + "=" * 60)
# print("پیشنهاد نهایی بر اساس تحلیل‌های انجام شده")
# print("=" * 60)
#
# # پیدا کردن بهترین مدل بر اساس F1-score (معیار بهتری برای داده‌های نامتوازن)
# best_model_name = results_df.index[0]
# best_f1_score = results_df.iloc[0]['f1_score']
#
# print(f"بهترین مدل بر اساس دقت: {best_model_name} (دقت: {results_df.iloc[0]['accuracy']:.4f})")
# print(f"بهترین مدل بر اساس F1-score: {best_model_name} (F1-score: {best_f1_score:.4f})")
#
# # بررسی overfitting برای بهترین مدل
# best_model_overfitting = overfitting_results[best_model_name]
# acc_diff = best_model_overfitting['accuracy_difference']
#
# if acc_diff > 0.05:
#     print(f"⚠️  هشدار: مدل {best_model_name} ممکن است overfit باشد (اختلاف دقت: {acc_diff:.4f})")
#     print("پیشنهاد: استفاده از Regularization یا کاهش پیچیدگی مدل")
# else:
#     print(f"✅ مدل {best_model_name} عملکرد متعادلی روی داده train و test دارد")
#
# print("\n" + "=" * 60)
# print("نتیجه‌گیری نهایی")
# print("=" * 60)
#
# if best_f1_score > 0.95 and acc_diff <= 0.05:
#     print("🎉 مدل انتخاب شده بسیار خوب عمل می‌کند و برای استفاده در production مناسب است")
#     print(f"مدل پیشنهادی: {best_model_name}")
# else:
#     print("🔍 نیاز به تنظیم بیشتر مدل یا استفاده از تکنیک‌های مقابله با overfitting داریم")
#     print("پیشنهادات:")
#     print("- استفاده از regularization در مدل‌ها")
#     print("- کاهش تعداد ویژگی‌ها (feature selection)")
#     print("- افزایش داده‌های آموزشی")
#     print("- استفاده از تکنیک‌های مقابله با داده‌های نامتوازن")
#
# # ذخیره بهترین مدل برای استفاده نهایی
# best_model = models[best_model_name]
# joblib.dump(best_model, 'models/best_model.joblib')
# joblib.dump(scaler, 'models/scaler.joblib')
# print(f"\n✅ مدل {best_model_name} به عنوان بهترین مدل ذخیره شد")
#
# # بررسی رابطه بین result و Class/ASD
# print("رابطه بین result و Class/ASD:")
# print(pd.crosstab(df['result'], df['Class/ASD']))
#
# # بررسی همبستگی
# print(f"\nهمبستگی بین result و Class/ASD: {df['result'].corr(df['Class/ASD'])}")
#
# # حذف ویژگی result و آموزش مجدد مدل‌ها
# X_new = df.drop(['Class/ASD', 'result'], axis=1)
# y_new = df['Class/ASD']
#
# # تقسیم داده‌ها و آموزش مجدد
# X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
#     X_new, y_new, test_size=0.2, random_state=42, stratify=y_new
# )
#
# # استانداردسازی و آموزش مدل‌ها بدون ویژگی result

# بررسی جامع داده و حل مشکل احتمالی Data Leakage
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
#     f1_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# import joblib
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # بارگذاری داده اصلی
# df = pd.read_csv('Autism-Adult-Data.csv')
#
# print("=" * 60)
# print("بررسی اولیه داده و شناسایی ویژگی‌های مشکوک")
# print("=" * 60)
#
# # بررسی رابطه بین result و Class/ASD
# print("توزیع result بر اساس Class/ASD:")
# result_asd_crosstab = pd.crosstab(df['result'], df['Class/ASD'], margins=True)
# print(result_asd_crosstab)
#
# # بررسی همبستگی بین result و سایر ویژگی‌های عددی
# numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# if 'result' in numeric_cols and 'Class/ASD' in df.columns:
#     # تبدیل Class/ASD به عددی برای محاسبه همبستگی
#     df_corr = df.copy()
#     le = LabelEncoder()
#     df_corr['Class/ASD_num'] = le.fit_transform(df_corr['Class/ASD'])
#
#     # محاسبه همبستگی
#     correlation = df_corr['result'].corr(df_corr['Class/ASD_num'])
#     print(f"\nهمبستگی بین result و Class/ASD: {correlation:.4f}")
#
# # پیش‌پردازش داده‌ها
# print("\n" + "=" * 60)
# print("پیش‌پردازش داده‌ها")
# print("=" * 60)
#
# # مدیریت مقادیر گمشده
# df['age'].fillna(df['age'].median(), inplace=True)
# df['ethnicity'].fillna('Unknown', inplace=True)
# df['relation'].fillna('Unknown', inplace=True)
#
# # حذف ستون‌های غیرضروری
# df.drop('age_desc', axis=1, inplace=True)
#
# # تبدیل متغیرهای کیفی به عددی
# binary_cols = ['gender', 'jundice', 'austim', 'used_app_before', 'Class/ASD']
# le = LabelEncoder()
# for col in binary_cols:
#     if col in df.columns:
#         df[col] = le.fit_transform(df[col])
#
# # One-Hot Encoding برای متغیرهای دسته‌ای
# categorical_cols = ['ethnicity', 'contry_of_res', 'relation']
# df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
#
# print(f"ابعاد داده پس از پیش‌پردازش: {df.shape}")
#
# # دو سناریو مختلف برای مدل‌سازی
# print("\n" + "=" * 60)
# print("مقایسه دو سناریو: با و بدون ویژگی result")
# print("=" * 60)
#
# scenarios = {
#     'با ویژگی result': df.drop('Class/ASD', axis=1),
#     'بدون ویژگی result': df.drop(['Class/ASD', 'result'], axis=1)
# }
#
# results_comparison = {}
#
# for scenario_name, X in scenarios.items():
#     print(f"\n{'=' * 30} {scenario_name} {'=' * 30}")
#
#     y = df['Class/ASD']
#
#     # تقسیم داده‌ها
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
#
#     # استانداردسازی
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#
#     # تعریف مدل‌ها
#     models = {
#         "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
#         "K-Nearest Neighbors": KNeighborsClassifier(),
#         "Support Vector Machine": SVC(random_state=42, class_weight='balanced', probability=True),
#         "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5),
#         "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100,
#                                                 max_depth=5),
#         "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3),
#         "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', max_depth=3)
#     }
#
#     scenario_results = {}
#
#     for name, model in models.items():
#         # آموزش مدل
#         model.fit(X_train_scaled, y_train)
#
#         # پیش‌بینی
#         y_pred = model.predict(X_test_scaled)
#
#         # ارزیابی
#         acc = accuracy_score(y_test, y_pred)
#         prec = precision_score(y_test, y_pred)
#         rec = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#
#         scenario_results[name] = {
#             'accuracy': acc,
#             'precision': prec,
#             'recall': rec,
#             'f1_score': f1
#         }
#
#         # بررسی overfitting
#         y_train_pred = model.predict(X_train_scaled)
#         train_acc = accuracy_score(y_train, y_train_pred)
#         acc_diff = train_acc - acc
#
#         print(f"{name}:")
#         print(f"  دقت (تست): {acc:.4f}, دقت (آموزش): {train_acc:.4f}, اختلاف: {acc_diff:.4f}")
#         if acc_diff > 0.05:
#             print("  ⚠️  احتمال overfitting")
#
#     # ذخیره نتایج این سناریو
#     results_comparison[scenario_name] = scenario_results
#
# # مقایسه نتایج دو سناریو
# print("\n" + "=" * 60)
# print("مقایسه جامع عملکرد مدل‌ها در دو سناریو")
# print("=" * 60)
#
# for model_name in models.keys():
#     print(f"\n{model_name}:")
#     print("-" * len(model_name))
#
#     for scenario_name in scenarios.keys():
#         if model_name in results_comparison[scenario_name]:
#             results = results_comparison[scenario_name][model_name]
#             print(f"  {scenario_name}:")
#             print(f"    دقت: {results['accuracy']:.4f}, F1-score: {results['f1_score']:.4f}")
#
# # انتخاب بهترین سناریو و مدل
# print("\n" + "=" * 60)
# print("انتخاب بهترین مدل بر اساس معیار F1-score")
# print("=" * 60)
#
# best_scenario = None
# best_model_name = None
# best_f1 = 0
#
# for scenario_name, scenario_results in results_comparison.items():
#     for model_name, results in scenario_results.items():
#         if results['f1_score'] > best_f1:
#             best_f1 = results['f1_score']
#             best_model_name = model_name
#             best_scenario = scenario_name
#
# print(f"بهترین مدل: {best_model_name}")
# print(f"بهترین سناریو: {best_scenario}")
# print(f"best F1-score: {best_f1:.4f}")
#
# # آموزش و ذخیره بهترین مدل
# print("\n" + "=" * 60)
# print("آموزش و ذخیره بهترین مدل")
# print("=" * 60)
#
# # انتخاب داده‌های مناسب بر اساس بهترین سناریو
# if best_scenario == 'با ویژگی result':
#     X_final = df.drop('Class/ASD', axis=1)
# else:
#     X_final = df.drop(['Class/ASD', 'result'], axis=1)
#
# y_final = df['Class/ASD']
#
# # تقسیم داده‌ها
# X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
#     X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
# )
#
# # استانداردسازی
# scaler_final = StandardScaler()
# X_train_final_scaled = scaler_final.fit_transform(X_train_final)
# X_test_final_scaled = scaler_final.transform(X_test_final)
#
# # آموزش بهترین مدل
# if best_model_name == "Logistic Regression":
#     best_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
# elif best_model_name == "K-Nearest Neighbors":
#     best_model = KNeighborsClassifier()
# elif best_model_name == "Support Vector Machine":
#     best_model = SVC(random_state=42, class_weight='balanced', probability=True)
# elif best_model_name == "Decision Tree":
#     best_model = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5)
# elif best_model_name == "Random Forest":
#     best_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=5)
# elif best_model_name == "Gradient Boosting":
#     best_model = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3)
# elif best_model_name == "XGBoost":
#     best_model = XGBClassifier(random_state=42, eval_metric='logloss', max_depth=3)
#
# best_model.fit(X_train_final_scaled, y_train_final)
#
# # ارزیابی نهایی
# y_pred_final = best_model.predict(X_test_final_scaled)
# final_acc = accuracy_score(y_test_final, y_pred_final)
# final_f1 = f1_score(y_test_final, y_pred_final)
#
# print(f"دقت نهایی بهترین مدل: {final_acc:.4f}")
# print(f"F1-score نهایی: {final_f1:.4f}")
#
# # ذخیره مدل و scaler
# joblib.dump(best_model, 'best_model_final.joblib')
# joblib.dump(scaler_final, 'scaler_final.joblib')
#
# # ذخیره نام ویژگی‌ها
# feature_names = list(X_final.columns)
# joblib.dump(feature_names, 'feature_names_final.joblib')
#
# print("\nمدل، scaler و نام ویژگی‌ها با موفقیت ذخیره شدند.")
#
# # تحلیل اهمیت ویژگی‌ها (اگر مدل درختی است)
# if hasattr(best_model, 'feature_importances_'):
#     print("\n" + "=" * 60)
#     print("تحلیل اهمیت ویژگی‌ها در بهترین مدل")
#     print("=" * 60)
#
#     feature_importance = pd.DataFrame({
#         'feature': X_final.columns,
#         'importance': best_model.feature_importances_
#     })
#     feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
#
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='importance', y='feature', data=feature_importance)
#     plt.title(f'ویژگی‌های مهم - {best_model_name}')
#     plt.tight_layout()
#     plt.show()
#
#     print("ده ویژگی مهم:")
#     print(feature_importance[['feature', 'importance']].to_string(index=False))
#
# print("\n" + "=" * 60)
# print("نتیجه‌گیری نهایی")
# print("=" * 60)
#
# if best_scenario == 'با ویژگی result' and best_f1 > 0.95:
#     print("⚠️  هشدار: بهترین مدل از ویژگی result استفاده می‌کند که ممکن است باعث data leakage شود!")
#     print("پیشنهاد: در محیط production از سناریوی بدون ویژگی result استفاده کنید.")
# elif best_f1 > 0.85:
#     print("✅ مدل انتخاب شده عملکرد خوبی دارد و برای استفاده در production مناسب است.")
# else:
#     print("🔍 مدل نیاز به بهبود دارد. پیشنهاد می‌شود از تکنیک‌های زیر استفاده کنید:")
#     print("   - تنظیم هیپرپارامترها")
#     print("   - استفاده از تکنیک‌های مقابله با داده‌های نامتوازن")
#     print("   - افزایش داده‌های آموزشی")
#
# print(f"\nبهترین مدل نهایی: {best_model_name}")
# print(f"سناریوی انتخابی: {best_scenario}")


# # بررسی جامع داده و حل مشکل احتمالی Data Leakage
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
#     f1_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# import joblib
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # بارگذاری داده اصلی
df = pd.read_csv('Autism-Adult-Data.csv')
#
# print("=" * 60)
# print("بررسی اولیه داده و شناسایی ویژگی‌های مشکوک")
# print("=" * 60)
#
# # بررسی رابطه بین result و Class/ASD
# print("توزیع result بر اساس Class/ASD:")
# result_asd_crosstab = pd.crosstab(df['result'], df['Class/ASD'], margins=True)
# print(result_asd_crosstab)
#
# # بررسی همبستگی بین result و سایر ویژگی‌های عددی
# numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# if 'result' in numeric_cols and 'Class/ASD' in df.columns:
#     # تبدیل Class/ASD به عددی برای محاسبه همبستگی
#     df_corr = df.copy()
#     le = LabelEncoder()
#     df_corr['Class/ASD_num'] = le.fit_transform(df_corr['Class/ASD'])
#
#     # محاسبه همبستگی
#     correlation = df_corr['result'].corr(df_corr['Class/ASD_num'])
#     print(f"\nهمبستگی بین result و Class/ASD: {correlation:.4f}")
#
# # پیش‌پردازش داده‌ها
# print("\n" + "=" * 60)
# print("پیش‌پردازش داده‌ها")
# print("=" * 60)
#
# # مدیریت مقادیر گمشده
# df['age'].fillna(df['age'].median(), inplace=True)
# df['ethnicity'].fillna('Unknown', inplace=True)
# df['relation'].fillna('Unknown', inplace=True)
#
# # حذف ستون‌های غیرضروری
# df.drop('age_desc', axis=1, inplace=True)
#
# # تبدیل متغیرهای کیفی به عددی
# binary_cols = ['gender', 'jundice', 'austim', 'used_app_before', 'Class/ASD']
# le = LabelEncoder()
# for col in binary_cols:
#     if col in df.columns:
#         df[col] = le.fit_transform(df[col])
#
# # One-Hot Encoding برای متغیرهای دسته‌ای
# categorical_cols = ['ethnicity', 'contry_of_res', 'relation']
# df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
#
# print(f"ابعاد داده پس از پیش‌پردازش: {df.shape}")
#
# # دو سناریو مختلف برای مدل‌سازی
# print("\n" + "=" * 60)
# print("مقایسه دو سناریو: با و بدون ویژگی result")
# print("=" * 60)
#
# scenarios = {
#     'با ویژگی result': df.drop('Class/ASD', axis=1),
#     'بدون ویژگی result': df.drop(['Class/ASD', 'result'], axis=1)
# }
#
# results_comparison = {}
#
# for scenario_name, X in scenarios.items():
#     print(f"\n{'=' * 30} {scenario_name} {'=' * 30}")
#
#     y = df['Class/ASD']
#
#     # تقسیم داده‌ها
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
#
#     # استانداردسازی
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#
#     # تعریف مدل‌ها
#     models = {
#         "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
#         "K-Nearest Neighbors": KNeighborsClassifier(),
#         "Support Vector Machine": SVC(random_state=42, class_weight='balanced', probability=True),
#         "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5),
#         "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100,
#                                                 max_depth=5),
#         "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3),
#         "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', max_depth=3)
#     }
#
#     scenario_results = {}
#
#     for name, model in models.items():
#         # آموزش مدل
#         model.fit(X_train_scaled, y_train)
#
#         # پیش‌بینی
#         y_pred = model.predict(X_test_scaled)
#
#         # ارزیابی
#         acc = accuracy_score(y_test, y_pred)
#         prec = precision_score(y_test, y_pred)
#         rec = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#
#         scenario_results[name] = {
#             'accuracy': acc,
#             'precision': prec,
#             'recall': rec,
#             'f1_score': f1
#         }
#
#         # بررسی overfitting
#         y_train_pred = model.predict(X_train_scaled)
#         train_acc = accuracy_score(y_train, y_train_pred)
#         acc_diff = train_acc - acc
#
#         print(f"{name}:")
#         print(f"  دقت (تست): {acc:.4f}, دقت (آموزش): {train_acc:.4f}, اختلاف: {acc_diff:.4f}")
#         if acc_diff > 0.05:
#             print("  ⚠️  احتمال overfitting")
#
#     # ذخیره نتایج این سناریو
#     results_comparison[scenario_name] = scenario_results
#
# # مقایسه نتایج دو سناریو
# print("\n" + "=" * 60)
# print("مقایسه جامع عملکرد مدل‌ها در دو سناریو")
# print("=" * 60)
#
# for model_name in models.keys():
#     print(f"\n{model_name}:")
#     print("-" * len(model_name))
#
#     for scenario_name in scenarios.keys():
#         if model_name in results_comparison[scenario_name]:
#             results = results_comparison[scenario_name][model_name]
#             print(f"  {scenario_name}:")
#             print(f"    دقت: {results['accuracy']:.4f}, F1-score: {results['f1_score']:.4f}")
#
# # انتخاب بهترین سناریو و مدل
# print("\n" + "=" * 60)
# print("انتخاب بهترین مدل بر اساس معیار F1-score")
# print("=" * 60)
#
# best_scenario = None
# best_model_name = None
# best_f1 = 0
#
# for scenario_name, scenario_results in results_comparison.items():
#     for model_name, results in scenario_results.items():
#         if results['f1_score'] > best_f1:
#             best_f1 = results['f1_score']
#             best_model_name = model_name
#             best_scenario = scenario_name
#
# print(f"بهترین مدل: {best_model_name}")
# print(f"بهترین سناریو: {best_scenario}")
# print(f"best F1-score: {best_f1:.4f}")
#
# # آموزش و ذخیره بهترین مدل
# print("\n" + "=" * 60)
# print("آموزش و ذخیره بهترین مدل")
# print("=" * 60)
#
# # انتخاب داده‌های مناسب بر اساس بهترین سناریو
# if best_scenario == 'با ویژگی result':
#     X_final = df.drop('Class/ASD', axis=1)
# else:
#     X_final = df.drop(['Class/ASD', 'result'], axis=1)
#
# y_final = df['Class/ASD']
#
# # تقسیم داده‌ها
# X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
#     X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
# )
#
# # استانداردسازی
# scaler_final = StandardScaler()
# X_train_final_scaled = scaler_final.fit_transform(X_train_final)
# X_test_final_scaled = scaler_final.transform(X_test_final)
#
# # آموزش بهترین مدل
# if best_model_name == "Logistic Regression":
#     best_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
# elif best_model_name == "K-Nearest Neighbors":
#     best_model = KNeighborsClassifier()
# elif best_model_name == "Support Vector Machine":
#     best_model = SVC(random_state=42, class_weight='balanced', probability=True)
# elif best_model_name == "Decision Tree":
#     best_model = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5)
# elif best_model_name == "Random Forest":
#     best_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=5)
# elif best_model_name == "Gradient Boosting":
#     best_model = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3)
# elif best_model_name == "XGBoost":
#     best_model = XGBClassifier(random_state=42, eval_metric='logloss', max_depth=3)
#
# best_model.fit(X_train_final_scaled, y_train_final)
#
# # ارزیابی نهایی
# y_pred_final = best_model.predict(X_test_final_scaled)
# final_acc = accuracy_score(y_test_final, y_pred_final)
# final_f1 = f1_score(y_test_final, y_pred_final)
#
# print(f"دقت نهایی بهترین مدل: {final_acc:.4f}")
# print(f"F1-score نهایی: {final_f1:.4f}")
#
# # ذخیره مدل و scaler
# joblib.dump(best_model, 'best_model_final.joblib')
# joblib.dump(scaler_final, 'scaler_final.joblib')
#
# # ذخیره نام ویژگی‌ها
# feature_names = list(X_final.columns)
# joblib.dump(feature_names, 'feature_names_final.joblib')
#
# print("\nمدل، scaler و نام ویژگی‌ها با موفقیت ذخیره شدند.")
#
# # تحلیل اهمیت ویژگی‌ها (اگر مدل درختی است)
# if hasattr(best_model, 'feature_importances_'):
#     print("\n" + "=" * 60)
#     print("تحلیل اهمیت ویژگی‌ها در بهترین مدل")
#     print("=" * 60)
#
#     feature_importance = pd.DataFrame({
#         'feature': X_final.columns,
#         'importance': best_model.feature_importances_
#     })
#     feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
#
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='importance', y='feature', data=feature_importance)
#     plt.title(f'ویژگی‌های مهم - {best_model_name}')
#     plt.tight_layout()
#     plt.show()
#
#     print("ده ویژگی مهم:")
#     print(feature_importance[['feature', 'importance']].to_string(index=False))
#
# print("\n" + "=" * 60)
# print("نتیجه‌گیری نهایی")
# print("=" * 60)
#
# if best_scenario == 'با ویژگی result' and best_f1 > 0.95:
#     print("⚠️  هشدار: بهترین مدل از ویژگی result استفاده می‌کند که ممکن است باعث data leakage شود!")
#     print("پیشنهاد: در محیط production از سناریوی بدون ویژگی result استفاده کنید.")
# elif best_f1 > 0.85:
#     print("✅ مدل انتخاب شده عملکرد خوبی دارد و برای استفاده در production مناسب است.")
# else:
#     print("🔍 مدل نیاز به بهبود دارد. پیشنهاد می‌شود از تکنیک‌های زیر استفاده کنید:")
#     print("   - تنظیم هیپرپارامترها")
#     print("   - استفاده از تکنیک‌های مقابله با داده‌های نامتوازن")
#     print("   - افزایش داده‌های آموزشی")
#
# print(f"\nبهترین مدل نهایی: {best_model_name}")
# print(f"سناریوی انتخابی: {best_scenario}")

# کد اصلاح شده برای استفاده از مدل واقعی
# کد کامل برای آموزش و ذخیره مدل واقعی بدون Data Leakage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# تنظیمات نمودار
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
sns.set_palette("husl")

# بارگذاری داده اصلی برای تست
df = pd.read_csv('Autism-Adult-Data.csv')

# پیش‌پردازش داده‌ها (مشابه قبل)
df['age'].fillna(df['age'].median(), inplace=True)
df['ethnicity'].fillna('Unknown', inplace=True)
df['relation'].fillna('Unknown', inplace=True)
df.drop('age_desc', axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

binary_cols = ['gender', 'jundice', 'austim', 'used_app_before', 'Class/ASD']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

categorical_cols = ['ethnicity', 'contry_of_res', 'relation']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# آماده سازی داده برای تست
X_with_result = df.drop('Class/ASD', axis=1)
X_without_result = df.drop(['Class/ASD', 'result'], axis=1)
y = df['Class/ASD']

# تقسیم داده
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_without_result, y, test_size=0.2, random_state=42, stratify=y)

# استانداردسازی
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

print("=" * 60)
print("مقایسه جامع همه مدل‌های ذخیره شده")
print("=" * 60)

# لیست همه مدل‌های ذخیره شده
model_files = {
    'final_model': 'best_model_final.joblib',
    'real_model': 'best_model_real.joblib',
    'logistic_regression': 'models/logistic_regression_model.joblib',
    'random_forest': 'models/random_forest_model.joblib',
    'gradient_boosting': 'models/gradient_boosting_model.joblib',
    'xgboost': 'models/xgboost_model.joblib'
}

results = []

# تست هر مدل
for model_name, file_path in model_files.items():
    try:
        # بارگذاری مدل
        model = joblib.load(file_path)

        # پیش‌بینی
        if 'result' in model_name:
            # برای مدل‌هایی که با result آموزش دیده‌اند، نیاز به داده متفاوت داریم
            X_test_special = X_with_result.iloc[X_test.index]
            X_test_special_scaled = scaler.transform(X_test_special)
            y_pred = model.predict(X_test_special_scaled)
        else:
            # برای مدل‌های بدون result
            y_pred = model.predict(X_test_scaled)

        # محاسبه معیارها
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            'model': model_name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })

        print(f"{model_name}: دقت = {acc:.4f}, F1-score = {f1:.4f}")

    except FileNotFoundError:
        print(f"فایل {file_path} یافت نشد")
    except Exception as e:
        print(f"خطا در بارگذاری {model_name}: {str(e)}")

# ایجاد DataFrame از نتایج
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1_score', ascending=False)

print("\n" + "=" * 60)
print("رده‌بندی مدل‌ها بر اساس F1-Score")
print("=" * 60)
print(results_df[['model', 'accuracy', 'f1_score']].round(4))

# رسم نمودار مقایسه
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('مقایسه جامع عملکرد مدل‌های مختلف', fontsize=16, fontweight='bold')

# نمودار دقت
axes[0, 0].barh(results_df['model'], results_df['accuracy'], color='skyblue')
axes[0, 0].set_title('دقت (Accuracy) مدل‌ها')
axes[0, 0].set_xlabel('دقت')
axes[0, 0].set_xlim(0, 1)

# نمودار F1-Score
axes[0, 1].barh(results_df['model'], results_df['f1_score'], color='lightgreen')
axes[0, 1].set_title('F1-Score مدل‌ها')
axes[0, 1].set_xlabel('F1-Score')
axes[0, 1].set_xlim(0, 1)

# نمودار Precision
axes[1, 0].barh(results_df['model'], results_df['precision'], color='lightcoral')
axes[1, 0].set_title('Precision مدل‌ها')
axes[1, 0].set_xlabel('Precision')
axes[1, 0].set_xlim(0, 1)

# نمودار Recall
axes[1, 1].barh(results_df['model'], results_df['recall'], color='gold')
axes[1, 1].set_title('Recall مدل‌ها')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_xlim(0, 1)

plt.tight_layout()
plt.show()

# نمودار Radar برای مقایسه همه معیارها
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)

# محاسبه زوایا
categories = ['دقت', 'Precision', 'Recall', 'F1-Score']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# رسم برای هر مدل
colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))

for idx, (_, row) in enumerate(results_df.iterrows()):
    values = [
        row['accuracy'],
        row['precision'],
        row['recall'],
        row['f1_score']
    ]
    values += values[:1]

    ax.plot(angles, values, color=colors[idx], linewidth=2, label=row['model'])
    ax.fill(angles, values, color=colors[idx], alpha=0.1)

# تنظیمات نمودار
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_yticklabels([])
ax.set_title('مقایسه جامع مدل‌ها با نمودار Radar', size=15, y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.show()

# پیدا کردن بهترین مدل
best_model_row = results_df.iloc[0]
best_model_name = best_model_row['model']
best_model_file = model_files[best_model_name]

print("\n" + "=" * 60)
print("نتیجه‌گیری نهایی و انتخاب بهترین مدل")
print("=" * 60)
print(f"بهترین مدل: {best_model_name}")
print(f"دقت: {best_model_row['accuracy']:.4f}")
print(f"F1-Score: {best_model_row['f1_score']:.4f}")
print(f"Precision: {best_model_row['precision']:.4f}")
print(f"Recall: {best_model_row['recall']:.4f}")

# هشدار درباره مدل‌های با data leakage
models_with_result = [name for name in results_df['model'] if 'result' in name]
if models_with_result:
    print(f"\n⚠️  هشدار: مدل‌های زیر ممکن است دچار data leakage شده باشند:")
    for model in models_with_result:
        print(f"   - {model}")

print("\n✅ پیشنهاد: از best_model_real.joblib استفاده کنید")
print("   زیرا بدون data leakage آموزش دیده و عملکرد واقعی دارد")

# نمایش اهمیت ویژگی‌ها برای بهترین مدل
try:
    best_model = joblib.load(best_model_file)
    if hasattr(best_model, 'feature_importances_'):
        print("\n" + "=" * 60)
        print("ده ویژگی مهم در بهترین مدل")
        print("=" * 60)

        feature_importance = pd.DataFrame({
            'feature': X_without_result.columns,
            'importance': best_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('ده ویژگی مهم در بهترین مدل')
        plt.tight_layout()
        plt.show()

        print(feature_importance[['feature', 'importance']].to_string(index=False))

except Exception as e:
    print(f"خطا در نمایش اهمیت ویژگی‌ها: {str(e)}")