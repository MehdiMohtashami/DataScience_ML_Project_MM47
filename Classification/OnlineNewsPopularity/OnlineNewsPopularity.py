# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# # بارگذاری دیتاست
# df = pd.read_csv('OnlineNewsPopularity.csv')
#
# # رسم هیستوگرام برای توزیع متغیر هدف (shares)
# # plt.figure(figsize=(10, 6))
# # sns.histplot(df['shares'], bins=50, kde=True)
# # plt.title('Distribution of Shares')
# # plt.xlabel('Number of Shares')
# # plt.ylabel('Frequency')
# # plt.xlim(0, 20000)  # محدود کردن محور x برای بهتر دیدن توزیع
# # plt.show()
# #
# # # رسم هیستوگرام لگاریتمی برای بررسی توزیع نرمال‌تر
# # plt.figure(figsize=(10, 6))
# # sns.histplot(np.log1p(df['shares']), bins=50, kde=True)
# # plt.title('Log-Transformed Distribution of Shares')
# # plt.xlabel('Log(Number of Shares)')
# # plt.ylabel('Frequency')
# # plt.show()
#
# # ماتریس همبستگی برای بررسی ارتباط ویژگی‌ها با shares
# # plt.figure(figsize=(12, 8))
# # correlation_matrix = df.drop(['url'], axis=1).corr()  # حذف ستون url
# # sns.heatmap(correlation_matrix[['shares']].sort_values(by='shares', ascending=False),
# #             annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# # plt.title('Correlation of Features with Shares')
# # plt.show()
#
# # بررسی توزیع برخی ویژگی‌های مهم
# # features_to_plot = ['n_tokens_content', 'num_hrefs', 'num_imgs', 'num_videos', 'average_token_length']
# # for feature in features_to_plot:
# #     plt.figure(figsize=(10, 6))
# #     sns.scatterplot(x=df[feature], y=df['shares'])
# #     plt.title(f'{feature} vs Shares')
# #     plt.xlabel(feature)
# #     plt.ylabel('Shares')
# #     plt.ylim(0, 20000)  # محدود کردن محور y
# #     plt.show()
#
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest, f_regression
#
# # حذف ستون‌های غیرپیش‌بینی‌کننده
# X = df.drop(['url', 'shares'], axis=1)
# y = df['shares']
#
# # تبدیل لگاریتمی متغیر هدف (اختیاری)
# y_log = np.log1p(y)
#
# # استانداردسازی ویژگی‌ها
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # انتخاب ویژگی‌های مهم
# selector = SelectKBest(score_func=f_regression, k=20)  # انتخاب 20 ویژگی برتر
# X_selected = selector.fit_transform(X_scaled, y)
# selected_features = X.columns[selector.get_support()].tolist()
# print("Selected Features:", selected_features)
#
# # تقسیم داده‌ها به آموزش و تست
# X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
#
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.neural_network import MLPRegressor
# import numpy as np
#
# # تعریف مدل‌ها
# models = {
#     'Linear Regression': LinearRegression(),
#     'Ridge Regression': Ridge(alpha=1.0),
#     'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#     'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
#     'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
# }
#
# # آموزش و ارزیابی مدل‌ها
# results = {}
# for name, model in models.items():
#     # آموزش مدل
#     model.fit(X_train, y_train)
#
#     # پیش‌بینی
#     y_pred = model.predict(X_test)
#
#     # محاسبه معیارها
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, y_pred)
#
#     results[name] = {'RMSE': rmse, 'R2': r2}
#
#     print(f"\n{name}:")
#     print(f"RMSE: {rmse:.2f}")
#     print(f"R2 Score: {r2:.2f}")
#
# # رسم مقایسه مدل‌ها
# plt.figure(figsize=(10, 6))
# rmse_values = [results[model]['RMSE'] for model in results]
# r2_values = [results[model]['R2'] for model in results]
# models_names = list(results.keys())
#
# # نمودار RMSE
# # plt.subplot(1, 2, 1)
# # sns.barplot(x=rmse_values, y=models_names)
# # plt.title('RMSE of Regression Models')
# # plt.xlabel('RMSE')
# #
# # # نمودار R2
# # plt.subplot(1, 2, 2)
# # sns.barplot(x=r2_values, y=models_names)
# # plt.title('R2 Score of Regression Models')
# # plt.xlabel('R2 Score')
# #
# # plt.tight_layout()
# # plt.show()
#
# # تبدیل shares به دسته‌های باینری
# y_class = (y > y.median()).astype(int)  # 1 برای محبوب، 0 برای غیرمحبوب
#
# # تقسیم داده‌ها
# X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_selected, y_class, test_size=0.2,
#                                                                             random_state=42)
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report
#
# # تعریف مدل‌های کلاسیفیکیشن
# class_models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
#     'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
#     'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, random_state=42)
# }
#
# # آموزش و ارزیابی مدل‌ها
# class_results = {}
# for name, model in class_models.items():
#     model.fit(X_train_class, y_train_class)
#     y_pred_class = model.predict(X_test_class)
#
#     accuracy = accuracy_score(y_test_class, y_pred_class)
#     class_results[name] = {'Accuracy': accuracy}
#
#     print(f"\n{name}:")
#     print(f"Accuracy: {accuracy:.2f}")
#     print(classification_report(y_test_class, y_pred_class))
#
# # رسم مقایسه مدل‌ها
# # plt.figure(figsize=(10, 6))
# # accuracy_values = [class_results[model]['Accuracy'] for model in class_results]
# # sns.barplot(x=accuracy_values, y=list(class_results.keys()))
# # plt.title('Accuracy of Classification Models')
# # plt.xlabel('Accuracy')
# # plt.show()
#
# import numpy as np
# from sklearn.model_selection import train_test_split
#
# # مدیریت outliers با روش IQR (Interquartile Range)
# Q1 = df['shares'].quantile(0.25)
# Q3 = df['shares'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
#
# # Clip کردن outliers (به جای حذف، برای حفظ داده‌ها)
# df['shares_clipped'] = df['shares'].clip(lower=lower_bound, upper=upper_bound)
#
# # تبدیل لگاریتمی برای رگرسیون
# df['shares_log'] = np.log1p(df['shares_clipped'])  # log1p برای جلوگیری از log(0)
#
# # حالا از shares_log برای رگرسیون و shares_clipped برای کلاسیفیکیشن استفاده کن
# X = df[selected_features]  # از ویژگی‌های انتخاب‌شده قبلی استفاده کن
# y_reg = df['shares_log']   # برای رگرسیون
# y_class = (df['shares_clipped'] > df['shares_clipped'].median()).astype(int)  # برای کلاسیفیکیشن
#
# # استانداردسازی (قبلی رو نگه دار)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # تقسیم داده‌ها
# X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
# X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)
#
# # حالا مدل‌ها رو دوباره اجرا کن (کد مدل‌ها رو از قبل کپی کن، اما y رو با y_reg یا y_class جایگزین کن)
# # برای رگرسیون، بعد از پیش‌بینی، از expm1(y_pred) برای برگرداندن به مقیاس اصلی استفاده کن تا RMSE واقعی محاسبه بشه
#
# y_pred_original = np.expm1(y_pred)  # برگرداندن از log
# y_test_original = np.expm1(y_test_reg)
# rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
#
# # مثلاً بعد از fit کردن Gradient Boosting Regressor
# gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# gb_model.fit(X_train_reg, y_train_reg)
#
# # استخراج Feature Importance
# importances = gb_model.feature_importances_
# feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
#
# print(feature_importance_df)
#
# # رسم نمودار
# plt.figure(figsize=(12, 8))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
# plt.title('Feature Importance from Gradient Boosting')
# plt.show()
#
# # اگر بخوای ویژگی‌های کم‌اهمیت رو حذف کنی (مثلاً کمتر از 0.01)
# important_features = feature_importance_df[feature_importance_df['Importance'] > 0.01]['Feature'].tolist()
# X_important = df[important_features]
# # حالا X_important رو استاندارد کن و مدل‌ها رو دوباره اجرا کن

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from joblib import dump

# بارگذاری دیتاست
df = pd.read_csv('OnlineNewsPopularity.csv')

# فیلتر داده‌ها (shares بین 100 تا 5000)
df = df[(df['shares'] >= 100) & (df['shares'] <= 5000)]

# مهندسی ویژگی‌های جدید
df['hrefs_per_token'] = df['num_hrefs'] / (df['n_tokens_content'] + 1)
df['imgs_per_token'] = df['num_imgs'] / (df['n_tokens_content'] + 1)
df['videos_per_token'] = df['num_videos'] / (df['n_tokens_content'] + 1)
df['keyword_avg'] = (df['kw_min_avg'] + df['kw_max_avg'] + df['kw_avg_avg']) / 3
df['multimedia_score'] = df['num_imgs'] + 2 * df['num_videos']
df['keyword_strength'] = 0.7 * df['kw_avg_avg'] + 0.3 * df['kw_max_avg']
df['content_length_score'] = df['n_tokens_content'] * df['average_token_length']
df['polarity_score'] = 0.5 * df['global_subjectivity'] + 0.5 * df['avg_negative_polarity']
df['engagement_score'] = 0.4 * df['num_hrefs'] + 0.3 * df['num_imgs'] + 0.3 * df['num_videos']
df['keyword_impact'] = 0.5 * df['kw_avg_avg'] + 0.3 * df['kw_max_avg'] + 0.2 * df['kw_min_avg']
df['social_impact'] = (df['self_reference_min_shares'] + df['self_reference_max_shares']) / 2
df['content_impact'] = df['n_tokens_content'] * df['num_keywords']
df['sentiment_impact'] = df['title_subjectivity'] * df['abs_title_sentiment_polarity']
df['interaction_score'] = df['num_hrefs'] * df['num_keywords']
df['text_complexity'] = df['n_tokens_content'] * df['n_unique_tokens']
df['keyword_content_ratio'] = df['num_keywords'] / (df['n_tokens_content'] + 1)
df['is_weekend'] = df[['is_weekend']].astype(int)

# ویژگی‌های انتخاب‌شده
selected_features = [
    'num_hrefs', 'num_imgs', 'num_videos', 'average_token_length', 'data_channel_is_world',
    'kw_max_min', 'kw_avg_min', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg',
    'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess',
    'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity', 'avg_negative_polarity',
    'title_subjectivity', 'abs_title_sentiment_polarity',
    'hrefs_per_token', 'imgs_per_token', 'videos_per_token', 'keyword_avg', 'is_weekend',
    'multimedia_score', 'keyword_strength', 'content_length_score', 'polarity_score',
    'engagement_score', 'keyword_impact', 'social_impact', 'content_impact', 'sentiment_impact',
    'interaction_score', 'text_complexity', 'keyword_content_ratio'
]

# مدیریت outliers با روش IQR (Clip کردن)
Q1 = df['shares'].quantile(0.25)
Q3 = df['shares'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['shares_clipped'] = df['shares'].clip(lower=lower_bound, upper=upper_bound)

# کلاسیفیکیشن با آستانه 1500
y_class = (df['shares_clipped'] > 2000).astype(int)

# آماده‌سازی داده‌ها
X = df[selected_features]
X = X.fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسیم داده‌ها
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)

# =====================================
# مدل‌های کلاسیفیکیشن
# =====================================
models_class = {
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 1.1}),
    'XGBoost Classifier': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', scale_pos_weight=1.1),
    'LightGBM Classifier': LGBMClassifier(n_estimators=100, random_state=42, scale_pos_weight=1.1, force_col_wise=True)
}

results_class = {}
for name, model in models_class.items():
    model.fit(X_train_class, y_train_class)
    y_pred_proba = model.predict_proba(X_test_class)[:, 1]
    for threshold in [0.3, 0.4, 0.5]:
        y_pred_class = (y_pred_proba > threshold).astype(int)
        accuracy = accuracy_score(y_test_class, y_pred_class)
        print(f"\n{name} (Classification - Pre-Tuning, Threshold={threshold}):")
        print(f"Accuracy: {accuracy:.2f}")
        print(classification_report(y_test_class, y_pred_class))

# =====================================
# Ensemble Model
# =====================================
ensemble_model = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 1.1})),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', scale_pos_weight=1.1)),
    ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, scale_pos_weight=1.1, force_col_wise=True))
], voting='soft', weights=[0.5, 0.25, 0.25])

ensemble_model.fit(X_train_class, y_train_class)
y_pred_proba = ensemble_model.predict_proba(X_test_class)[:, 1]
for threshold in [0.3, 0.4, 0.5]:
    y_pred_class = (y_pred_proba > threshold).astype(int)
    accuracy = accuracy_score(y_test_class, y_pred_class)
    print(f"\nEnsemble Model (Threshold={threshold}):")
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test_class, y_pred_class))

# =====================================
# Feature Importance (با LightGBM Classifier)
# =====================================
lgbm_class = LGBMClassifier(n_estimators=100, random_state=42, scale_pos_weight=1.1, force_col_wise=True)
lgbm_class.fit(X_train_class, y_train_class)
importances = lgbm_class.feature_importances_ / lgbm_class.feature_importances_.sum()
feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance (LightGBM Classifier):")
print(feature_importance_df)

# رسم نمودار
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance from LightGBM Classifier')
plt.show()

# انتخاب ویژگی‌های مهم
important_features = feature_importance_df[feature_importance_df['Importance'] > 0.01]['Feature'].tolist()
print("\nImportant Features:", important_features)

# بروزرسانی داده‌ها با ویژگی‌های مهم
X_important_scaled = scaler.fit_transform(df[important_features])
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_important_scaled, y_class, test_size=0.2, random_state=42)

# =====================================
# تنظیم هایپرپارامترها برای LightGBM Classifier
# =====================================
param_grid_class = {
    'n_estimators': [100, 150],
    'learning_rate': [0.1],
    'max_depth': [7, 10],
    'num_leaves': [31, 50],
    'min_child_samples': [20],
    'subsample': [0.8]
}
lgbm_class = LGBMClassifier(random_state=42, scale_pos_weight=1.1, force_col_wise=True)
grid_search_class = GridSearchCV(estimator=lgbm_class, param_grid=param_grid_class, cv=3, scoring='f1_weighted', n_jobs=1)
grid_search_class.fit(X_train_class, y_train_class)

print("\nBest Parameters for LightGBM Classification:", grid_search_class.best_params_)
best_lgbm_class = grid_search_class.best_estimator_

y_pred_proba = best_lgbm_class.predict_proba(X_test_class)[:, 1]
for threshold in [0.3, 0.4, 0.5]:
    y_pred_class = (y_pred_proba > threshold).astype(int)
    accuracy = accuracy_score(y_test_class, y_pred_class)
    print(f"\nBest LightGBM (Tuned, Threshold={threshold}):")
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test_class, y_pred_class))

# =====================================
# ذخیره مدل
# =====================================
dump(best_lgbm_class, 'best_lgbm_classification_model.joblib')
dump(ensemble_model, 'ensemble_model.joblib')
dump(scaler, 'scaler.joblib')
dump(important_features, 'important_features.joblib')

print("\nModels saved successfully!")