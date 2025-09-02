# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from joblib import dump
#
# # Load dataset
# df = pd.read_csv('OnlineNewsPopularity.csv')
#
# # Filter data (shares between 100 and 5000)
# df = df[(df['shares'] >= 100) & (df['shares'] <= 5000)]
#
# # Feature engineering
# df['hrefs_per_token'] = df['num_hrefs'] / (df['n_tokens_content'] + 1)
# df['imgs_per_token'] = df['num_imgs'] / (df['n_tokens_content'] + 1)
# df['videos_per_token'] = df['num_videos'] / (df['n_tokens_content'] + 1)
# df['keyword_avg'] = (df['kw_min_avg'] + df['kw_max_avg'] + df['kw_avg_avg']) / 3
# df['multimedia_score'] = df['num_imgs'] + 2 * df['num_videos']
# df['keyword_strength'] = 0.7 * df['kw_avg_avg'] + 0.3 * df['kw_max_avg']
# df['content_length_score'] = df['n_tokens_content'] * df['average_token_length']
# df['polarity_score'] = 0.5 * df['global_subjectivity'] + 0.5 * df['avg_negative_polarity']
# df['engagement_score'] = 0.4 * df['num_hrefs'] + 0.3 * df['num_imgs'] + 0.3 * df['num_videos']
# df['keyword_impact'] = 0.5 * df['kw_avg_avg'] + 0.3 * df['kw_max_avg'] + 0.2 * df['kw_min_avg']
# df['social_impact'] = (df['self_reference_min_shares'] + df['self_reference_max_shares']) / 2
# df['content_impact'] = df['n_tokens_content'] * df['num_keywords']
# df['sentiment_impact'] = df['title_subjectivity'] * df['abs_title_sentiment_polarity']
# df['interaction_score'] = df['num_hrefs'] * df['num_keywords']
# df['text_complexity'] = df['n_tokens_content'] * df['n_unique_tokens']
# df['keyword_content_ratio'] = df['num_keywords'] / (df['n_tokens_content'] + 1)
# df['is_weekend'] = df[['is_weekend']].astype(int)
#
# # Selected features
# selected_features = [
#     'num_hrefs', 'num_imgs', 'num_videos', 'average_token_length', 'data_channel_is_world',
#     'kw_max_min', 'kw_avg_min', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg',
#     'self_reference_min_shares', 'self_reference_max_shares', 'self_reference_avg_sharess',
#     'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity', 'avg_negative_polarity',
#     'title_subjectivity', 'abs_title_sentiment_polarity',
#     'hrefs_per_token', 'imgs_per_token', 'videos_per_token', 'keyword_avg', 'is_weekend',
#     'multimedia_score', 'keyword_strength', 'content_length_score', 'polarity_score',
#     'engagement_score', 'keyword_impact', 'social_impact', 'content_impact', 'sentiment_impact',
#     'interaction_score', 'text_complexity', 'keyword_content_ratio'
# ]
#
# # Handle outliers with IQR method (Clipping)
# Q1 = df['shares'].quantile(0.25)
# Q3 = df['shares'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
# df['shares_clipped'] = df['shares'].clip(lower=lower_bound, upper=upper_bound)
#
# # Classification with threshold 2000
# y_class = (df['shares_clipped'] > 2000).astype(int)
#
# # Prepare data
# X = df[selected_features]
# X = X.fillna(0)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Split data
# X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)
#
# # =====================================
# # Classification models
# # =====================================
# models_class = {
#     'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 1.1}),
#     'XGBoost Classifier': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', scale_pos_weight=1.1),
#     'LightGBM Classifier': LGBMClassifier(n_estimators=100, random_state=42, scale_pos_weight=1.1, force_col_wise=True)
# }
#
# results_class = {}
# for name, model in models_class.items():
#     model.fit(X_train_class, y_train_class)
#     y_pred_proba = model.predict_proba(X_test_class)[:, 1]
#     for threshold in [0.3, 0.4, 0.5]:
#         y_pred_class = (y_pred_proba > threshold).astype(int)
#         accuracy = accuracy_score(y_test_class, y_pred_class)
#         print(f"\n{name} (Classification - Pre-Tuning, Threshold={threshold}):")
#         print(f"Accuracy: {accuracy:.2f}")
#         print(classification_report(y_test_class, y_pred_class))
#
# # =====================================
# # Ensemble Model
# # =====================================
# ensemble_model = VotingClassifier(estimators=[
#     ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 1.1})),
#     ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', scale_pos_weight=1.1)),
#     ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, scale_pos_weight=1.1, force_col_wise=True))
# ], voting='soft', weights=[0.5, 0.25, 0.25])
#
# ensemble_model.fit(X_train_class, y_train_class)
# y_pred_proba = ensemble_model.predict_proba(X_test_class)[:, 1]
# for threshold in [0.3, 0.4, 0.5]:
#     y_pred_class = (y_pred_proba > threshold).astype(int)
#     accuracy = accuracy_score(y_test_class, y_pred_class)
#     print(f"\nEnsemble Model (Threshold={threshold}):")
#     print(f"Accuracy: {accuracy:.2f}")
#     print(classification_report(y_test_class, y_pred_class))
#
# # =====================================
# # Feature Importance (with LightGBM Classifier)
# # =====================================
# lgbm_class = LGBMClassifier(n_estimators=100, random_state=42, scale_pos_weight=1.1, force_col_wise=True)
# lgbm_class.fit(X_train_class, y_train_class)
# importances = lgbm_class.feature_importances_ / lgbm_class.feature_importances_.sum()
# feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# print("\nFeature Importance (LightGBM Classifier):")
# print(feature_importance_df)
#
# # Plot chart
# plt.figure(figsize=(12, 8))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
# plt.title('Feature Importance from LightGBM Classifier')
# plt.show()
#
# # Select important features
# important_features = feature_importance_df[feature_importance_df['Importance'] > 0.01]['Feature'].tolist()
# print("\nImportant Features:", important_features)
#
# # Update data with important features
# X_important_scaled = scaler.fit_transform(df[important_features])
# X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_important_scaled, y_class, test_size=0.2, random_state=42)
#
# # =====================================
# # Hyperparameter tuning for LightGBM Classifier
# # =====================================
# param_grid_class = {
#     'n_estimators': [100, 150],
#     'learning_rate': [0.1],
#     'max_depth': [7, 10],
#     'num_leaves': [31, 50],
#     'min_child_samples': [20],
#     'subsample': [0.8]
# }
# lgbm_class = LGBMClassifier(random_state=42, scale_pos_weight=1.1, force_col_wise=True)
# grid_search_class = GridSearchCV(estimator=lgbm_class, param_grid=param_grid_class, cv=3, scoring='f1_weighted', n_jobs=1)
# grid_search_class.fit(X_train_class, y_train_class)
#
# print("\nBest Parameters for LightGBM Classification:", grid_search_class.best_params_)
# best_lgbm_class = grid_search_class.best_estimator_
#
# y_pred_proba = best_lgbm_class.predict_proba(X_test_class)[:, 1]
# for threshold in [0.3, 0.4, 0.5]:
#     y_pred_class = (y_pred_proba > threshold).astype(int)
#     accuracy = accuracy_score(y_test_class, y_pred_class)
#     print(f"\nBest LightGBM (Tuned, Threshold={threshold}):")
#     print(f"Accuracy: {accuracy:.2f}")
#     print(classification_report(y_test_class, y_pred_class))
#
# # =====================================
# # Save models
# # =====================================
# dump(best_lgbm_class, 'best_lgbm_classification_model.joblib')
# dump(ensemble_model, 'ensemble_model.joblib')
# dump(scaler, 'scaler.joblib')
# dump(important_features, 'important_features.joblib')
#
# print("\nModels saved successfully!")