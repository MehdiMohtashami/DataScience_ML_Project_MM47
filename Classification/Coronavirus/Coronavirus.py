# import pandas as  pd
# df_covid_19= pd.read_csv('covid_19_data.csv')
# df_covid_19_confirmed= pd.read_csv('time_series_covid_19_confirmed.csv')
# df_covid_19_recovered= pd.read_csv('time_series_covid_19_recovered.csv')
# df_covid_19_deaths= pd.read_csv('time_series_covid_19_deaths.csv')
# #For df_covid_19
# print('#'*20,'For covid_19_data.csv', '#'*20)
# print(df_covid_19.describe(include='all').to_string())
# print(df_covid_19.shape)
# print(df_covid_19.columns)
# print(df_covid_19.info)
# print(df_covid_19.dtypes)
# print(df_covid_19.isna().sum())
# print(df_covid_19.head(10).to_string())
# print('='*70)
# # For df_covid_19_confirmed.csv
# print('#'*20,'For df_covid_19_confirmed.csv', '#'*20)
# print(df_covid_19_confirmed.describe(include='all').to_string())
# print(df_covid_19_confirmed.shape)
# print(df_covid_19_confirmed.columns)
# print(df_covid_19_confirmed.info)
# print(df_covid_19_confirmed.dtypes)
# print(df_covid_19_confirmed.isna().sum())
# print(df_covid_19_confirmed.head(10).to_string())
# print('='*70)
# # For df_covid_19_recovered.csv
# print('#'*20,'For df_covid_19_recovered.csv', '#'*20)
# print(df_covid_19_recovered.describe(include='all').to_string())
# print(df_covid_19_recovered.shape)
# print(df_covid_19_recovered.columns)
# print(df_covid_19_recovered.info)
# print(df_covid_19_recovered.dtypes)
# print(df_covid_19_recovered.isna().sum())
# print(df_covid_19_recovered.head(10).to_string())
# print('='*70)
# # For df_covid_19_deaths.csv
# print('#'*20,'For df_covid_19_deaths.csv', '#'*20)
# print(df_covid_19_deaths.describe(include='all').to_string())
# print(df_covid_19_deaths.shape)
# print(df_covid_19_deaths.columns)
# print(df_covid_19_deaths.info)
# print(df_covid_19_deaths.dtypes)
# print(df_covid_19_deaths.isna().sum())
# print(df_covid_19_deaths.head(10).to_string())
# print('='*70)



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
#
# # Load datasets
# df_main = pd.read_csv('covid_19_data.csv')
# df_confirmed = pd.read_csv('time_series_covid_19_confirmed.csv')
# df_recovered = pd.read_csv('time_series_covid_19_recovered.csv')
# df_deaths = pd.read_csv('time_series_covid_19_deaths.csv')
#
# # Convert wide to long for time series data
# def convert_wide_to_long(df, value_name):
#     df_long = df.melt(
#         id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
#         var_name='Date',
#         value_name=value_name
#     )
#
#     df_long['Date'] = pd.to_datetime(df_long['Date'], format='%m/%d/%y')
#     return df_long
#
# df_confirmed_long = convert_wide_to_long(df_confirmed, 'Confirmed')
# df_recovered_long = convert_wide_to_long(df_recovered, 'Recovered')
# df_deaths_long = convert_wide_to_long(df_deaths, 'Deaths')
#
# # Merge all time series data
# df_merged = df_confirmed_long.merge(
#     df_recovered_long, on=['Province/State', 'Country/Region', 'Lat', 'Long', 'Date'], how='outer'
# ).merge(
#     df_deaths_long, on=['Province/State', 'Country/Region', 'Lat', 'Long', 'Date'], how='outer'
# )
#
# # Convert Date in the main dataset to datetime
# df_main['ObservationDate'] = pd.to_datetime(df_main['ObservationDate'])
#
# # Merge with the main dataset
# df_combined = df_main.merge(
#     df_merged,
#     left_on=['Province/State', 'Country/Region', 'ObservationDate'],
#     right_on=['Province/State', 'Country/Region', 'Date'],
#     how='left'
# )
#
# # Drop unnecessary columns
# df_combined.drop(['SNo', 'Last Update', 'Date'], axis=1, inplace=True)
#
# # Fill missing values
# df_combined['Province/State'] = df_combined['Province/State'].fillna('Unknown')
# df_combined['Confirmed'] = df_combined['Confirmed_y'].fillna(df_combined['Confirmed_x'])
# df_combined['Deaths'] = df_combined['Deaths_y'].fillna(df_combined['Deaths_x'])
# df_combined['Recovered'] = df_combined['Recovered_y'].fillna(df_combined['Recovered_x'])
# df_combined.drop(['Confirmed_x', 'Confirmed_y', 'Deaths_x', 'Deaths_y', 'Recovered_x', 'Recovered_y'], axis=1, inplace=True)
#
# # Create new features
# df_combined['Mortality_Rate'] = df_combined['Deaths'] / df_combined['Confirmed'].replace(0, np.nan)
# df_combined['Recovery_Rate'] = df_combined['Recovered'] / df_combined['Confirmed'].replace(0, np.nan)
# df_combined['Active_Cases'] = df_combined['Confirmed'] - df_combined['Deaths'] - df_combined['Recovered']
# df_combined['Day_of_Week'] = df_combined['ObservationDate'].dt.dayofweek
# df_combined['Month'] = df_combined['ObservationDate'].dt.month
#
# # Fill NaN values
# df_combined.fillna(0, inplace=True)
#
#
# # Set style
# sns.set_style("whitegrid")
# plt.figure(figsize=(15, 10))
#
# # 1. Distribution of confirmed cases
# plt.subplot(2, 2, 1)
# sns.histplot(np.log1p(df_combined['Confirmed']), kde=True)
# plt.title('Log Distribution of Confirmed Cases')
#
# # 2. Mortality rate over time
# plt.subplot(2, 2, 2)
# mortality_over_time = df_combined.groupby('ObservationDate')['Mortality_Rate'].mean()
# mortality_over_time.plot()
# plt.title('Average Mortality Rate Over Time')
# plt.xticks(rotation=45)
#
# # 3. Top 10 countries by confirmed cases
# plt.subplot(2, 2, 3)
# top_countries = df_combined.groupby('Country/Region')['Confirmed'].max().nlargest(10)
# top_countries.plot(kind='barh')
# plt.title('Top 10 Countries by Confirmed Cases')
#
# # 4. Correlation heatmap
# plt.subplot(2, 2, 4)
# numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Lat', 'Long', 'Mortality_Rate', 'Recovery_Rate', 'Active_Cases']
# sns.heatmap(df_combined[numeric_cols].corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
#
# plt.tight_layout()
# plt.show()
#
#
# # Define target variable - let's predict if cases will double in next week
# # This is just an example, you can define your own target variable
# df_combined = df_combined.sort_values(['Country/Region', 'Province/State', 'ObservationDate'])
# df_combined['Future_Confirmed'] = df_combined.groupby(['Country/Region', 'Province/State'])['Confirmed'].shift(-7)
# df_combined['Cases_Double'] = (df_combined['Future_Confirmed'] >= 2 * df_combined['Confirmed']).astype(int)
# df_combined.dropna(subset=['Future_Confirmed'], inplace=True)
#
# # Encode categorical variables
# le = LabelEncoder()
# df_combined['Country_Encoded'] = le.fit_transform(df_combined['Country/Region'])
# df_combined['Province_Encoded'] = le.fit_transform(df_combined['Province/State'])
#
# # Select features for modeling
# features = ['Lat', 'Long', 'Confirmed', 'Deaths', 'Recovered',
#             'Mortality_Rate', 'Recovery_Rate', 'Active_Cases',
#             'Day_of_Week', 'Month', 'Country_Encoded', 'Province_Encoded']
#
# X = df_combined[features]
# y = df_combined['Cases_Double']
#
# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # Scale the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Define models
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000),
#     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#     'SVM': SVC(kernel='rbf', random_state=42),
#     'K-Nearest Neighbors': KNeighborsClassifier(),
#     'Gradient Boosting': GradientBoostingClassifier(random_state=42),
#     'Decision Tree': DecisionTreeClassifier(random_state=42)
# }
#
# # Train and evaluate models
# results = {}
# for name, model in models.items():
#     model.fit(X_train_scaled, y_train)
#     y_pred = model.predict(X_test_scaled)
#     accuracy = accuracy_score(y_test, y_pred)
#     cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
#
#     results[name] = {
#         'accuracy': accuracy,
#         'cv_mean': cv_scores.mean(),
#         'cv_std': cv_scores.std()
#     }
#
#     print(f"{name}:")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"CV Scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
#     print(classification_report(y_test, y_pred))
#     print("=" * 60)
#
# # Compare model performance
# results_df = pd.DataFrame(results).T
# results_df.sort_values('accuracy', ascending=False, inplace=True)
#
# plt.figure(figsize=(12, 6))
# sns.barplot(x=results_df.index, y=results_df['accuracy'])
# plt.title('Model Comparison - Accuracy')
# plt.xticks(rotation=45)
# plt.ylabel('Accuracy')
# plt.show()
#
# from sklearn.model_selection import GridSearchCV
#
# # Let's assume Random Forest was the best model
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
#
# rf = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train_scaled, y_train)
#
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)
#
# # Use the best model
# best_model = grid_search.best_estimator_
# y_pred_best = best_model.predict(X_test_scaled)
# print("Optimized Model Accuracy:", accuracy_score(y_test, y_pred_best))
#
# # Feature importance
# feature_importance = best_model.feature_importances_
# feature_names = features
#
# importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': feature_importance
# }).sort_values('Importance', ascending=False)
#
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Importance', y='Feature', data=importance_df)
# plt.title('Feature Importance')
# plt.tight_layout()
# plt.show()
#
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
#
# # Load and preprocess data (same as before)
# # ... (previous code for loading and combining data)
#
# # Create target feature: Will cases double in the next 7 days?
# df_combined = df_combined.sort_values(['Country/Region', 'Province/State', 'ObservationDate'])
# df_combined['Future_Confirmed'] = df_combined.groupby(['Country/Region', 'Province/State'])['Confirmed'].shift(-7)
# df_combined['Cases_Double'] = (df_combined['Future_Confirmed'] >= 2 * df_combined['Confirmed']).astype(int)
# df_combined.dropna(subset=['Future_Confirmed'], inplace=True)
#
# # Coding category variables
# le_country = LabelEncoder()
# le_province = LabelEncoder()
# df_combined['Country_Encoded'] = le_country.fit_transform(df_combined['Country/Region'])
# df_combined['Province_Encoded'] = le_province.fit_transform(df_combined['Province/State'])
#
# # Features and purpose
# features = ['Lat', 'Long', 'Confirmed', 'Deaths', 'Recovered',
#             'Mortality_Rate', 'Recovery_Rate', 'Active_Cases',
#             'Day_of_Week', 'Month', 'Country_Encoded', 'Province_Encoded']
# X = df_combined[features]
# y = df_combined['Cases_Double']
#
# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # Standardization
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Train Random Forest model with best parameters
# best_rf = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=10,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     random_state=42
# )
# best_rf.fit(X_train_scaled, y_train)
#
# # Save model, scaler and encoders
# joblib.dump(best_rf, 'random_forest_covid_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(le_country, 'label_encoder_country.pkl')
# joblib.dump(le_province, 'label_encoder_province.pkl')
#
# # You can also save the list of features
# with open('feature_names.txt', 'w') as f:
#     for feature in features:
#         f.write(feature + '\n')
#
# print("Model and preprocessors saved successfully!")
#
# # Load model and preprocessors
# model = joblib.load('random_forest_covid_model.pkl')
# scaler = joblib.load('scaler.pkl')
# le_country = joblib.load('label_encoder_country.pkl')
# le_province = joblib.load('label_encoder_province.pkl')
#
# # Prepare new data for prediction
# new_data = pd.DataFrame({
#     'Lat': [31.8257],
#     'Long': [117.2264],
#     'Confirmed': [100],
#     'Deaths': [2],
#     'Recovered': [50],
#     'Mortality_Rate': [0.02],
#     'Recovery_Rate': [0.5],
#     'Active_Cases': [48],
#     'Day_of_Week': [3],
#     'Month': [2],
#     'Country_Encoded': [le_country.transform(['Mainland China'])[0]],
#     'Province_Encoded': [le_province.transform(['Anhui'])[0]]
# })
#
# # Standardize new data
# new_data_scaled = scaler.transform(new_data)
#
# #  Prediction
# prediction = model.predict(new_data_scaled)
# prediction_proba = model.predict_proba(new_data_scaled)
#
# print(f" Prediction: {prediction[0]}")
# print(f"Probabilities: {prediction_proba[0]}")