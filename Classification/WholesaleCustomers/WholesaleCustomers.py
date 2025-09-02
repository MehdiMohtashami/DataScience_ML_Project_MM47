# import pandas as pd
# df_dataset= pd.read_csv('Wholesale customers data.csv')
# #For wholesale customers data.csv
# # print('#'*40,'For Wholesale customers data.csv', '#'*40)
# # print(df_dataset.describe(include='all').to_string())
# # print(df_dataset.shape)
# # print(df_dataset.columns)
# # print(df_dataset.info)
# # print(df_dataset.dtypes)
# # print(df_dataset.isna().sum())
# # print(df_dataset.head(10).to_string())
# # print('='*90)
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Style setting
# sns.set(style="whitegrid")
#
# # Histogram for each feature
# df_dataset.hist(bins=20, figsize=(12, 10))
# plt.suptitle('Histograms of Features')
# plt.show()
#
# # Boxplot for outliers
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=df_dataset.drop(['Channel', 'Region'], axis=1)) # Without Channel and Region
# plt.title('Boxplots for Spending Features')
# plt.show()
#
# # Correlation Heatmap (to see relationships between features)
# plt.figure(figsize=(10, 8))
# corr = df_dataset.corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()
#
# # Scatterplot for High correlation features (like Grocery and Detergents_Paper)
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='Grocery', y='Detergents_Paper', data=df_dataset, hue='Channel') # Color by Channel
# plt.title('Scatterplot: Grocery vs Detergents_Paper by Channel')
# plt.show()
#
# # Pairplot for overall relationships
# sns.pairplot(df_dataset, hue='Channel') # Color by Channel to see class separation
# plt.show()
#
#
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
#
# # Example: Predicting Grocery based on Detergents_Paper
# X = df_dataset[['Detergents_Paper']]
# y = df_dataset['Grocery']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(f'R2 Score: {r2_score(y_test, y_pred):.2f}') # high accuracy because of strong correlation
#
#
#
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
#
# # prepare data (target: Channel, Features: The rest without Region because it may leak)
# X = df_dataset.drop(['Channel', 'Region'], axis=1)
# y = df_dataset['Channel']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Scaling
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Models
# models = {
# 'Logistic Regression': LogisticRegression(),
# 'Random Forest': RandomForestClassifier(random_state=42),
# 'SVM': SVC(),
# 'KNN': KNeighborsClassifier()
# }
#
# for name, model in models.items():
# model.fit(X_train_scaled, y_train)
# y_pred = model.predict(X_test_scaled)
# acc = accuracy_score(y_test, y_pred)
# print(f'{name} Accuracy: {acc:.2f}')
# print(classification_report(y_test, y_pred))
#
# # For example, for linear regression if you want to check relationships, but not for classification
#
#
## from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import silhouette_score
#
# # Features (without Channel and Region, because unsupervised)
# X = df_dataset.drop(['Channel', 'Region'], axis=1)
# X_scaled = StandardScaler().fit_transform(X)
#
# # Finding the best number of clusters for K-Means (with Elbow method)
# inertias = []
# sil_scores = []
# for k in range(2, 10):
# kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
# kmeans.fit(X_scaled)
# inertias.append(kmeans.inertia_)
# sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))
#
# # Plot Elbow
# plt.plot(range(2, 10), inertias, marker='o')
# plt.title('Elbow Method for K')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.show()
#
# # Plot Silhouette
# plt.plot(range(2, 10), sil_scores, marker='o')
# plt.title('Silhouette Scores')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Silhouette Score')
# plt.show()
#
# # Models (assuming best k=6 based on analyses)
# kmeans = KMeans(n_clusters=6, random_state=42, n_init=10).fit(X_scaled)
# hier = AgglomerativeClustering(n_clusters=6).fit(X_scaled)
# gmm = GaussianMixture(n_components=6, random_state=42).fit(X_scaled)
#
# # Evaluation
# print(f'K-Means Silhouette: {silhouette_score(X_scaled, kmeans.labels_):.2f}')
# print(f'Hierarchical Silhouette: {silhouette_score(X_scaled, hier.labels_):.2f}')
# print(f'GMM Silhouette: {silhouette_score(X_scaled, gmm.predict(X_scaled)):.2f}')
#
# # Add clusters to data and plot
# df_dataset['Cluster'] = kmeans.labels_ # For exampleK-Means
# # sns.pairplot(df_dataset, hue='Cluster')
# # plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
import joblib

# Load data
df = pd.read_csv('Wholesale customers data.csv')

# Section 1: Logistic Regression with Hyperparameter Tuning (GridSearchCV)
X_class = df.drop(['Channel', 'Region'], axis=1)
y_class = df['Channel']
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

scaler_class = StandardScaler()
X_train_scaled = scaler_class.fit_transform(X_train)
X_test_scaled = scaler_class.transform(X_test)

# Define grid for tuning
param_grid = {
    'C': [0.1, 1, 10, 100],  # Values for regularization strength
    'penalty': ['l1', 'l2'],  # Type of penalty
    'solver': ['liblinear']   # Solver suitable for l1 and l2
}

# GridSearchCV to find the best parameters
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best model
best_lr = grid_search.best_estimator_

# Prediction and evaluation
y_pred = best_lr.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
error_rate = 1 - acc  # Error rate

print(f'Best Parameters for Logistic Regression: {grid_search.best_params_}')
print(f'Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}')
print(f'Test Accuracy: {acc:.2f}')
print(f'Test Error Rate: {error_rate:.2f}')

# Save model and scaler with joblib
joblib.dump(best_lr, 'logistic_regression_model.joblib')
joblib.dump(scaler_class, 'scaler_class.joblib')
print('Logistic Regression model and scaler saved as joblib files.')

# Section 2: K-Means (without extensive tuning, as n_clusters is predetermined)
X_cluster = df.drop(['Channel', 'Region'], axis=1)
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Evaluation
sil = silhouette_score(X_scaled, kmeans.labels_)
print(f'K-Means Silhouette Score: {sil:.2f}')

# Save model and scaler with joblib
# joblib.dump(kmeans, 'kmeans_model.joblib')
# joblib.dump(scaler_cluster, 'scaler_cluster.joblib')
# print('K-Means model and scaler saved as joblib files.')