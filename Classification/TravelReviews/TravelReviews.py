# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load dataset
# df = pd.read_csv('tripadvisor_review.csv')
#
# # Drop User ID column as it is not needed for analysis
# df_numeric = df.drop('User ID', axis=1)
#
# # 1. Histogram for each category
# plt.figure(figsize=(15, 10))
# for i, column in enumerate(df_numeric.columns, 1):
#     plt.subplot(4, 3, i)
#     sns.histplot(df_numeric[column], kde=True)
#     plt.title(f'Distribution of {column}')
# plt.tight_layout()
# plt.show()
#
# # 2. Correlation Heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()
#
# # 3. Pairplot to examine relationships between categories (e.g., first 4 categories)
# sns.pairplot(df_numeric.iloc[:, :4])  # Limited to first 4 categories to reduce runtime
# plt.show()
#
# # 4. Boxplot to examine spread and outliers
# plt.figure(figsize=(15, 6))
# sns.boxplot(data=df_numeric)
# plt.title('Boxplot of Categories')
# plt.xticks(rotation=45)
# plt.show()
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
#
# # Standardize data
# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df_numeric)
#
# # Dimensionality reduction with PCA
# pca = PCA(n_components=2)  # 2 components for visualization
# df_pca = pca.fit_transform(df_scaled)
#
# # Display explained variance ratio
# print(f'Explained Variance Ratio: {pca.explained_variance_ratio_}')
#
# # Visualize reduced data
# plt.figure(figsize=(8, 6))
# plt.scatter(df_pca[:, 0], df_pca[:, 1], alpha=0.5)
# plt.title('PCA of TravelReviews')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()
#
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
# from sklearn.metrics import silhouette_score
#
# # 1. K-Means
# kmeans_scores = []
# for k in range(2, 10):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(df_scaled)
#     score = silhouette_score(df_scaled, labels)
#     kmeans_scores.append(score)
#     print(f'K-Means with {k} clusters: Silhouette Score = {score}')
#
# # Best number of clusters
# best_k = range(2, 10)[kmeans_scores.index(max(kmeans_scores))]
# print(f'Best K: {best_k}')
#
# # Visualize K-Means
# kmeans = KMeans(n_clusters=best_k, random_state=42)
# df['KMeans_Cluster'] = kmeans.fit_predict(df_scaled)
#
# plt.figure(figsize=(8, 6))
# plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['KMeans_Cluster'], cmap='viridis', alpha=0.5)
# plt.title(f'K-Means Clustering (k={best_k})')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()
#
# # 2. DBSCAN
# dbscan = DBSCAN(eps=0.5, min_samples=5)
# df['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)
# if len(set(df['DBSCAN_Cluster'])) > 1:  # If clusters are found
#     score = silhouette_score(df_scaled, df['DBSCAN_Cluster'])
#     print(f'DBSCAN: Silhouette Score = {score}')
#
# # Visualize DBSCAN
# plt.figure(figsize=(8, 6))
# plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['DBSCAN_Cluster'], cmap='viridis', alpha=0.5)
# plt.title('DBSCAN Clustering')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()
#
# # 3. Hierarchical Clustering
# hierarchical = AgglomerativeClustering(n_clusters=best_k)
# df['Hierarchical_Cluster'] = hierarchical.fit_predict(df_scaled)
# score = silhouette_score(df_scaled, df['Hierarchical_Cluster'])
# print(f'Hierarchical Clustering: Silhouette Score = {score}')
#
# # Visualize Hierarchical
# plt.figure(figsize=(8, 6))
# plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['Hierarchical_Cluster'], cmap='viridis', alpha=0.5)
# plt.title(f'Hierarchical Clustering (k={best_k})')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()
#
# # Create labels based on average ratings
# df['Average_Rating'] = df_numeric.mean(axis=1)
# df['Label'] = pd.qcut(df['Average_Rating'], q=3, labels=['Low', 'Medium', 'High'])
#
# # Prepare data for classification
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
#
# X = df_scaled
# y = df['Label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Test multiple models
# models = {
#     'Logistic Regression': LogisticRegression(random_state=42),
#     'Random Forest': RandomForestClassifier(random_state=42),
#     'SVM': SVC(random_state=42)
# }
#
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(f'\n{name}:')
#     print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
#     print(classification_report(y_test, y_pred))
#
# from sklearn.mixture import GaussianMixture
#
# gmm_scores = []
# for k in range(2, 10):
#     gmm = GaussianMixture(n_components=k, random_state=42)
#     labels = gmm.fit_predict(df_scaled)
#     score = silhouette_score(df_scaled, labels)
#     gmm_scores.append(score)
#     print(f'GMM with {k} components: Silhouette Score = {score}')
#
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
# grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5)
# grid.fit(X_train, y_train)
# print(f'Best Parameters: {grid.best_params_}')
# print(f'Best CV Score: {grid.best_score_}')
#
# model = LogisticRegression(random_state=42)
# model.fit(X_train, y_train)
# importance = pd.DataFrame({'Feature': df_numeric.columns, 'Coefficient': abs(model.coef_[0])})
# importance = importance.sort_values(by='Coefficient', ascending=False)
# print(importance)
#
# # Visualize feature importance
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Coefficient', y='Feature', data=importance)
# plt.title('Feature Importance (Logistic Regression)')
# plt.show()
#
# from sklearn.model_selection import cross_val_score
#
# model = LogisticRegression(random_state=42)
# scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
# print(f'Cross-Validation Accuracy: {scores.mean()} ± {scores.std()}')
#
# import joblib
#
# final_model = LogisticRegression(random_state=42)
# final_model.fit(X, y)  # Fit on all data
# joblib.dump(final_model, 'travel_reviews_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')  # Save scaler for preprocessing new data
#
# import joblib
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
#
# # Prepare data
# df = pd.read_csv('tripadvisor_review.csv')
# df_numeric = df.drop('User ID', axis=1)
# scaler = StandardScaler()
# X = scaler.fit_transform(df_numeric)
# y = pd.qcut(df_numeric.mean(axis=1), q=3, labels=['Low', 'Medium', 'High'])
#
# # Train final model
# final_model = LogisticRegression(C=100, solver='lbfgs', random_state=42)
# final_model.fit(X, y)
#
# # Save model and scaler
# joblib.dump(final_model, 'travel_reviews_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')