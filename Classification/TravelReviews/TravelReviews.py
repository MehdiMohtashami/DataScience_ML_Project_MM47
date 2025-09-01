# import pandas as  pd
# df_dataset= pd.read_csv('tripadvisor_review.csv')
# #For tripadvisor_review.csv
# print('#'*40,'For tripadvisor_review.csv', '#'*40)
# print(df_dataset.describe(include='all').to_string())
# print(df_dataset.shape)
# print(df_dataset.columns)
# print(df_dataset.info)
# print(df_dataset.dtypes)
# print(df_dataset.isna().sum())
# print(df_dataset.head(10).to_string())
# print('='*90)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# بارگذاری دیتاست
df = pd.read_csv('tripadvisor_review.csv')

# حذف ستون User ID چون برای تحلیل نیازی بهش نداریم
df_numeric = df.drop('User ID', axis=1)

# ۱. نمودار توزیع (Histogram) برای هر دسته
plt.figure(figsize=(15, 10))
for i, column in enumerate(df_numeric.columns, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df_numeric[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# ۲. نمودار همبستگی (Correlation Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# ۳. نمودار جفتی (Pairplot) برای بررسی روابط بین چند دسته (مثلاً ۴ دسته اول)
sns.pairplot(df_numeric.iloc[:, :4])  # برای کاهش زمان اجرا، فقط ۴ دسته اول
plt.show()

# ۴. نمودار باکس‌پلات برای بررسی پراکندگی و مقادیر پرت
plt.figure(figsize=(15, 6))
sns.boxplot(data=df_numeric)
plt.title('Boxplot of Categories')
plt.xticks(rotation=45)
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# استانداردسازی داده‌ها
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# کاهش ابعاد با PCA
pca = PCA(n_components=2)  # برای ویژوالایزیشن، ۲ بعد کافیه
df_pca = pca.fit_transform(df_scaled)

# نمایش درصد واریانس توضیح‌داده‌شده
print(f'Explained Variance Ratio: {pca.explained_variance_ratio_}')

# ویژوالایزیشن داده‌های کاهش‌یافته
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], alpha=0.5)
plt.title('PCA of TravelReviews')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ۱. K-Means
kmeans_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    kmeans_scores.append(score)
    print(f'K-Means with {k} clusters: Silhouette Score = {score}')

# بهترین تعداد خوشه‌ها
best_k = range(2, 10)[kmeans_scores.index(max(kmeans_scores))]
print(f'Best K: {best_k}')

# ویژوالایزیشن K-Means
kmeans = KMeans(n_clusters=best_k, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(df_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['KMeans_Cluster'], cmap='viridis', alpha=0.5)
plt.title(f'K-Means Clustering (k={best_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# ۲. DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)
if len(set(df['DBSCAN_Cluster'])) > 1:  # اگر خوشه‌ای پیدا شد
    score = silhouette_score(df_scaled, df['DBSCAN_Cluster'])
    print(f'DBSCAN: Silhouette Score = {score}')

# ویژوالایزیشن DBSCAN
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['DBSCAN_Cluster'], cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# ۳. Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=best_k)
df['Hierarchical_Cluster'] = hierarchical.fit_predict(df_scaled)
score = silhouette_score(df_scaled, df['Hierarchical_Cluster'])
print(f'Hierarchical Clustering: Silhouette Score = {score}')

# ویژوالایزیشن Hierarchical
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['Hierarchical_Cluster'], cmap='viridis', alpha=0.5)
plt.title(f'Hierarchical Clustering (k={best_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# ساخت برچسب بر اساس میانگین امتیازات
df['Average_Rating'] = df_numeric.mean(axis=1)
df['Label'] = pd.qcut(df['Average_Rating'], q=3, labels=['Low', 'Medium', 'High'])

# آماده‌سازی داده‌ها برای کلاسیفیکیشن
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

X = df_scaled
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تست چند مدل
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'\n{name}:')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

from sklearn.mixture import GaussianMixture

gmm_scores = []
for k in range(2, 10):
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    gmm_scores.append(score)
    print(f'GMM with {k} components: Silhouette Score = {score}')

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f'Best Parameters: {grid.best_params_}')
print(f'Best CV Score: {grid.best_score_}')

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
importance = pd.DataFrame({'Feature': df_numeric.columns, 'Coefficient': abs(model.coef_[0])})
importance = importance.sort_values(by='Coefficient', ascending=False)
print(importance)

# ویژوالایزیشن اهمیت ویژگی‌ها
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=importance)
plt.title('Feature Importance (Logistic Regression)')
plt.show()

from sklearn.model_selection import cross_val_score

model = LogisticRegression(random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy: {scores.mean()} ± {scores.std()}')

# import joblib
#
# final_model = LogisticRegression(random_state=42)
# final_model.fit(X, y)  # فیت روی کل داده‌ها
# joblib.dump(final_model, 'travel_reviews_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')  # ذخیره اسکیلر برای پیش‌پردازش داده‌های جدید


# import joblib
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
#
# # آماده‌سازی داده‌ها
# df = pd.read_csv('tripadvisor_review.csv')
# df_numeric = df.drop('User ID', axis=1)
# scaler = StandardScaler()
# X = scaler.fit_transform(df_numeric)
# y = pd.qcut(df_numeric.mean(axis=1), q=3, labels=['Low', 'Medium', 'High'])
#
# # آموزش مدل نهایی
# final_model = LogisticRegression(C=100, solver='lbfgs', random_state=42)
# final_model.fit(X, y)
#
# # ذخیره مدل و اسکیلر
# joblib.dump(final_model, 'travel_reviews_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')