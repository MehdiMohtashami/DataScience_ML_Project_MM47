# import pandas as  pd
# df_dataset= pd.read_csv('Data_User_Modeling_Dataset_Hamdi Tolga.csv')
# #For Data_User_Modeling_Dataset_Hamdi Tolga.csv
# print('#'*40,'For Data_User_Modeling_Dataset_Hamdi Tolga.csv', '#'*40)
# print(df_dataset.describe(include='all').to_string())
# print(df_dataset.shape)
# print(df_dataset.columns)
# print(df_dataset.info)
# print(df_dataset.dtypes)
# print(df_dataset.isna().sum())
# print(df_dataset.head(10).to_string())
# print('='*90)

# Importing Necessary Libraries
# =============================================================================
# 1. Data Manipulation and Analysis
import pandas as pd
import numpy as np

# 2. Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 3. Data Preprocessing and Model Evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, f1_score)

# 4. Machine Learning Models (A wide variety)
# 4.1 Linear Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# 4.2 Tree-Based Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# 4.3 Distance-Based Models
from sklearn.neighbors import KNeighborsClassifier
# 4.4 Naive Bayes
from sklearn.naive_bayes import GaussianNB
# 4.5 Clustering (for the sake of exploration, we'll use K-Means)
from sklearn.cluster import KMeans

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings('ignore')

# Load the Dataset
# =============================================================================
# Note: There seems to be an extra space in column name ' UNS'. We'll handle it.
df = pd.read_csv('Data_User_Modeling_Dataset_Hamdi Tolga.csv')
# Strip whitespace from column names
df.columns = df.columns.str.strip()
print("Columns after stripping whitespace:", df.columns.tolist())

# Exploratory Data Analysis (EDA)
# =============================================================================
print("\n" + "=" * 50 + " EXPLORATORY DATA ANALYSIS " + "=" * 50)
print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())
print("\nTarget Variable ('UNS') Distribution:")
print(df['UNS'].value_counts())

# Data Visualization
# =============================================================================
print("\n" + "=" * 50 + " DATA VISUALIZATION " + "=" * 50)

# 1. Distribution of the Target Variable
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='UNS', order=df['UNS'].value_counts().index)
plt.title('Distribution of Knowledge Level (UNS)')
plt.show()

# 2. Boxplots of all features wrt the target variable
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows, 3 columns
axes = axes.ravel()  # Flatten the 2D array of axes for easy indexing
for i, col in enumerate(df.columns[:-1]):  # Exclude the last column ('UNS')
    sns.boxplot(data=df, x='UNS', y=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col} by Knowledge Level')
plt.tight_layout()
plt.show()

# 3. Pairplot to see relationships and distributions
# It might be heavy if you have many features, but ours is manageable.
sns.pairplot(df, hue='UNS', diag_kind='kde')
plt.suptitle('Pairplot of Features by Knowledge Level', y=1.02)
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Data Preprocessing for Machine Learning
# =============================================================================
print("\n" + "=" * 50 + " DATA PREPROCESSING " + "=" * 50)

# Encode the target variable (convert labels to numbers)
# Example: very_low -> 0, Low -> 1, Middle -> 2, High -> 3
le = LabelEncoder()
df['UNS_encoded'] = le.fit_transform(df['UNS'])
# Map the encoding for reference
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Target Class Mapping:", class_mapping)

# Define Features (X) and Target (y)
X = df.drop(['UNS', 'UNS_encoded'], axis=1)  # Keep only the feature columns
y = df['UNS_encoded']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Standardize the features (very important for SVM, KNN, etc.)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training and Evaluation
# =============================================================================
print("\n" + "=" * 50 + " MODEL COMPARISON " + "=" * 50)

# Define a list of models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'k-NN': KNeighborsClassifier(),
    'Linear SVM': SVC(kernel='linear', random_state=42),
    'RBF SVM': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

# Dictionary to store the results
results = {}

# Iterate over each model, train it, and evaluate
for name, model in models.items():
    # Train the model on the scaled training data
    if name in ['Logistic Regression', 'k-NN', 'Linear SVM', 'RBF SVM']:
        # These models benefit greatly from scaling
        model.fit(X_train_scaled, y_train)
        # Make predictions on the scaled test set
        y_pred = model.predict(X_test_scaled)
    else:
        # Tree-based models are often scale-invariant
        model.fit(X_train, y_train)
        # Make predictions on the original test set
        y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Calculate F1-Score (macro average to handle multi-class)
    f1 = f1_score(y_test, y_pred, average='macro')

    # Store the results
    results[name] = {'Accuracy': accuracy, 'F1-Score': f1}

    # Print the result for this model
    print(f"{name:20} | Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}")

# Convert results to a DataFrame for easier viewing and plotting
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.sort_values(by='Accuracy', ascending=False, inplace=True)
print("\n" + "=" * 30 + " FINAL RANKING " + "=" * 30)
print(results_df.to_string())

# Visualize the model comparison
plt.figure(figsize=(12, 8))
scores = results_df[['Accuracy', 'F1-Score']]
scores.plot(kind='barh')
plt.title('Model Comparison - Accuracy and F1-Score')
plt.xlabel('Score')
plt.xlim(0.7, 1.0)  # Zoom in to see differences more clearly
plt.tight_layout()
plt.show()

# Detailed Analysis of the Top Performer
# =============================================================================
print("\n" + "=" * 50 + " DETAILED ANALYSIS OF TOP MODEL " + "=" * 50)
# Get the name of the model with the highest accuracy
top_model_name = results_df.index[0]
print(f"Top Performing Model: {top_model_name}")

# Retrain the top model (using scaled data if it was used before)
if top_model_name in ['Logistic Regression', 'k-NN', 'Linear SVM', 'RBF SVM']:
    top_model = models[top_model_name]
    top_model.fit(X_train_scaled, y_train)
    y_pred = top_model.predict(X_test_scaled)
    X_test_for_cm = X_test_scaled
else:
    top_model = models[top_model_name]
    top_model.fit(X_train, y_train)
    y_pred = top_model.predict(X_test)
    X_test_for_cm = X_test

# Generate a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Plot a confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix for {top_model_name}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# (Optional) Hyperparameter Tuning for the Top Model using GridSearchCV
# =============================================================================
print("\n" + "=" * 50 + " HYPERPARAMETER TUNING (Optional) " + "=" * 50)
# Example for Random Forest. You would change the param_grid based on the top model.
if isinstance(top_model, RandomForestClassifier):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    # Fit on the correct data (scaled or not)
    grid_search.fit(X_train, y_train)  # RF doesn't need scaling
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    best_rf = grid_search.best_estimator_
    y_pred_tuned = best_rf.predict(X_test)
    print(f"Tuned Model Accuracy on Test Set: {accuracy_score(y_test, y_pred_tuned):.4f}")
# Add similar blocks for other model types (SVM, etc.)

# Clustering Exploration (K-Means)
# =============================================================================
print("\n" + "=" * 50 + " CLUSTERING EXPLORATION (K-Means) " + "=" * 50)
# We use the scaled data for clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # We know there are 4 classes
cluster_labels = kmeans.fit_predict(X_train_scaled)

# Add cluster labels to the training data for analysis
df_train_with_clusters = X_train.copy()
df_train_with_clusters['True_Label'] = y_train
df_train_with_clusters['KMeans_Cluster'] = cluster_labels

# Cross-tabulation to see if clusters match the true labels
ct = pd.crosstab(df_train_with_clusters['True_Label'], df_train_with_clusters['KMeans_Cluster'])
print("\nCross-Tabulation: True Labels vs K-Means Clusters")
print(ct)

# Note: The cluster numbers are arbitrary. We need to see if they group similar true labels together.

# پس از آموزش و رضایت از مدل نهایی (مثلاً Linear SVM)
# import joblib
#
# # ذخیره مدل
# best_model = SVC(kernel='linear', random_state=42)
# best_model.fit(X_train_scaled, y_train) # حتماً روی داده scaled آموزش بده
# joblib.dump(best_model, 'user_knowledge_model.pkl')
#
# # ذخیره Scaler (بسیار مهم!)
# joblib.dump(scaler, 'scaler.pkl')
#
# # ذخیره LabelEncoder (برای decode کردن پیش‌بینی‌ها)
# joblib.dump(le, 'label_encoder.pkl')