# Importing Essential Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models from Scikit-Learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Classification Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# To ignore warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Magic function for plots in Jupyter Notebook (اگر از پایچارم استفاده می‌کنید، این خط ممکن است نیاز نباشد)
# %matplotlib inline
# Load the dataset
df = pd.read_csv('wifi_localization.csv')

# Let's take a first look
print("First 5 rows:")
print(df.head())
print("\n" + "="*50 + "\n")

# Your initial analysis (خیلی خوب بود)
print("Dataset Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescriptive Stats:")
print(df.describe())
print("\nNull Values Check:")
print(df.isna().sum())
print("\n" + "="*50 + "\n")

# Check the balance of the target variable ('Room')
print("Distribution of Target Variable (Room):")
print(df['Room'].value_counts())
# 1. Distribution of the target variable
plt.figure(figsize=(10, 6))
sns.countplot(x='Room', data=df, palette='Set2')
plt.title('Distribution of Rooms')
plt.xlabel('Room Number')
plt.ylabel('Count')
plt.show()

# 2. Distribution of all Wifi signals (Boxplots)
df_wifi = df.drop('Room', axis=1) # Separate the features
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_wifi, palette='viridis')
plt.title('Distribution of WiFi Signal Strengths')
plt.xticks(rotation=45)
plt.savefig('Distribution_WiFi_Signal_Strengths.png')
plt.show()

# 3. Correlation Heatmap (چقدر ویژگی‌ها با هم و با هدف همبستگی دارند؟)
plt.figure(figsize=(12, 8))
# Since Room is categorical, we need to encode it temporarily for correlation calc
corr_df = df.copy()
corr_matrix = corr_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.savefig('Correlation_Matrix_Heatmap.png')
plt.show()

# 4. Pairplot to see relationships (ممکن است به خاطر تعداد ویژگی زیاد شلوغ شود)
# ما یک نمونه تصادفی می‌گیریم تا نمودار قابل مدیریت باشد
sample_df = df.sample(n=100, random_state=42) # 100 sample for clarity
sns.pairplot(sample_df, hue='Room', palette='Set1', corner=True)
plt.suptitle('Pair-plot of Features (Sampled 100 points)', y=1.02)
plt.savefig('Pair-plot_Features.png')
plt.show()

# Separate Features (X) and Target (y)
X = df.drop('Room', axis=1)
y = df['Room']

# Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 'stratify=y' ensures the class distribution is same in both train and test sets

# Feature Scaling (خیلی مهم برای مدل‌هایی مانند SVM، KNN و Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Note: We don't need to encode 'Room' (target) because it's already numerical (1,2,3,4).

# Initialize a list of models with default parameters
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Linear SVM': SVC(kernel='linear', random_state=42),
    'RBF SVM': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

# Dictionary to store the accuracy results
results = {}

# Iterate through each model, train it, and store the accuracy
for name, model in models.items():
    # For models that benefit greatly from scaling
    if name in ['Logistic Regression', 'K-Nearest Neighbors', 'Linear SVM', 'RBF SVM', 'Naive Bayes']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:  # For tree-based models that don't need scaling
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")

    # Print a detailed report for one model as an example (e.g., Random Forest)
    if name == 'Random Forest':
        print("\nDetailed Classification Report for Random Forest:")
        print(classification_report(y_test, y_pred))
        # Plot confusion matrix for Random Forest
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
        plt.title('Confusion Matrix - Random Forest')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('ConfusionMatrix-RandomForest.png')
        plt.show()
    print("-" * 50)

# Compare all models
print("\nModel Comparison based on Accuracy:")
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df = results_df.sort_values('Accuracy', ascending=False)
print(results_df)

# Plot the comparison
plt.figure(figsize=(12, 8))
sns.barplot(x='Accuracy', y='Model', data=results_df, palette='rocket')
plt.title('Model Comparison (Accuracy)')
plt.xlim(0.9, 1.0)  # Zoom in on the high accuracy range
plt.savefig('ModelComparison_Accuracy.png')
plt.show()

# Importing Essential Libraries
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# import joblib
#
# # 1. Load the Dataset
# df = pd.read_csv('wifi_localization.csv')
#
# # 2. Separate Features (X) and Target (y)
# X = df.drop('Room', axis=1)
# y = df['Room']
#
# # 3. Split the data into Training and Testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # 4. Initialize and Fit the Scaler ONLY ONCE on the training data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train) # Fit and Transform the training data
# # We don't transform X_test here because we are just training and saving models.
#
# # 5. Train and Save K-Nearest Neighbors (KNN) - OUR MAIN CHOICE
# print("Training KNN model...")
# knn_model = KNeighborsClassifier()
# knn_model.fit(X_train_scaled, y_train) # Train on SCALED data
# joblib.dump(knn_model, 'knn_wifi_model.pkl')
# print("KNN Model saved as 'knn_wifi_model.pkl'")
#
# # 6. Train and Save Linear SVM
# print("Training Linear SVM model...")
# svm_linear_model = SVC(kernel='linear', random_state=42)
# svm_linear_model.fit(X_train_scaled, y_train) # Train on SCALED data
# joblib.dump(svm_linear_model, 'svm_linear_wifi_model.pkl')
# print("Linear SVM Model saved as 'svm_linear_wifi_model.pkl'")
#
# # 7. Train and Save Naive Bayes
# print("Training Naive Bayes model...")
# nb_model = GaussianNB()
# nb_model.fit(X_train_scaled, y_train) # Train on SCALED data
# joblib.dump(nb_model, 'nb_wifi_model.pkl')
# print("Naive Bayes Model saved as 'nb_wifi_model.pkl'")
#
# # 8. Save the Scaler (THE MOST IMPORTANT PART!)
# print("Saving the scaler...")
# joblib.dump(scaler, 'standard_scaler.pkl')
# print("Scaler saved as 'standard_scaler.pkl'")
#
# print("\n✅ All 3 models and 1 scaler have been saved successfully!")
# print("📁 Total files created: 4 (.pkl files)")