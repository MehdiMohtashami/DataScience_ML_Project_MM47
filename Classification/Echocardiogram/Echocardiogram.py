import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Load the dataset
df = pd.read_csv('echocardiogram.csv')

# 1. CREATE THE TARGET VARIABLE ('target')
# Logic: If 'Still-alive' is 1, they survived at least 1 year.
#        If 'Still-alive' is 0, we need to check 'Survival'. If survival < 12, they did not survive 1 year.
def create_target(row):
    if pd.isna(row['Still-alive']) or pd.isna(row['Survival']):
        return np.nan
    if row['Still-alive'] == 1:
        return 1
    else:
        return 1 if row['Survival'] >= 12 else 0

# Apply the function to create the new target column
df['target'] = df.apply(create_target, axis=1)

# 2. DROP IRRELEVANT/REDUNDANT COLUMNS
# We drop the old 'Alive-at-1', the columns used to create it, and the meaningless ones.
df_clean = df.drop(['Name', 'Group', 'Alive-at-1', 'Survival', 'Still-alive'], axis=1)

# 3. HANDLE MISSING VALUES IN THE TARGET
# Drop rows where our newly created target is still NaN (e.g., had missing 'Survival' or 'Still-alive')
df_clean = df_clean.dropna(subset=['target'])

# 4. HANDLE MISSING VALUES IN FEATURES
# Impute missing values in numerical features with the median (more robust to outliers)
numerical_features = ['Age-heart-attack', 'Pericardial-effusion', 'Fractional-shortening', 'Epss', 'Lvdd', 'Wall-motion-score', 'Wall-motion-index', 'Mult']
imputer = SimpleImputer(strategy='median')
df_clean[numerical_features] = imputer.fit_transform(df_clean[numerical_features])

# 5. CHECK THE CLASS BALANCE
print("Target Variable Value Counts:")
print(df_clean['target'].value_counts())
print(f"\nClass Ratio (1: survived): {df_clean['target'].mean():.2f}")

# Separate features (X) and target (y)
X = df_clean.drop('target', axis=1)
y = df_clean['target']

# 6. SPLIT THE DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify helps maintain class ratio

# 7. SCALE THE FEATURES (Crucial for SVM and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# 1. CORRELATION MATRIX HEATMAP
plt.figure(figsize=(10, 8))
# Calculate correlation
corr_matrix = df_clean.corr()
# Create a mask to hide the upper triangle (it's symmetric)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# Plot the heatmap
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# 2. PAIRPLOT OF KEY FEATURES (Can be slow with many features, let's pick a few)
# Let's see how the top correlated features with 'target' relate to each other
top_corr_features = corr_matrix['target'].abs().sort_values(ascending=False).index[1:4] # Skip 'target' itself
sns.pairplot(df_clean, vars=top_corr_features, hue='target', palette='viridis', corner=True)
plt.suptitle("Pairplot of Top Correlated Features with Target", y=1.02)
plt.show()

# 3. BOXPLOTS TO SEE DISTRIBUTION AND OUTLIERS
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel() # Flatten the 2D array of axes for easy indexing
for i, col in enumerate(numerical_features):
    sns.boxplot(x='target', y=col, data=df_clean, ax=axes[i], palette='pastel')
    axes[i].set_title(f'Boxplot of {col} by Survival')
plt.tight_layout()
plt.show()

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM (RBF)": SVC(random_state=42, class_weight='balanced')
}

# We will use F1-Score (weighted average) as our main metric to compare models
scoring = {'f1_weighted': make_scorer(f1_score, average='weighted')}

# Dictionary to store CV results
cv_results = {}

for name, model in models.items():
    # Tree-based models don't need scaling
    if "Logistic" in name or "SVM" in name:
        X_data = scaler.fit_transform(X)  # Scale the entire X for CV
    else:
        X_data = X

    # Perform Cross-Validation (5-Fold)
    scores = cross_validate(model, X_data, y, cv=5, scoring=scoring, return_train_score=False)

    # Store the results
    cv_results[name] = {
        'model': model,
        'f1_weighted_mean': scores['test_f1_weighted'].mean(),
        'f1_weighted_std': scores['test_f1_weighted'].std()
    }

    # Print CV results for each model
    print(f"\n{name} - Cross Validation Results:")
    print("=" * 40)
    print(
        f"F1-Score (Weighted) Mean: {scores['test_f1_weighted'].mean():.3f} (+/- {scores['test_f1_weighted'].std() * 2:.3f})")

# Create a comparison DataFrame
comparison_df = pd.DataFrame.from_dict(cv_results, orient='index')
comparison_df = comparison_df[['f1_weighted_mean', 'f1_weighted_std']]
comparison_df = comparison_df.sort_values('f1_weighted_mean', ascending=False)

print("\n\nFinal Model Ranking based on Weighted F1-Score from Cross-Validation:")
print(comparison_df.to_string())

# Let's also check the performance on the test set for the best model (SVM)
best_model = SVC(random_state=42, class_weight='balanced')
best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)

print("\n\nDetailed Performance of Best Model (SVM) on Test Set:")
print(classification_report(y_test, y_pred_best, target_names=['Did Not Survive (0)', 'Survived (1)']))

# After training the final model, save it with joblib
import joblib

# Train the final model on all training data.
best_model = SVC(random_state=42, class_weight='balanced', kernel='rbf')
best_model.fit(X_train_scaled, y_train)

# Save the trained model
joblib.dump(best_model, 'svm_heart_attack_model.pkl')

# Save the scaler for later use in preprocessing new data.
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved successfully!")