import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Create a Simulated Titanic-like Dataset (for demonstration purposes) ---
# This avoids needing to download a file
data = {
    'PassengerId': range(1, 892),
    'Survived': np.random.randint(0, 2, 891),
    'Pclass': np.random.randint(1, 4, 891),
    'Name': [f'Passenger_{i}' for i in range(1, 892)],
    'Sex': np.random.choice(['male', 'female'], 891),
    'Age': np.random.normal(30, 15, 891),
    'SibSp': np.random.randint(0, 5, 891),
    'Parch': np.random.randint(0, 5, 891),
    'Ticket': [f'Ticket_{i}' for i in range(1, 892)],
    'Fare': np.random.lognormal(2.5, 1.0, 891), # Simulate a skewed fare distribution
    'Cabin': [np.random.choice([f'C{i}', f'B{i}', None]) for i in range(1, 892)],
    'Embarked': np.random.choice(['S', 'C', 'Q', None], 891, p=[0.7, 0.2, 0.08, 0.02])
}

df = pd.DataFrame(data)

# Introduce some NaN values in 'Age' and 'Embarked' to simulate real-world data
nan_indices_age = np.random.choice(df.index, 177, replace=False) # ~20% missing
df.loc[nan_indices_age, 'Age'] = np.nan
nan_indices_embarked = np.random.choice(df.index, 2, replace=False) # 2 missing
df.loc[nan_indices_embarked, 'Embarked'] = np.nan

print("--- Simulated Dataset Created ---")
print(df.head())
print("\n")

# --- 2. Data Cleaning ---

# Handle Missing Values
print("--- Missing Values Before Cleaning ---")
print(df.isnull().sum())
print("\n")

# Fill missing 'Age' values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)
print("Missing 'Age' values filled with median.")

# Fill missing 'Embarked' values with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print("Missing 'Embarked' values filled with mode.")

# 'Cabin' column often has many missing values; it might be dropped or used to extract deck info.
# For simplicity, let's drop it for now if it has too many missing values (e.g., > 70%)
if df['Cabin'].isnull().sum() / len(df) > 0.7:
    df.drop('Cabin', axis=1, inplace=True)
    print("Dropped 'Cabin' column due to high number of missing values.")
else:
    print("'Cabin' column retained (fewer missing values or other strategy needed).")

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())
print("\n")

# Convert 'Sex' to numerical representation (e.g., male=0, female=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
print("Converted 'Sex' to numerical (male=0, female=1).")

# Create a new feature 'FamilySize'
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("Created 'FamilySize' feature.")

# --- 3. Exploratory Data Analysis (EDA) ---

print("\n--- Basic Statistics After Cleaning ---")
print(df.describe())
print("\n")

print("--- Value Counts for Categorical Features ---")
print("\nPclass:\n", df['Pclass'].value_counts())
print("\nSurvived:\n", df['Survived'].value_counts())
print("\nSex:\n", df['Sex'].value_counts())
print("\nEmbarked:\n", df['Embarked'].value_counts())
print("\n")

# Visualizations

# Distribution of 'Age'
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Distribution of 'Fare'
plt.figure(figsize=(10, 6))
sns.histplot(df['Fare'], kde=True, bins=30)
plt.title('Distribution of Fare')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Survival Rate by Sex
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=df, errorbar=None) # errorbar=None to remove confidence intervals
plt.title('Survival Rate by Sex (0: Male, 1: Female)')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.xticks([0, 1], ['Male', 'Female'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Survival Rate by Pclass
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=df, errorbar=None)
plt.title('Survival Rate by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Survival Rate by Embarked Port
plt.figure(figsize=(8, 5))
sns.barplot(x='Embarked', y='Survived', data=df, errorbar=None)
plt.title('Survival Rate by Embarked Port')
plt.xlabel('Embarked Port')
plt.ylabel('Survival Rate')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Age vs. Fare (scatter plot, potentially colored by Survived)
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, alpha=0.7)
plt.title('Age vs. Fare (Colored by Survival)')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Correlation Heatmap
# Select numerical columns for correlation
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'Survived', 'Pclass', 'Sex']
# Ensure all selected columns exist in the DataFrame
numerical_df = df[numerical_cols].dropna()

plt.figure(figsize=(10, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

print("\n--- EDA Complete ---")
