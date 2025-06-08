# Prodigy_DS_02

Titanic Dataset: Data Cleaning & Exploratory Data Analysis (EDA)
This repository contains Python code for performing data cleaning and exploratory data analysis (EDA) on the classic Titanic survival dataset. The goal is to understand the dataset, identify patterns, and visualize relationships between various features and the survival outcome.

Project Structure
.
├── titanic_eda.py
└── README.md

Getting Started
Prerequisites
To run this code, you'll need Python installed along with the following libraries:

pandas

numpy

matplotlib

seaborn

You can install them using pip:

pip install pandas numpy matplotlib seaborn

Dataset
This project uses the famous Titanic dataset, typically split into train.csv and test.csv. For this EDA, we primarily focus on train.csv.

Download the train.csv file from Kaggle:
You can download the dataset directly from the official Kaggle competition page:
Titanic - Machine Learning from Disaster Data

Place the train.csv file in the same directory as the titanic_eda.py script.

Running the Code
Navigate to the project directory in your terminal and execute the Python script:

python titanic_eda.py

The script will print various data summaries and display several plots. The plots will pop up in separate windows (or be embedded if you're running in an environment like Jupyter Notebook). Close each plot window to proceed to the next one.

Code Overview (titanic_eda.py)
The titanic_eda.py script performs the following steps:

Load Data: Reads the train.csv file into a pandas DataFrame.

Initial Inspection: Displays the head of the DataFrame, its information (.info()), and descriptive statistics (.describe()).

Data Cleaning:

Handles missing values in 'Age' (imputes with median).

Handles missing values in 'Embarked' (imputes with mode).

Drops the 'Cabin' column due to a high percentage of missing values.

Converts 'Sex' to a numerical representation (0 for male, 1 for female).

Feature Engineering: Creates a new feature FamilySize by combining SibSp (siblings/spouses aboard) and Parch (parents/children aboard).

Exploratory Data Analysis (EDA):

Prints value counts for key categorical features.

Generates various visualizations to explore distributions and relationships:

Histogram of 'Age' distribution.

Histogram of 'Fare' distribution.

Bar plot of 'Survival Rate by Sex'.

Bar plot of 'Survival Rate by Pclass'.

Bar plot of 'Survival Rate by Embarked Port'.

Scatter plot of 'Age vs. Fare' colored by 'Survived'.

Heatmap of the correlation matrix for numerical features.

Visualizations
The script generates several plots to help understand the data:

Histograms: Show the distribution of numerical features like Age and Fare.

Bar Plots: Illustrate survival rates across different categorical groups (Sex, Pclass, Embarked).

Scatter Plots: Help visualize relationships between two numerical variables, often colored by a third categorical variable (e.g., survival).

Heatmap: Provides a quick overview of correlations between all numerical features.

Contributing
Feel free to fork this repository, open issues, or submit pull requests.

License
This project is open source and available under the MIT License.
