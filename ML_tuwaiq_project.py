#!/usr/bin/env python3

# import modules for dataset
import os
import sys

# clear screen
if os.name == "posix":
    os.system("clear")
elif os.name == "nt":
    os.system("cls")
else:
    sys.stderr.write(f"Error: no clear screen command found to '{os.name}' system.\n")

# print dynamic sections functions
sections = 1;
def sec_print(s):
    global sections

    if sections != 1: print("\n")
    print("#" * 50)
    print(f"{sections} - {s}".center(50))
    print("#" * 50)
    input("Press enter to continue...")
    print("\n")
    sections += 1

###### dataset section ######

# setup variables for dataset
DATA_SET = "weather_type.csv"
URL = "https://raw.githubusercontent.com/Abdulahk1/tuwaiq_project/refs/heads/main/weather_type.csv"

# if dataset not exist
if not os.path.exists(DATA_SET):
    sec_print("Downlaod Dataset")
    input("Press enter to continue...")
    # if 'curl' commnad failed to download dataset (or curl command not found)
    if  os.system(f"curl -O {URL}"):
        sys.stderr.write(f"Error: missing dataset file: '{DATA_SET}'\n")
        sys.stderr.write(f"you can try download it from:\n{URL}\n")
        sys.exit(1)


###### ML section ######

# import libraris

sec_print("Import Modules")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sec_print("Load Dataset")

# load dataset
df_data = pd.read_csv("weather_type.csv")

# General info
sec_print("General Information")

print("Dataset Shape:", df_data.shape)

print("\nInfo:")
df_data.info()

print("\nMissing values:")
print(df_data.isnull().sum())

# Data Exploration (EDA)
sec_print("Data Exploration (EDA)")

# Numeric Columns Statistics
df_data.describe()

# Categorical Value Counts

categorical_cols = ["Cloud Cover", "Season", "Location", "Weather Type"]

for col in categorical_cols:
    print("\nValue counts for:", col)
    print(df_data[col].value_counts())


# Extra EDA Added
sec_print("Extra EDA Added")

# Heatmap Before Encoding

print("Plot is showing now")

plt.figure(figsize=(10,6))
sns.heatmap(df_data.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Before Encoding)")
plt.show()


# Data Visualization
sec_print("Data Visualization")

print("Plot is showing now")
df_data.hist(figsize=(15, 10))
plt.show()

for col in categorical_cols:
    print("Plot is showing now")
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df_data, x=col)
    plt.title(f"Count Plot: {col}")
    plt.xticks(rotation=45)
    plt.show()


# Data Cleaning & Encoding
sec_print("Data Cleaning And Encoding")

le = LabelEncoder()

df_data["Location"] = le.fit_transform(df_data["Location"])
df_data["Cloud Cover"] = le.fit_transform(df_data["Cloud Cover"])
df_data["Season"] = le.fit_transform(df_data["Season"])
df_data["Weather Type"] = le.fit_transform(df_data["Weather Type"])

df_data.head()


# Split Data
sec_print("Split Data")

x = df_data.drop("Weather Type", axis=1)
y = df_data["Weather Type"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=56
)


# Model Training
sec_print("Model Training")

model = DecisionTreeClassifier(max_depth=9, min_samples_split=9)
model.fit(x_train, y_train)


# Model Evaluation
sec_print("Model Evaluation")

pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, average='macro'))
print("Recall:", recall_score(y_test, pred, average='macro'))
print("F1 Score:", f1_score(y_test, pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))


# We chose another algorithm to test whether the model's performance would improve
# We chose a RandomForestClassifier algorithm
# 
# * No need to edit the dataset
# * It has similar performance, but it may be slower than the DecisionTreeClassifier algorithm

from sklearn.ensemble import RandomForestClassifier

model_forest = RandomForestClassifier(max_depth = 15,max_features = 'log2',min_samples_split = 10,n_estimators = 200)
model_forest.fit(x_train, y_train)

forest_pred = model_forest.predict(x_test)

print("Accuracy:", accuracy_score(y_test, forest_pred))
print("Precision:", precision_score(y_test, forest_pred, average='macro'))
print("Recall:", recall_score(y_test, forest_pred, average='macro'))
print("F1 Score:", f1_score(y_test, forest_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, forest_pred))


# Comparison
sec_print("Comparison")
print("Decision Tree Accuracy:", accuracy_score(y_test, pred))
print("Random Forest Accuracy:", accuracy_score(y_test, forest_pred))
