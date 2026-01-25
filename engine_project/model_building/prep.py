# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/jarpan03/engine/engine_data.csv"
engine_df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ----------------------------
# Define the target variable
# ----------------------------
target = 'Engine Condition'   # 1 if the engine needs maintenance, else 0

# ----------------------------
# List of numerical features
# ----------------------------
numeric_features = [
    'Engine rpm',                     # number of revolutions per minute (RPM)
    'Lub oil pressure',                # pressure of the lubricating oil in the engine
    'Fuel pressure',         # pressure at which fuel is supplied to the engine
    'Coolant pressure',  # pressure of the engine coolant
    'lub oil temp',       # temperature of the lubricating oil
    'Coolant temp'   # temperature of the engine coolant
]


# ----------------------------
# Combine features to form X (feature matrix)
# ----------------------------
X = engine_df[numeric_features]

# ----------------------------
# Define target vector y
# ----------------------------
y = engine_df[target]

# ----------------------------
# Split dataset into training and test sets
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="jarpan03/engine",
        repo_type="dataset",
    )
