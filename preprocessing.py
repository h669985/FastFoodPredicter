# preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = "fast_food_cleaned.csv"  # Cleaned by ChatGPT5
df = pd.read_csv(csv_path)

target_col = "Company"

non_features_cols = [target_col, "Item"]
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in non_features_cols]

def load_data(seed=None, test_size=0.2):
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols
