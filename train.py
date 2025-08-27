import ast
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train(data_csv: str, model_out: str):
    df = pd.read_csv(data_csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")
    
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        )),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("=== Classification Report (Test) ===")
    print(classification_report(y_test, y_pred))

    out_dir = os.path.dirname(model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    joblib.dump(pipe, model_out)
    print(f"Model saved to: {model_out}")


if __name__ == "__main__":
    # Default values
    data_path = r"sample_tickets.csv"
    out_path = r"ticket_router.joblib"

    # If arguments are passed: python train.py "data.csv" "model.joblib"
    if len(sys.argv) > 1:
        try:
            data_path = ast.literal_eval(sys.argv[1])
        except Exception:
            data_path = sys.argv[1]   # fallback to raw string

    if len(sys.argv) > 2:
        try:
            out_path = ast.literal_eval(sys.argv[2])
        except Exception:
            out_path = sys.argv[2]   # fallback to raw string

    train(data_path, out_path)
