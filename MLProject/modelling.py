import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    df = pd.read_csv("vgsales_preprocessing.csv")

    if "High_Sales" not in df.columns:
        median_sales = df["Global_Sales"].median()
        df["High_Sales"] = (df["Global_Sales"] >= median_sales).astype(int)

    num_features = ["Year", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
    cat_features = ["Platform", "Genre", "Publisher"]

    X = df[num_features + cat_features]
    y = df["High_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    print("Model training berhasil (CI Workflow).")

if __name__ == "__main__":
    main()
