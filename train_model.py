import pandas as pd
from ucimlrepo import fetch_ucirepo

def main():
    print("Downloading UCI Student Performance dataset (ID 320)...")
    student = fetch_ucirepo(id=320)
    df = pd.concat([student.data.features, student.data.targets], axis=1)
    df = df.drop_duplicates()
    if "G3" not in df.columns:
        raise ValueError("The downloaded dataset does not contain G3.")
    df.to_csv("student_data.csv", index=False)
    print(f"Saved student_data.csv: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    print("\nMissing values:")
    print(df.isna().sum())

if __name__ == "__main__":
    main()
