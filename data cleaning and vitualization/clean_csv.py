import pandas as pd
import numpy as np
import os

print("🚀 Starting data cleaning...")

# File paths
input_csv = r"C:\Users\ADMIN\Desktop\AI Data Analyzer\countries_visa_free_access.csv"
output_csv = r"C:\Users\ADMIN\Desktop\AI Data Analyzer\countries_visa_free_access_cleaned.csv"

# 1️⃣ Load CSV
print("📖 Loading CSV file...")

if not os.path.exists(input_csv):
    raise FileNotFoundError("❌ CSV file not found")

df = pd.read_csv(input_csv)

print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# 2️⃣ Clean column names
print("🧹 Cleaning column names...")

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

print("✅ Cleaned column names:")
print(df.columns.tolist())

# 3️⃣ Trim whitespace in text columns
print("✂️ Trimming whitespace in text columns...")

text_columns = df.select_dtypes(include=["object", "string"]).columns
for col in text_columns:
    df[col] = df[col].astype(str).str.strip()

# 4️⃣ Handle missing values
print("🕳 Handling missing values...")

df.replace(["NA", "N/A", "-", "", "None"], np.nan, inplace=True)

# 5️⃣ Remove duplicates
print("🗑 Removing duplicate rows...")

before = len(df)
df.drop_duplicates(inplace=True)
after = len(df)

print(f"✅ Removed {before - after} duplicate rows")

# 6️⃣ Convert numeric columns (correct for pandas 3)
print("🔢 Converting numeric columns...")

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 7️⃣ Save cleaned CSV
df.to_csv(output_csv, index=False)

print("✅ CLEANING COMPLETE!")
print(f"📄 Cleaned file saved at:\n{output_csv}")