import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv("countries_visa_free_access_cleaned.csv")

# Ensure rank is numeric
df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

# Drop invalid rows
df = df.dropna(subset=["rank"])

# Select top 10 strongest passports
df = df.sort_values("rank").head(10)

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(df["country"], df["rank"], color="steelblue")

# Labels and title
plt.xlabel("Country")
plt.ylabel("Passport Rank")
plt.title("Top 10 Countries with Strongest Passports")

# Improve readability
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("top10_passport_rank_bar_chart.png")
print("✅ Bar chart generated successfully")