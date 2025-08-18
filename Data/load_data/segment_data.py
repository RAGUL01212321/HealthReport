import pandas as pd
import json

# Example: Read first 20 rows
df = pd.read_csv(r"C:\Users\uragu\Desktop\MIMIC IV\discharge.csv\discharge.csv", nrows=20)

# Convert DataFrame to list of dicts
records = df.to_dict(orient="records")

# Write JSON with actual newlines inside strings
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=4)

print("Saved with pretty formatting and real line breaks.")
