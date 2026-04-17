import os
import re
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required for this script (pip install pandas)")

INPUT_CSV = "NSF_Award_Search.csv"
OUTPUT_CSV = "NSF_Award_Search_cleaned.csv"
OUTPUT_JSON = "NSF_Award_Search_cleaned.json"

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

print(f"Loading {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV, dtype=str, encoding='utf-8', low_memory=False)

# Normalize column names
cols = {c.strip(): c.strip() for c in df.columns}
df.rename(columns=cols, inplace=True)

source_columns = {
    "AwardTitle": "Title",
    "StartDate": "StartDate",
    "EndDate": "EndDate",
    "AwardAmount": "AwardedAmountToDate",
}

for target, source in source_columns.items():
    if source in df.columns:
        df[target] = df[source]

keep_cols = ["AwardTitle", "StartDate", "EndDate", "AwardAmount"]
df = df[[c for c in keep_cols if c in df.columns]]

print("Filtering empty / missing fields...")
df.dropna(subset=keep_cols, inplace=True)

# Normalize AwardAmount
print("Normalizing AwardAmount and filtering zeros...")

def parse_amount(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace("$", "").replace(",", "").replace("USD", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return None


df["AwardAmount"] = df["AwardAmount"].apply(parse_amount)

df.dropna(subset=["AwardAmount"], inplace=True)

df = df[df["AwardAmount"] > 0]

# Normalize dates
print("Parsing StartDate/EndDate and computing timeline...")
for date_col in ["StartDate", "EndDate"]:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)

# Drop rows where parsing failed

df.dropna(subset=["StartDate", "EndDate"], inplace=True)

# timeline in months

def diff_months(start, end):
    return (end.year - start.year) * 12 + (end.month - start.month)


df["timeline_months"] = df.apply(lambda r: diff_months(r["StartDate"], r["EndDate"]), axis=1)

# Clean titles
print("Cleaning AwardTitle text...")

def clean_title(t):
    if pd.isna(t):
        return t
    s = str(t).strip()
    s = re.sub(r"\s+", " ", s)
    # Keep readable chars, punctuation; remove weird symbols
    s = re.sub(r"[^A-Za-z0-9 .,;:'\"\-()&%/\\]+", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


df["AwardTitle"] = df["AwardTitle"].apply(clean_title)

df = df[df["AwardTitle"] != ""]

# Optional but recommended filters
print("Removing timeline out-of-range proposals...")
df = df[(df["timeline_months"] >= 1) & (df["timeline_months"] <= 60)]

# Remove budget outliers using 99th percentile (or IQR)
print("Removing extreme budget outliers...")
max_budget = df["AwardAmount"].quantile(0.99)
if max_budget > 0:
    df = df[df["AwardAmount"] <= max_budget]

print("Final summary:")
print(f"  Rows remaining: {len(df)}")
print(f"  AwardAmount range: {df['AwardAmount'].min():.2f} - {df['AwardAmount'].max():.2f}")
print(f"  Timeline range: {df['timeline_months'].min()} - {df['timeline_months'].max()} months")

print(f"Saving cleaned CSV to {OUTPUT_CSV}...")
df.to_csv(OUTPUT_CSV, index=False)

print(f"Also saving cleaned JSON to {OUTPUT_JSON}...")
df.to_json(OUTPUT_JSON, orient='records', date_format='iso', force_ascii=False)

print("Done.")
