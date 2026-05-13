"""
populate_dim_date.py

One-time setup script. Generates a row for every calendar date between
START_DATE and END_DATE and loads them into dim_date.

Run once after the schema is created. Safe to re-run — INSERT IGNORE
skips any dates that already exist.

Usage:
    python src/pipeline/populate_dim_date.py
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import quote_plus

# --- Locate project root and load credentials ---
# Walk up the directory tree until environment.yml is found.
# This works regardless of where VS Code starts the kernel.
def find_project_root():
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "environment.yml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root — environment.yml not found.")

PROJECT_ROOT = find_project_root()
load_dotenv(PROJECT_ROOT / ".env")

# --- Config ---
START_DATE = "2020-01-01"
END_DATE   = "2027-12-31"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASSWORD")
if DB_PASS is None:
    raise ValueError("DB_PASSWORD not found in .env — check your .env file exists and is populated")
DB_NAME = os.getenv("DB_NAME", "news_pulse")

# --- Build the date dimension dataframe ---
# pandas date_range generates every calendar day in the range.
# We then derive all the calendar attributes from that single column.
def build_dim_date(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="D")

    df = pd.DataFrame({"full_date": dates})
    df["date_id"]     = df["full_date"].dt.strftime("%Y%m%d").astype(int)
    df["year"]        = df["full_date"].dt.year
    df["quarter"]     = df["full_date"].dt.quarter
    df["month"]       = df["full_date"].dt.month
    df["month_name"]  = df["full_date"].dt.strftime("%B")
    df["week"]        = df["full_date"].dt.isocalendar().week.astype(int)
    df["day_of_month"]= df["full_date"].dt.day
    df["day_of_week"] = df["full_date"].dt.dayofweek + 1  # 1 = Monday, 7 = Sunday
    df["day_name"]    = df["full_date"].dt.strftime("%A")
    df["is_weekend"]  = df["day_of_week"].isin([6, 7])

    return df

# --- Load into MySQL ---
def load_dim_date(df: pd.DataFrame) -> None:
    password = quote_plus(DB_PASS)
    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    # Write to a staging table first, then INSERT IGNORE into dim_date.
    # This avoids primary key errors if the script is re-run.
    with engine.begin() as conn:
        df.to_sql("dim_date_staging", conn, if_exists="replace", index=False)

        conn.execute(text("""
            INSERT IGNORE INTO dim_date
                (date_id, full_date, year, quarter, month, month_name,
                 week, day_of_month, day_of_week, day_name, is_weekend)
            SELECT
                date_id, full_date, year, quarter, month, month_name,
                week, day_of_month, day_of_week, day_name, is_weekend
            FROM dim_date_staging
        """))

        conn.execute(text("DROP TABLE IF EXISTS dim_date_staging"))

        result = conn.execute(text("SELECT COUNT(*) FROM dim_date"))
        count  = result.scalar()
        print(f"dim_date loaded — {count} rows")

if __name__ == "__main__":
    print("Building date dimension...")
    df = build_dim_date(START_DATE, END_DATE)
    print(f"Generated {len(df)} rows ({START_DATE} to {END_DATE})")
    load_dim_date(df)
    print("Done.")