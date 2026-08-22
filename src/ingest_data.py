"""
Step 2 — Structured data ingestion: Excel → SQLite

Reads each data sheet (accounts, orders, tickets) from the ParcelPilot
assessment Excel file and loads them into a local SQLite database.

Design decisions:
  - pandas read_excel + to_sql keeps the script simple and preservable.
  - Column names are preserved exactly as they appear in the Excel file.
  - Datetime strings are kept as TEXT in SQLite — parsing happens at query
    time in the agent layer so we don't lose timezone info or precision.
  - if_exists="replace" makes the script idempotent (safe to re-run).
  - Row-count sanity check printed at the end.
"""

import sqlite3
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = PROJECT_ROOT / "data" / "raw" / "ParcelPilot_Assessment_Data.xlsx"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "parcelpilot.db"

# Sheets we want to ingest (skip README — it's metadata, not a data table)
DATA_SHEETS = ["accounts", "orders", "tickets"]


def main():
    # Ensure output directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    for sheet in DATA_SHEETS:
        df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
        # Write to SQLite — table name matches the sheet name
        df.to_sql(sheet, conn, if_exists="replace", index=False)
        print(f"  ✓ {sheet:12s} → {len(df)} rows, {len(df.columns)} columns")

    # ── Sanity check: read back row counts ───────────────────────────────
    print("\n── Verification (row counts from SQLite) ──")
    cursor = conn.cursor()
    for sheet in DATA_SHEETS:
        count = cursor.execute(f"SELECT COUNT(*) FROM {sheet}").fetchone()[0]
        print(f"  {sheet:12s}: {count} rows")

    conn.close()
    print(f"\n✅ Database saved to {DB_PATH}")


if __name__ == "__main__":
    main()
