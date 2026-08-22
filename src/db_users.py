"""
User authentication & thread listing — MongoDB backed.

This module handles user identity and thread ownership only.
Actual conversation content lives in LangGraph's SqliteSaver checkpointer,
keyed by thread_id. This module just tracks:
  - Who the user is (email, hashed password, account_id)
  - Which thread_ids belong to them

Design decisions:
  - Passwords hashed with bcrypt (salt built-in, no plaintext stored).
  - MONGO_URI read from environment / .env file — no hardcoded credentials.
  - Each function gets its own client → simple, stateless, fine for a
    low-traffic assessment project. A production version would use a
    connection pool singleton.
  - thread_ids stored as a list on the user doc — simple, no extra
    collection needed for this scale.
"""

import os
from pathlib import Path

import bcrypt
from dotenv import load_dotenv
from pymongo import MongoClient

# ── Load .env from project root ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "parcelpilot"
COLLECTION_NAME = "users"


def _get_collection():
    """Return the users collection from MongoDB."""
    if not MONGO_URI:
        raise RuntimeError(
            "MONGO_URI not set. Add it to .env or export it as an env var."
        )
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]


# ═════════════════════════════════════════════════════════════════════════
# CRUD FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════
def create_user(email: str, password: str, account_id: str) -> dict:
    """
    Create a new user with a hashed password and empty thread list.

    Returns the inserted user doc (without the raw password).
    Raises ValueError if the email already exists.
    """
    coll = _get_collection()

    # Check for duplicate
    if coll.find_one({"email": email}):
        raise ValueError(f"User with email {email} already exists")

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    user_doc = {
        "email": email,
        "password_hash": hashed,  # stored as bytes in MongoDB
        "account_id": account_id,
        "thread_ids": [],
    }
    coll.insert_one(user_doc)

    # Return a safe copy (no password hash)
    return {
        "email": email,
        "account_id": account_id,
        "thread_ids": [],
    }


def authenticate_user(email: str, password: str) -> dict | None:
    """
    Verify email + password. Returns user info dict if valid, else None.

    The returned dict contains: email, account_id, thread_ids.
    Password hash is never returned.
    """
    coll = _get_collection()
    user = coll.find_one({"email": email})

    if not user:
        return None

    if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"]):
        return None

    return {
        "email": user["email"],
        "account_id": user["account_id"],
        "thread_ids": user.get("thread_ids", []),
    }


def add_thread_to_user(email: str, thread_id: str) -> bool:
    """
    Append a thread_id to the user's thread list.

    Uses $addToSet to avoid duplicates.
    Returns True if the user was found and updated.
    """
    coll = _get_collection()
    result = coll.update_one(
        {"email": email},
        {"$addToSet": {"thread_ids": thread_id}},
    )
    return result.matched_count > 0


def get_user_threads(email: str) -> list[str]:
    """Return the list of thread_ids for a user, or empty list if not found."""
    coll = _get_collection()
    user = coll.find_one({"email": email}, {"thread_ids": 1})
    if not user:
        return []
    return user.get("thread_ids", [])


# ═════════════════════════════════════════════════════════════════════════
# SEED + TEST
# ═════════════════════════════════════════════════════════════════════════
def seed_mock_users():
    """
    Create two test users. Safe to re-run — skips if user already exists.

    Accounts (from parcelpilot.db):
      ACCT-001 = Northstar Logistics
      ACCT-002 = LumenWorks
    """
    mock_users = [
        ("northstar@test.com", "password123", "ACCT-001"),
        ("lumenworks@test.com", "password123", "ACCT-002"),
    ]
    for email, password, account_id in mock_users:
        try:
            create_user(email, password, account_id)
            print(f"  ✓ Created {email} → {account_id}")
        except ValueError:
            print(f"  – Skipped {email} (already exists)")


if __name__ == "__main__":
    print("=" * 60)
    print("USER AUTH MODULE — SEED & TEST")
    print("=" * 60)

    # ── Seed ─────────────────────────────────────────────────────────────
    print("\n── Seeding mock users ──")
    seed_mock_users()

    # ── Test 1: Authenticate northstar (correct password) ────────────────
    print("\n── Test 1: authenticate northstar@test.com (correct pw) ──")
    user = authenticate_user("northstar@test.com", "password123")
    if user and user["account_id"] == "ACCT-001":
        print(f"  ✅ PASS — Logged in as {user['email']}, account={user['account_id']}")
    else:
        print(f"  ❌ FAIL — Got: {user}")

    # ── Test 2: Authenticate lumenworks (correct password) ───────────────
    print("\n── Test 2: authenticate lumenworks@test.com (correct pw) ──")
    user = authenticate_user("lumenworks@test.com", "password123")
    if user and user["account_id"] == "ACCT-002":
        print(f"  ✅ PASS — Logged in as {user['email']}, account={user['account_id']}")
    else:
        print(f"  ❌ FAIL — Got: {user}")

    # ── Test 3: Wrong password ───────────────────────────────────────────
    print("\n── Test 3: authenticate northstar@test.com (WRONG pw) ──")
    user = authenticate_user("northstar@test.com", "wrongpassword")
    if user is None:
        print("  ✅ PASS — Returned None (login rejected)")
    else:
        print(f"  ❌ FAIL — Should have returned None, got: {user}")

    # ── Test 4: Non-existent user ────────────────────────────────────────
    print("\n── Test 4: authenticate nonexistent@test.com ──")
    user = authenticate_user("nonexistent@test.com", "password123")
    if user is None:
        print("  ✅ PASS — Returned None (user not found)")
    else:
        print(f"  ❌ FAIL — Should have returned None, got: {user}")

    # ── Test 5: Add thread and retrieve ──────────────────────────────────
    print("\n── Test 5: add_thread_to_user + get_user_threads ──")
    add_thread_to_user("northstar@test.com", "thread-test-001")
    add_thread_to_user("northstar@test.com", "thread-test-002")
    add_thread_to_user("northstar@test.com", "thread-test-001")  # duplicate — should be ignored
    threads = get_user_threads("northstar@test.com")
    if "thread-test-001" in threads and "thread-test-002" in threads and len(threads) >= 2:
        print(f"  ✅ PASS — Threads: {threads}")
        # Check no duplicate
        if threads.count("thread-test-001") == 1:
            print("  ✅ PASS — No duplicate thread_ids ($addToSet works)")
        else:
            print("  ❌ FAIL — Duplicate thread_id found")
    else:
        print(f"  ❌ FAIL — Got: {threads}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
