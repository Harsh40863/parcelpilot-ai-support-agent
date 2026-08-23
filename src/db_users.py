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
import sqlite3
from datetime import datetime

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


def _get_collection_by_name(name: str):
    """Return the specified collection from MongoDB."""
    if not MONGO_URI:
        raise RuntimeError(
            "MONGO_URI not set. Add it to .env or export it as an env var."
        )
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][name]


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

    threads = get_user_threads(email)

    return {
        "email": user["email"],
        "account_id": user["account_id"],
        "thread_ids": threads,
    }


def add_thread_to_user(email: str, thread_id: str, label: str = "New conversation") -> bool:
    """
    Append a thread object to the user's thread list, or update an existing thread's label.

    Returns True if the user was found and updated.
    """
    coll = _get_collection()
    
    # Check if the thread_id already exists in the array
    user = coll.find_one({"email": email, "thread_ids.thread_id": thread_id})
    
    if user:
        # Update existing thread's label
        result = coll.update_one(
            {"email": email, "thread_ids.thread_id": thread_id},
            {"$set": {"thread_ids.$.label": label}}
        )
        return result.matched_count > 0
    else:
        # Append new thread object
        created_at = datetime.utcnow().isoformat() + "Z"
        new_thread = {
            "thread_id": thread_id,
            "label": label,
            "created_at": created_at
        }
        result = coll.update_one(
            {"email": email},
            {"$push": {"thread_ids": new_thread}}
        )
        return result.matched_count > 0


def get_user_threads(email: str) -> list[dict]:
    """Return the list of thread objects for a user, or empty list if not found.
    
    Performs on-the-fly migration for any legacy string thread_ids.
    """
    coll = _get_collection()
    user = coll.find_one({"email": email}, {"thread_ids": 1})
    if not user:
        return []
        
    raw_threads = user.get("thread_ids", [])
    threads = []
    updated = False
    
    for t in raw_threads:
        if isinstance(t, str):
            # Migrate on-the-fly
            threads.append({
                "thread_id": t,
                "label": "New conversation",
                "created_at": datetime.utcnow().isoformat() + "Z"
            })
            updated = True
        elif isinstance(t, dict):
            threads.append(t)
            
    if updated:
        coll.update_one({"email": email}, {"$set": {"thread_ids": threads}})
        
    return threads


def delete_thread_from_user(email: str, thread_id: str) -> bool:
    """
    Remove a thread_id entry from the user's thread_ids array in MongoDB using $pull.
    Supports both new object format and legacy string format.
    """
    coll = _get_collection()
    # Pull object format
    res1 = coll.update_one(
        {"email": email},
        {"$pull": {"thread_ids": {"thread_id": thread_id}}}
    )
    # Pull legacy string format
    res2 = coll.update_one(
        {"email": email},
        {"$pull": {"thread_ids": thread_id}}
    )
    return res1.modified_count > 0 or res2.modified_count > 0


def delete_thread_checkpoints(thread_id: str) -> bool:
    """
    Delete all checkpoints and writes associated with a given thread_id 
    from the SqliteSaver checkpointer (chat_memory.db).
    """
    db_path = PROJECT_ROOT / "data" / "processed" / "chat_memory.db"
    if not db_path.exists():
        return False
        
    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            c1 = conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            c2 = conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            rows_deleted = c1.rowcount + c2.rowcount
            return rows_deleted > 0
    except Exception as e:
        print(f"Error deleting checkpoints for thread {thread_id}: {e}")
        return False
    finally:
        conn.close()


def blacklist_token(token: str) -> bool:
    """
    Store the blacklisted token in token_blacklist collection.
    Creates a TTL index on blacklisted_at set to expire after 86400 seconds (24 hours).
    """
    coll = _get_collection_by_name("token_blacklist")
    try:
        coll.create_index("blacklisted_at", expireAfterSeconds=86400)
    except Exception as e:
        print(f"Warning: could not create TTL index: {e}")
        
    doc = {
        "token": token,
        "blacklisted_at": datetime.utcnow()
    }
    result = coll.insert_one(doc)
    return result.acknowledged


def is_token_blacklisted(token: str) -> bool:
    """
    Check if a token is in the token_blacklist collection.
    """
    coll = _get_collection_by_name("token_blacklist")
    return coll.find_one({"token": token}) is not None


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
    add_thread_to_user("northstar@test.com", "thread-test-001", "Test Label 1")
    add_thread_to_user("northstar@test.com", "thread-test-002", "Test Label 2")
    add_thread_to_user("northstar@test.com", "thread-test-001", "Test Label 1 Updated")  # update label — shouldn't duplicate
    threads = get_user_threads("northstar@test.com")
    thread_ids = [t["thread_id"] for t in threads]
    if "thread-test-001" in thread_ids and "thread-test-002" in thread_ids and len(threads) >= 2:
        print(f"  ✅ PASS — Threads: {threads}")
        # Check no duplicate
        if thread_ids.count("thread-test-001") == 1:
            print("  ✅ PASS — No duplicate thread_ids")
            # Check label update
            t1 = next(t for t in threads if t["thread_id"] == "thread-test-001")
            if t1["label"] == "Test Label 1 Updated":
                print("  ✅ PASS — Label successfully updated")
            else:
                print(f"  ❌ FAIL — Expected label 'Test Label 1 Updated', got '{t1['label']}'")
        else:
            print("  ❌ FAIL — Duplicate thread_id found")
    else:
        print(f"  ❌ FAIL — Got: {threads}")

    # ── Test 6: Delete thread ────────────────────────────────────────────
    print("\n── Test 6: delete_thread_from_user ──")
    delete_thread_from_user("northstar@test.com", "thread-test-001")
    threads = get_user_threads("northstar@test.com")
    thread_ids = [t["thread_id"] for t in threads]
    if "thread-test-001" not in thread_ids:
        print("  ✅ PASS — Thread 'thread-test-001' successfully deleted from MongoDB")
    else:
        print(f"  ❌ FAIL — Thread 'thread-test-001' still exists: {threads}")

    # ── Test 7: Blacklist Token ──────────────────────────────────────────
    print("\n── Test 7: blacklist_token + is_token_blacklisted ──")
    test_token = "eyTestTokenBlacklist123"
    blacklist_token(test_token)
    if is_token_blacklisted(test_token):
        print("  ✅ PASS — Token successfully blacklisted and checked")
    else:
        print("  ❌ FAIL — Token was not blacklisted")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
