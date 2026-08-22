"""
Phase 1 — Core tool functions (plain Python, no agent wiring).

Each function is a self-contained unit that can be tested standalone.
The agent layer (Phase 2+) will call these as tools.

Access-control principle:
  - Structured-data tools (get_order, get_ticket, list_*) enforce row-level
    isolation via account_id filtering at the SQL level.
  - Document search (search_docs) enforces isolation via scope metadata
    filtering on the FAISS index.
  - The agent layer maps the logged-in customer → account_id + scope and
    passes them down. These functions never trust the caller to have already
    filtered — they always apply the filter themselves.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "parcelpilot.db"
FAISS_DIR = PROJECT_ROOT / "data" / "processed" / "faiss_index"
ESCALATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "escalations.json"

# ── Shared resources (lazy-loaded) ───────────────────────────────────────
_vectorstore: Optional[FAISS] = None
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_vectorstore() -> FAISS:
    """Lazy-load the FAISS index so we only pay the cost once per process."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
        )
        _vectorstore = FAISS.load_local(
            str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True
        )
    return _vectorstore


def _query_db(sql: str, params: tuple = ()) -> list[dict]:
    """Run a read-only query and return rows as list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ═════════════════════════════════════════════════════════════════════════
# 1. DOCUMENT SEARCH
# ═════════════════════════════════════════════════════════════════════════
def search_docs(
    query: str,
    scope: Optional[str] = None,
    version: Optional[str] = None,
    k: int = 3,
) -> list[dict]:
    """
    Similarity search over the FAISS document index.

    Args:
        query:   Natural-language search query.
        scope:   If provided, only return chunks whose scope is "all" OR
                 matches this value (e.g. "northstar", "lumenworks").
        version: If provided, only return chunks with this exact version tag
                 (e.g. "v3_current", "v2_deprecated").
        k:       Number of results to return.

    Returns:
        List of {content, metadata, score} dicts, ordered by relevance.
    """
    vs = _get_vectorstore()

    # Build a metadata filter function
    def meta_filter(meta: dict) -> bool:
        if scope and meta.get("scope") not in ("all", scope):
            return False
        if version and meta.get("version") != version:
            return False
        return True

    # Only pass the filter if we actually need one
    needs_filter = scope is not None or version is not None
    results = vs.similarity_search_with_score(
        query, k=k, filter=meta_filter if needs_filter else None
    )

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        }
        for doc, score in results
    ]


# ═════════════════════════════════════════════════════════════════════════
# 2. ACCOUNT LOOKUP
# ═════════════════════════════════════════════════════════════════════════
def get_account(account_id: str) -> Optional[dict]:
    """Return one account row or None if not found."""
    rows = _query_db("SELECT * FROM accounts WHERE account_id = ?", (account_id,))
    return rows[0] if rows else None


# ═════════════════════════════════════════════════════════════════════════
# 3. ORDER LOOKUP (access-controlled)
# ═════════════════════════════════════════════════════════════════════════
def get_order(order_id: str, account_id: str) -> Optional[dict]:
    """
    Return an order only if it belongs to the given account.

    If the order exists but belongs to a different account, returns None
    and prints a warning (potential cross-customer access attempt).
    """
    # Check if the order exists at all
    all_matches = _query_db("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    if not all_matches:
        return None

    # Check ownership
    order = all_matches[0]
    if order["account_id"] != account_id:
        print(
            f"  ⚠ ACCESS DENIED: order {order_id} belongs to "
            f"{order['account_id']}, not {account_id}"
        )
        return None

    return order


# ═════════════════════════════════════════════════════════════════════════
# 4. TICKET LOOKUP (access-controlled)
# ═════════════════════════════════════════════════════════════════════════
def get_ticket(ticket_id: str, account_id: str) -> Optional[dict]:
    """
    Return a ticket only if it belongs to the given account.

    Same access-control pattern as get_order.
    """
    all_matches = _query_db("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
    if not all_matches:
        return None

    ticket = all_matches[0]
    if ticket["account_id"] != account_id:
        print(
            f"  ⚠ ACCESS DENIED: ticket {ticket_id} belongs to "
            f"{ticket['account_id']}, not {account_id}"
        )
        return None

    return ticket


# ═════════════════════════════════════════════════════════════════════════
# 5. LIST ORDERS (scoped to one account)
# ═════════════════════════════════════════════════════════════════════════
def list_orders(account_id: str) -> list[dict]:
    """Return all orders belonging to one account."""
    return _query_db("SELECT * FROM orders WHERE account_id = ?", (account_id,))


# ═════════════════════════════════════════════════════════════════════════
# 6. LIST TICKETS (scoped to one account)
# ═════════════════════════════════════════════════════════════════════════
def list_tickets(account_id: str) -> list[dict]:
    """Return all tickets belonging to one account."""
    return _query_db("SELECT * FROM tickets WHERE account_id = ?", (account_id,))


# ═════════════════════════════════════════════════════════════════════════
# 7. SLA STATUS CALCULATOR
# ═════════════════════════════════════════════════════════════════════════
def calc_sla_status(
    order_id: str,
    account_id: str,
    snapshot_time: str = "2026-08-16 11:00",
) -> Optional[dict]:
    """
    Compute factual SLA timing data for an order.

    This function reports FACTS only:
      - Was the pickup on time or late?
      - How many hours late?
      - Was it carrier_fault or customer_fault?

    It does NOT decide whether a service credit is owed — that policy
    decision lives in the SOP documents and is made by the agent layer
    using search_docs() results.

    Args:
        order_id:      The order to check.
        account_id:    Must match the order's account (access control).
        snapshot_time: The "current" time for the dataset. Used when
                       pickup hasn't happened yet.

    Returns:
        Dict with SLA facts, or None if order not found / access denied.
    """
    order = get_order(order_id, account_id)
    if order is None:
        return None

    fmt = "%Y-%m-%d %H:%M"
    pickup_window_end = datetime.strptime(order["pickup_window_end"], fmt)
    snapshot_dt = datetime.strptime(snapshot_time, fmt)

    # Determine the comparison time: actual pickup if it happened, else snapshot
    if order["pickup_actual_at"] and order["pickup_actual_at"] != "":
        compare_time = datetime.strptime(order["pickup_actual_at"], fmt)
        pickup_happened = True
    else:
        compare_time = snapshot_dt
        pickup_happened = False

    # Calculate delta
    delta: timedelta = compare_time - pickup_window_end
    delta_hours = round(delta.total_seconds() / 3600, 2)

    if delta_hours <= 0:
        sla_status = "on_time"
    else:
        sla_status = "late"

    return {
        "order_id": order_id,
        "account_id": account_id,
        "status": order["status"],
        "pickup_window_end": order["pickup_window_end"],
        "pickup_actual_at": order["pickup_actual_at"],
        "snapshot_time": snapshot_time,
        "pickup_happened": pickup_happened,
        "sla_status": sla_status,
        "hours_late": max(0, delta_hours),
        "carrier_fault": order["carrier_fault"],
        "customer_fault": order["customer_fault"],
    }


# ═════════════════════════════════════════════════════════════════════════
# 8. CREATE ESCALATION (mock action)
# ═════════════════════════════════════════════════════════════════════════
def create_escalation(
    ticket_id: str,
    account_id: str,
    reason: str,
    status: str = "pending_confirmation",
) -> dict:
    """
    Append an escalation record to a local JSON file.

    This is a mock write-action. The agent layer (Phase 3/4) handles
    confirmation flow before calling this function.

    Returns:
        The created escalation record.
    """
    # Load existing escalations (or start fresh)
    if ESCALATIONS_PATH.exists():
        with open(ESCALATIONS_PATH) as f:
            escalations = json.load(f)
    else:
        escalations = []

    record = {
        "escalation_id": f"ESC-{uuid.uuid4().hex[:8].upper()}",
        "ticket_id": ticket_id,
        "account_id": account_id,
        "reason": reason,
        "status": status,
        "created_at": datetime.now().isoformat(),
    }
    escalations.append(record)

    ESCALATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ESCALATIONS_PATH, "w") as f:
        json.dump(escalations, f, indent=2)

    return record


# ═════════════════════════════════════════════════════════════════════════
# TEST BLOCK
# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import pprint

    pp = pprint.PrettyPrinter(indent=2, width=100)

    print("=" * 70)
    print("TOOL FUNCTION TESTS")
    print("=" * 70)

    # ── Test 1: Access control — wrong account_id ────────────────────────
    print("\n── Test 1: get_order() with WRONG account_id ──")
    # ORD-1001 belongs to ACCT-001, try fetching with ACCT-002
    result = get_order("ORD-1001", "ACCT-002")
    if result is None:
        print("  ✅ PASS — Returned None (blocked cross-customer access)")
    else:
        print("  ❌ FAIL — Returned data for wrong account!")
        pp.pprint(result)

    # ── Test 2: Access control — correct account_id ──────────────────────
    print("\n── Test 2: get_order() with CORRECT account_id ──")
    result = get_order("ORD-1001", "ACCT-001")
    if result and result["order_id"] == "ORD-1001":
        print("  ✅ PASS — Returned the correct order")
        pp.pprint(result)
    else:
        print("  ❌ FAIL — Did not return the order!")

    # ── Test 3: Access control on tickets — wrong account ────────────────
    print("\n── Test 3: get_ticket() with WRONG account_id ──")
    # TKT-501 belongs to ACCT-001, try with ACCT-002
    result = get_ticket("TKT-501", "ACCT-002")
    if result is None:
        print("  ✅ PASS — Returned None (blocked cross-customer access)")
    else:
        print("  ❌ FAIL — Returned data for wrong account!")

    # ── Test 4: SLA status calculation ───────────────────────────────────
    print("\n── Test 4: calc_sla_status() on ORD-1002 (picked up late) ──")
    sla = calc_sla_status("ORD-1002", "ACCT-001")
    if sla:
        print(f"  ✅ PASS — SLA status computed")
        pp.pprint(sla)
    else:
        print("  ❌ FAIL — No SLA result returned")

    # ── Test 5: SLA status for order not yet picked up ───────────────────
    print("\n── Test 5: calc_sla_status() on ORD-2002 (not picked up, carrier fault) ──")
    sla = calc_sla_status("ORD-2002", "ACCT-002")
    if sla:
        print(f"  ✅ PASS — SLA status computed")
        pp.pprint(sla)
    else:
        print("  ❌ FAIL — No SLA result returned")

    # ── Test 6: search_docs with version filter ──────────────────────────
    print("\n── Test 6: search_docs() with version='v2_deprecated' filter ──")
    docs = search_docs("support policy", version="v2_deprecated", k=3)
    all_deprecated = all(d["metadata"]["version"] == "v2_deprecated" for d in docs)
    if docs and all_deprecated:
        print(f"  ✅ PASS — {len(docs)} result(s), all v2_deprecated")
        for d in docs:
            src = d["metadata"]["source"]
            ver = d["metadata"]["version"]
            print(f"     src={src}  ver={ver}  score={d['score']:.4f}")
    elif not docs:
        print("  ❌ FAIL — No results returned")
    else:
        print("  ❌ FAIL — Non-deprecated docs leaked through!")
        for d in docs:
            print(f"     ver={d['metadata']['version']}  src={d['metadata']['source']}")

    # ── Test 7: search_docs with scope filter ────────────────────────────
    print("\n── Test 7: search_docs() scope='lumenworks' must NOT return northstar docs ──")
    docs = search_docs("cancellation terms", scope="lumenworks", k=5)
    leaked = [d for d in docs if d["metadata"]["scope"] == "northstar"]
    if not leaked:
        print(f"  ✅ PASS — {len(docs)} result(s), no northstar leakage")
        for d in docs:
            print(f"     scope={d['metadata']['scope']}  src={d['metadata']['source']}")
    else:
        print("  ❌ FAIL — Northstar-scoped doc leaked into lumenworks results!")

    # ── Test 8: create_escalation ────────────────────────────────────────
    print("\n── Test 8: create_escalation() ──")
    esc = create_escalation(
        ticket_id="TKT-501",
        account_id="ACCT-001",
        reason="Platform-wide outage affecting all Northstar shipments",
    )
    if esc and esc["escalation_id"].startswith("ESC-"):
        print(f"  ✅ PASS — Escalation created")
        pp.pprint(esc)
    else:
        print("  ❌ FAIL — Escalation not created properly")

    # ── Test 9: list_orders scoped ───────────────────────────────────────
    print("\n── Test 9: list_orders('ACCT-001') ──")
    orders = list_orders("ACCT-001")
    all_correct = all(o["account_id"] == "ACCT-001" for o in orders)
    if orders and all_correct:
        print(f"  ✅ PASS — {len(orders)} orders, all belong to ACCT-001")
    else:
        print("  ❌ FAIL")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
