"""
Phase 2 — LangGraph ReAct agent with tool-calling + HITL for escalations.

Wraps the Phase 1 tool functions into LangChain @tool definitions and
builds a LangGraph ReAct agent graph with SqliteSaver checkpointer.

Design decisions:
  - Each @tool wrapper injects account_id from the RunnableConfig so
    the LLM can never override the access-control scope.
  - tool_create_escalation uses langgraph.types.interrupt() to pause
    execution and surface the proposed escalation to the user. The
    Streamlit UI renders Approve/Reject buttons; on resume the tool
    either writes the escalation or returns a rejection message.
    All other tools run without interruption.
  - The graph is compiled once with a checkpointer; each invocation
    passes a thread_id via config so LangGraph auto-loads/saves the
    right conversation history.
  - System prompt tells the agent its role, constraints, and how to
    use the tools. It explicitly instructs: never reveal other
    customers' data, always use the injected account_id.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables first so LangSmith tracing is active at import/definition time
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

import sqlite3
import groq
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from tools import (
    calc_sla_status,
    create_escalation,
    get_account,
    get_order,
    get_ticket,
    list_orders,
    list_tickets,
    search_docs,
)

CHAT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "chat_memory.db"
CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════
# TOOL WRAPPERS — inject account_id from config, not from the LLM
# ═════════════════════════════════════════════════════════════════════════

def _get_account_id(config: RunnableConfig) -> str:
    """Extract account_id from the agent's runtime config."""
    return config["configurable"]["account_id"]


def _get_scope(account_id: str) -> str:
    """Map account_id to document-search scope tag."""
    mapping = {
        "ACCT-001": "northstar",
        "ACCT-002": "lumenworks",
    }
    return mapping.get(account_id, "all")


@tool
def tool_search_docs(query: str, version: str = None, k: int = 3,
                     config: RunnableConfig = None) -> list[dict] | str:
    """Search ParcelPilot's policy documents and knowledge base.
    Use this to look up support policies, cancellation rules, SLA terms,
    service credit procedures, known issues, and contract-specific terms.
    Args:
        query: Natural language search query.
        version: Optional version filter (e.g. 'v3_current', 'v2_deprecated').
        k: Number of results to return (default 3).
    """
    scope = _get_scope(_get_account_id(config))
    res = search_docs(query=query, scope=scope, version=version, k=k)
    if not res:
        return "No results found."
    return res


@tool
def tool_get_account(config: RunnableConfig = None) -> dict | str:
    """Look up the current customer's account details (plan, CSM, status, etc.).
    No arguments needed — automatically scoped to the logged-in customer.
    """
    res = get_account(_get_account_id(config))
    if not res:
        return "Account not found."
    return res


@tool
def tool_get_order(order_id: str, config: RunnableConfig = None) -> dict | str:
    """Look up a specific order by order_id.
    Returns the order only if it belongs to the current customer.
    Args:
        order_id: The order ID to look up (e.g. 'ORD-1001').
    """
    res = get_order(order_id, _get_account_id(config))
    if not res:
        return "Order not found."
    return res


@tool
def tool_get_ticket(ticket_id: str, config: RunnableConfig = None) -> dict | str:
    """Look up a specific support ticket by ticket_id.
    Returns the ticket only if it belongs to the current customer.
    Args:
        ticket_id: The ticket ID to look up (e.g. 'TKT-501').
    """
    res = get_ticket(ticket_id, _get_account_id(config))
    if not res:
        return "Ticket not found."
    return res


@tool
def tool_list_orders(config: RunnableConfig = None) -> list[dict] | str:
    """List all orders belonging to the current customer.
    No arguments needed — automatically scoped to the logged-in customer.
    """
    res = list_orders(_get_account_id(config))
    if not res:
        return "No orders found."
    return res


@tool
def tool_list_tickets(config: RunnableConfig = None) -> list[dict] | str:
    """List all support tickets belonging to the current customer.
    No arguments needed — automatically scoped to the logged-in customer.
    """
    res = list_tickets(_get_account_id(config))
    if not res:
        return "No tickets found."
    return res


@tool
def tool_calc_sla_status(order_id: str,
                         snapshot_time: str = "2026-08-16 11:00",
                         config: RunnableConfig = None) -> dict | str:
    """Calculate SLA/pickup timing status for an order.
    Returns factual timing data: on_time/late, hours late, carrier/customer fault.
    Does NOT decide on service credits — combine with policy docs for that.
    Args:
        order_id: The order to check (e.g. 'ORD-2002').
        snapshot_time: The reference 'now' time (default: dataset snapshot).
    """
    res = calc_sla_status(order_id, _get_account_id(config),
                           snapshot_time=snapshot_time)
    if not res:
        return "Order SLA status could not be calculated."
    return res


@tool
def tool_create_escalation(ticket_id: str, reason: str,
                           config: RunnableConfig = None) -> dict:
    """Create an escalation record for a support ticket.
    Use this when a ticket needs to be escalated to a human agent.
    Args:
        ticket_id: The ticket to escalate (e.g. 'TKT-501').
        reason: Brief description of why escalation is needed.
    """
    from datetime import datetime
    now_str = datetime.now().isoformat()
    print(f"[{now_str}] [SERVER LOG] tool_create_escalation called: ticket_id={ticket_id}, reason={reason}")
    return create_escalation(ticket_id, _get_account_id(config), reason)


# ═════════════════════════════════════════════════════════════════════════
# AGENT GRAPH
# ═════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are ParcelPilot Support Assistant, a customer-facing AI chatbot for ParcelPilot, a B2B logistics platform.

ROLE:
- Help customers with order inquiries, ticket status, cancellation requests, SLA questions, and service credit eligibility.
- You are professional, concise, and helpful.

ACCESS CONTROL (CRITICAL):
- You can ONLY access data belonging to the currently logged-in customer.
- NEVER reveal, reference, or discuss data from other customers.
- The account_id is injected automatically — you cannot override it.

HOW TO ANSWER:
1. For factual questions about orders/tickets: use the lookup and list tools.
2. For policy questions (cancellation rules, SLA terms, credits): use tool_search_docs to find the relevant policy, then cite what the policy says.
3. For SLA/timing questions: use tool_calc_sla_status to get the facts, then use tool_search_docs to find the applicable policy/SOP to determine eligibility.
4. For escalations: call tool_create_escalation with the ticket_id and reason. The system will automatically pause and ask the customer for approval before executing — you do NOT need to ask for confirmation yourself.
5. If you don't have enough information, say so — don't make things up.

STRICT SAFEGUARDS (CRITICAL):
- NEVER fabricate, mock, or hallucinate tool execution responses, escalation IDs (e.g., ESC-XXXXXXXX), or confirmation statuses.
- NEVER tell the user that a ticket has been escalated, queued, or submitted based on your own generation. You are strictly forbidden from claiming success in your text responses until the tool has actually run and returned its output containing the real escalation ID and status.
- When performing an action (like an escalation), do not output introductory text claiming it is being done or has been done. Simply invoke the tool. Only after the tool has run and returned the result to your context should you report the success details (ID, status) to the user.

IMPORTANT:
- The dataset snapshot time is 2026-08-16 11:00 Asia/Kolkata.
- Historical ticket resolutions may be incorrect — treat them as context, not policy authority.
- Always prefer the CURRENT version of policies (v3, v4) over deprecated ones unless the customer specifically asks about old policies.
"""

ALL_TOOLS = [
    tool_search_docs,
    tool_get_account,
    tool_get_order,
    tool_get_ticket,
    tool_list_orders,
    tool_list_tickets,
    tool_calc_sla_status,
    tool_create_escalation,
]


def build_agent():
    """Build and return (graph, checkpointer) — caller manages checkpointer lifecycle."""
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_retries=3,
    )

    # Use a raw sqlite3 connection with check_same_thread=False so the
    # checkpointer survives across Streamlit reruns without needing a
    # context manager.
    conn = sqlite3.connect(str(CHAT_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "tool_create_escalation": {"allowed_decisions": ["approve", "reject"]}
                }
            )
        ],
        checkpointer=checkpointer,
    )

    return graph, checkpointer
