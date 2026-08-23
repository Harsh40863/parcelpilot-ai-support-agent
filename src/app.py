"""
Phase 3 — Streamlit chat app with streaming + human-in-the-loop.

Ties together: user auth (MongoDB), LangGraph agent (Groq + tools),
SqliteSaver checkpointer (per-thread message history), sidebar thread
switcher, streamed responses, and Approve/Reject buttons for escalations.

Run with:  uv run streamlit run src/app.py
"""

import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load env variables first so LangSmith tracing is active at import/definition time
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, AIMessageChunk
from langgraph.types import Command

# ── Must be first Streamlit call ─────────────────────────────────────────
st.set_page_config(page_title="ParcelPilot Support", page_icon="📦", layout="wide")

# Imports from our own modules (same src/ directory)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import jwt
import datetime
import extra_streamlit_components as stx

from agent import build_agent  # noqa: E402
from db_users import (  # noqa: E402
    add_thread_to_user,
    authenticate_user,
    get_user_threads,
    delete_thread_from_user,
    delete_thread_checkpoints,
    blacklist_token,
    is_token_blacklisted,
)


# ═════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═════════════════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "authenticated": False,
        "email": None,
        "account_id": None,
        "active_thread_id": None,
        "stop_requested": False,
        "partial_text": "",
        "partial_tools": [],
        "is_streaming": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


def get_cookie_manager():
    if "cookie_manager_instance" not in st.session_state:
        st.session_state.cookie_manager_instance = stx.CookieManager(key="cookie_manager_widget")
    return st.session_state.cookie_manager_instance


def check_jwt_cookie():
    cookie_manager = get_cookie_manager()
    cookies = cookie_manager.get_all()
    
    if cookies is None:
        st.write("Initializing session...")
        st.stop()
        
    token = cookies.get("parcelpilot_jwt")
    if token:
        if not st.session_state.get("authenticated"):
            try:
                secret_key = os.environ.get("JWT_SECRET_KEY")
                if not secret_key:
                    st.error("JWT_SECRET_KEY is not configured.")
                    return
                
                payload = jwt.decode(token, secret_key, algorithms=["HS256"])
                
                if is_token_blacklisted(token):
                    cookie_manager.delete("parcelpilot_jwt", key="delete_blacklisted_jwt")
                    st.rerun()
                    return
                
                st.session_state.authenticated = True
                st.session_state.email = payload["email"]
                st.session_state.account_id = payload["account_id"]
                st.rerun()
                
            except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
                cookie_manager.delete("parcelpilot_jwt", key="delete_expired_jwt")
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════
# AGENT SINGLETON (built once per session)
# ═════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_agent():
    """Build the agent graph + checkpointer once, reuse across reruns."""
    return build_agent()


# ═════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════
def _make_config(thread_id: str, account_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "account_id": account_id,
        }
    }


def load_thread_messages(graph, thread_id: str, account_id: str) -> list:
    """Load past messages for a thread from the LangGraph checkpointer."""
    config = _make_config(thread_id, account_id)
    try:
        state = graph.get_state(config)
        if state and state.values:
            return state.values.get("messages", [])
    except Exception:
        pass
    return []


def get_pending_interrupts(graph, thread_id: str, account_id: str):
    """Check if the graph is paused waiting for human input."""
    config = _make_config(thread_id, account_id)
    try:
        state = graph.get_state(config)
        if state and state.tasks:
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    return task.interrupts
    except Exception:
        pass
    return []


def extract_tool_names_from_msg(msg: AIMessage) -> list[str]:
    """Extract tool names from an AIMessage's tool_calls."""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        return [tc.get("name", "unknown") for tc in msg.tool_calls]
    return []


# ═════════════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ═════════════════════════════════════════════════════════════════════════
def render_login():
    st.title("📦 ParcelPilot Support")
    st.subheader("Customer Login")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", use_container_width=True)

    if submitted:
        email_clean = email.strip().lower()
        password_clean = password.strip()
        if not email_clean or not password_clean:
            st.error("Please enter both email and password.")
            return

        user = authenticate_user(email_clean, password_clean)
        if user:
            secret_key = os.environ.get("JWT_SECRET_KEY")
            if not secret_key:
                st.error("JWT_SECRET_KEY is not configured.")
                return
            
            payload = {
                "email": user["email"],
                "account_id": user["account_id"],
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }
            token = jwt.encode(payload, secret_key, algorithm="HS256")
            
            cookie_manager = get_cookie_manager()
            expires = datetime.datetime.now() + datetime.timedelta(hours=24)
            cookie_manager.set(
                "parcelpilot_jwt",
                token,
                key="set_login_jwt",
                expires_at=expires
            )
            
            st.session_state.authenticated = True
            st.session_state.email = user["email"]
            st.session_state.account_id = user["account_id"]
            st.rerun()
        else:
            st.error("Invalid email or password.")


# ═════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.email}")
        st.markdown(f"**Account:** {st.session_state.account_id}")
        st.divider()

        # ── New conversation button ──────────────────────────────────────
        if st.button("➕ New Conversation", use_container_width=True):
            new_id = str(uuid.uuid4())
            add_thread_to_user(st.session_state.email, new_id)
            st.session_state.active_thread_id = new_id
            st.rerun()

        st.divider()

        # ── Thread list ──────────────────────────────────────────────────
        st.markdown("**Conversations**")
        threads = get_user_threads(st.session_state.email)

        if not threads:
            st.caption("No conversations yet. Click '➕ New Conversation' to start.")
        else:
            for t in reversed(threads):  # newest first
                tid = t["thread_id"]
                label_text = t.get("label", "New conversation")
                if not label_text:
                    label_text = "New conversation"
                
                display_label = label_text
                if len(display_label) > 28:
                    display_label = display_label[:28].rstrip() + "..."
                
                is_active = tid == st.session_state.active_thread_id
                button_type = "primary" if is_active else "secondary"
                
                if st.session_state.get("deleting_thread_id") == tid:
                    st.markdown(f"**Delete conversation?**\n*{display_label}*")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes", key=f"confirm_yes_{tid}", type="primary", use_container_width=True):
                            delete_thread_from_user(st.session_state.email, tid)
                            delete_thread_checkpoints(tid)
                            
                            if st.session_state.active_thread_id == tid:
                                remaining_threads = [x for x in threads if x["thread_id"] != tid]
                                if remaining_threads:
                                    st.session_state.active_thread_id = remaining_threads[-1]["thread_id"]
                                else:
                                    st.session_state.active_thread_id = None
                            
                            st.session_state.deleting_thread_id = None
                            st.rerun()
                    with col_no:
                        if st.button("Cancel", key=f"confirm_no_{tid}", use_container_width=True):
                            st.session_state.deleting_thread_id = None
                            st.rerun()
                else:
                    col_btn, col_del = st.columns([5, 1])
                    with col_btn:
                        if st.button(f"💬 {display_label}", key=f"thread_{tid}",
                                     use_container_width=True, type=button_type):
                            st.session_state.active_thread_id = tid
                            st.rerun()
                    with col_del:
                        if st.button("🗑", key=f"del_{tid}", use_container_width=True):
                            st.session_state.deleting_thread_id = tid
                            st.rerun()

        st.divider()

        # ── Logout ───────────────────────────────────────────────────────
        if st.button("🚪 Logout", use_container_width=True):
            cookie_manager = get_cookie_manager()
            token = cookie_manager.get("parcelpilot_jwt")
            if token:
                blacklist_token(token)
            cookie_manager.delete("parcelpilot_jwt", key="logout_delete_jwt")
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════
# DISPLAY PAST MESSAGES (from checkpointer state)
# ═════════════════════════════════════════════════════════════════════════
def display_history(messages: list):
    """Render all past messages from the checkpointer."""
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)

        elif isinstance(msg, AIMessage) and msg.content:
            with st.chat_message("assistant"):
                st.markdown(msg.content)
                tool_names = extract_tool_names_from_msg(msg)
                if tool_names:
                    names_str = ", ".join(tool_names)
                    with st.expander(f"🔧 Used: {names_str}"):
                        for tc in msg.tool_calls:
                            st.code(
                                f"{tc['name']}({tc.get('args', {})})",
                                language="python",
                            )

        # ToolMessages are shown in the AI expander, skip standalone render


# ═════════════════════════════════════════════════════════════════════════
# STREAM A NEW AGENT RESPONSE
# ═════════════════════════════════════════════════════════════════════════
def stream_agent_response(graph, input_data, config: dict):
    """
    Stream the agent's response, showing tool calls live and collecting
    the final text. Returns (final_text, tools_used).
    """
    full_text = ""
    tools_used = []
    text_placeholder = st.empty()
    status_placeholder = st.empty()
    stop_button_placeholder = st.empty()

    st.session_state.is_streaming = True
    st.session_state.partial_text = ""
    st.session_state.stop_requested = False

    def stop_callback():
        st.session_state.stop_requested = True

    # Render Stop button while streaming
    stop_button_placeholder.button("⏹ Stop", on_click=stop_callback, key="stop_stream_btn")

    try:
        for msg, metadata in graph.stream(input_data, config=config, stream_mode="messages"):
            if st.session_state.stop_requested:
                break

            if isinstance(msg, AIMessageChunk):
                if msg.content:
                    full_text += msg.content
                    text_placeholder.markdown(full_text)
                    st.session_state.partial_text = full_text
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.get("name")
                        if name and name not in tools_used:
                            tools_used.append(name)
                            status_placeholder.caption(f"🔧 Calling {name}...")
            elif isinstance(msg, ToolMessage):
                status_placeholder.empty()
    except Exception as e:
        status_placeholder.empty()
        err_msg = str(e).lower()
        if "rate_limit" in err_msg or "rate limit" in err_msg or "429" in err_msg:
            st.error("⚠️ The AI service is busy, please wait a moment and try again.")
        else:
            st.error(f"⚠️ Error: {str(e)}")

    status_placeholder.empty()
    stop_button_placeholder.empty()
    st.session_state.is_streaming = False
    return full_text, tools_used


# ═════════════════════════════════════════════════════════════════════════
# HITL APPROVAL UI
# ═════════════════════════════════════════════════════════════════════════
def render_hitl_approval(graph, thread_id: str, account_id: str,
                         interrupts: list):
    """Render Approve/Reject buttons for pending escalation approvals."""
    config = _make_config(thread_id, account_id)

    for intr in interrupts:
        data = intr.value if hasattr(intr, "value") else intr
        if isinstance(data, dict):
            if "action_requests" in data and len(data["action_requests"]) > 0:
                action = data["action_requests"][0]
                ticket_id = action["args"].get("ticket_id", "unknown")
                reason = action["args"].get("reason", "")
                message = action.get("description", "Escalation pending approval")
            else:
                ticket_id = data.get("ticket_id", "unknown")
                reason = data.get("reason", "")
                message = data.get("message", "Escalation pending approval")
        else:
            ticket_id = "unknown"
            reason = str(data)
            message = str(data)

        with st.chat_message("assistant"):
            st.warning(f"⏸️ **Escalation Approval Required**\n\n{message}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve", key=f"approve_{ticket_id}",
                             use_container_width=True, type="primary"):
                    # Resume the graph with approval
                    with st.chat_message("assistant"):
                        with st.spinner("Processing escalation..."):
                            result_text, result_tools = stream_agent_response(
                                graph,
                                Command(resume={"decisions": [{"type": "approve"}]}),
                                config,
                            )
                        if result_tools:
                            with st.expander(f"🔧 Used: {', '.join(result_tools)}"):
                                for t in result_tools:
                                    st.text(t)
                    st.rerun()

            with col2:
                if st.button("❌ Reject", key=f"reject_{ticket_id}",
                             use_container_width=True):
                    # Resume the graph with rejection
                    with st.chat_message("assistant"):
                        with st.spinner("Processing..."):
                            result_text, result_tools = stream_agent_response(
                                graph,
                                Command(resume={"decisions": [{"type": "reject"}]}),
                                config,
                            )
                    st.rerun()


# ═════════════════════════════════════════════════════════════════════════
# MAIN CHAT PANEL
# ═════════════════════════════════════════════════════════════════════════
def render_chat():
    graph, _checkpointer = get_agent()
    thread_id = st.session_state.active_thread_id
    account_id = st.session_state.account_id

    if not thread_id:
        st.info("Select a conversation from the sidebar or start a new one.")
        return

    st.caption(f"Thread: `{thread_id[:16]}…`")

    # ── Check if the user requested to stop the current stream ───────────
    if st.session_state.stop_requested:
        partial_text = st.session_state.partial_text
        if partial_text:
            config = _make_config(thread_id, account_id)
            stopped_msg = AIMessage(
                content=f"{partial_text}\n\n*⏹ Stopped by user*"
            )
            graph.update_state(config, {"messages": [stopped_msg]})
        st.session_state.stop_requested = False
        st.session_state.partial_text = ""
        st.session_state.partial_tools = []
        st.session_state.is_streaming = False
        st.rerun()

    # ── Load and display past messages ───────────────────────────────────
    messages = load_thread_messages(graph, thread_id, account_id)
    is_new_thread = len(messages) == 0
    display_history(messages)

    # ── Check for pending HITL interrupts ────────────────────────────────
    interrupts = get_pending_interrupts(graph, thread_id, account_id)
    if interrupts:
        render_hitl_approval(graph, thread_id, account_id, interrupts)
        return  # Don't show chat_input while waiting for approval

    # ── Chat input ───────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask ParcelPilot Support…"):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        config = _make_config(thread_id, account_id)

        if is_new_thread:
            label = prompt.strip()
            if len(label) > 45:
                label = label[:45] + "..."
            add_thread_to_user(st.session_state.email, thread_id, label)

        # Stream the agent response
        with st.chat_message("assistant"):
            full_text, tools_used = stream_agent_response(
                graph,
                {"messages": [HumanMessage(content=prompt)]},
                config,
            )

            if tools_used:
                names_str = ", ".join(tools_used)
                with st.expander(f"🔧 Used: {names_str}"):
                    for t in tools_used:
                        st.text(t)

        # Check if the agent hit an interrupt or if we need to refresh the sidebar label
        new_interrupts = get_pending_interrupts(graph, thread_id, account_id)
        if new_interrupts or is_new_thread:
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    check_jwt_cookie()
    
    if not st.session_state.authenticated:
        render_login()
    else:
        render_sidebar()
        render_chat()


if __name__ == "__main__":
    main()
