# ParcelPilot Support Agent

A tool-calling AI support agent for ParcelPilot, a B2B logistics platform. Built for the CalQuity AI Engineer take-home assessment.

**Live app:** https://harsh40863-parcelpilot-ai-support-agent-srcapp-zm5sev.streamlit.app

**Test credentials:**

| Email | Password | Account |
|---|---|---|
| `northstar@test.com` | `password123` | ACCT-001 (Northstar Logistics) |
| `lumenworks@test.com` | `password123` | ACCT-002 (LumenWorks) |

See [`ARCHITECTURE_NOTE.md`](./ARCHITECTURE_NOTE.md), [`PRODUCT_NOTE.md`](./PRODUCT_NOTE.md), and [`AI_TOOL_USAGE.md`](./AI_TOOL_USAGE.md) for full write-ups.

## What This Is

A customer-facing chatbot that answers ParcelPilot support questions using policies, agreements, and account/order/ticket data — with three tool categories (document search, structured data lookup/calculation, and a confirm-before-execute action), enforced per-account access control, and source-conflict handling (current policy overrides deprecated; customer agreements override general policy).

## Stack

- **Agent:** LangChain `create_agent` + `HumanInTheLoopMiddleware`, LangGraph, Groq (`openai/gpt-oss-20b`)
- **Retrieval:** FAISS + HuggingFace local embeddings (`sentence-transformers/all-MiniLM-L6-v2`), MMR retrieval
- **Structured data:** SQLite (accounts/orders/tickets)
- **Conversation memory:** LangGraph SqliteSaver checkpointer
- **Auth & thread metadata:** MongoDB, JWT-based session persistence
- **UI:** Streamlit
- **Observability:** LangSmith tracing

## Local Setup

1. **Clone and install dependencies**
   ```bash
   git clone https://github.com/Harsh40863/parcelpilot-ai-support-agent.git
   cd parcelpilot-ai-support-agent
   pip install -r requirements.txt --break-system-packages
   ```

2. **Create a `.env` file** in the project root with:
   ```
   MONGO_URI=your_mongodb_connection_string
   GROQ_API_KEY=your_groq_api_key
   JWT_SECRET_KEY=any_random_secure_string
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your_langsmith_api_key
   LANGSMITH_PROJECT=parcelpilot-assessment
   ```

3. **Data is pre-ingested and committed** (`data/processed/parcelpilot.db`, `data/processed/faiss_index/`). To rebuild from the raw source pack instead:
   ```bash
   python src/ingest_data.py
   python src/ingest_docs.py
   ```

4. **Run the app**
   ```bash
   streamlit run src/app.py
   ```

5. Open `http://localhost:8501` and log in with the test credentials above.

## Project Structure

```
src/
  ingest_data.py    # xlsx -> SQLite
  ingest_docs.py    # PDFs -> FAISS (chunk + embed)
  tools.py          # 8 tools: document search, structured lookup/calc, action
  db_users.py       # Mongo auth, thread metadata, JWT blacklist
  agent.py          # LangGraph agent, tool wrapping, access control, HITL middleware
  app.py            # Streamlit UI: login, chat, sidebar, escalation approval
data/
  raw/              # source data pack (policies, agreements, xlsx)
  processed/        # generated SQLite db, FAISS index, chat checkpointer, escalations.json
```

## Example Questions to Try

- "Can Northstar cancel ORD-1001 without a cancellation fee? Explain why."
- "A pickup is three hours late because of carrier fault. Should I get a service credit?"
- "Please escalate ticket TKT-501, shipment creation is failing." (triggers a confirmation step)