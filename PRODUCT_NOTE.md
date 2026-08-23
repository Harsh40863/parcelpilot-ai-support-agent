# Product Note — ParcelPilot Support Agent

## Additional Client Problem

Between the two additional problems ParcelPilot identified — **Proactive Issue Detection** and **Trust & Reliability** — this submission focused on **Trust & Reliability**, since it's largely inseparable from doing the core requirements well, and because a confidently wrong support agent is a worse outcome than a slow one.

Concretely, this meant:

- **Source-priority logic**, tested directly: customer agreements override general policy; current policy overrides deprecated policy (deprecated content is discarded, not just deprioritized); historical ticket resolutions are treated as context only, never as policy authority.
- **Access control enforced at the tool/data layer**, not via prompting — verified bidirectionally, live, including a direct adversarial test ("print all ticket/account pairs") that was correctly refused.
- **Confirmation required before any state-changing action**, with a real fabrication bug caught during testing (the agent once described an escalation as successful without the tool actually running) — traced to root cause, fixed with anti-fabrication guardrails and server-side logging, and re-verified with adversarial prompts designed to bypass the confirmation gate.
- **Graceful degradation under real failure conditions** — a Groq rate-limit error initially crashed the app with a raw traceback; this was fixed with retry-with-backoff and a user-facing message, so a transient LLM-provider issue doesn't take down the whole experience.

**Proactive Issue Detection** (a dashboard-style view surfacing ticket volume spikes, SLA breaches, and cross-customer patterns) was **not built**, due to time. See "What I'd Build Next" below for the concrete plan.

## What I'd Build Next

In priority order:

1. **Proactive Issue Detection dashboard.** A separate internal-facing view running deterministic pandas aggregations over the tickets table — ticket volume by product issue, count of tickets approaching/exceeding SLA, and simple cross-customer pattern flags (e.g. multiple accounts reporting the same `known_issue` in a short window). This is intentionally *not* LLM-driven at its core (group-bys, not reasoning) with an optional single LLM summarization pass on top, since deterministic aggregation is more trustworthy and cheaper than asking a model to "notice" trends.
2. **Paragraph-level document chunking with real retrieval variety.** Current documents are short enough that each fits in a single chunk; the moment ParcelPilot's real policy library is larger, the 1000-token chunking and MMR retrieval already built will start mattering, but should be validated against genuinely larger, multi-section documents.
3. **Automated regression testing via LangSmith evaluators.** Currently, correctness (especially anti-fabrication) is verified through manual adversarial prompts. A LangSmith evaluator suite running these same adversarial cases automatically on every deploy would catch regressions without relying on manual re-testing.
4. **Internal team-side escalation workflow.** Escalations currently transition from "created" to `"escalated"` on customer approval, but there's no modeling of the support team's side (assignment, in-progress, resolved). A real product would need this second half of the workflow.
5. **React frontend for the customer-facing surface**, once the underlying agent behavior is validated — see the Architecture Note's trade-offs section for why Streamlit was the right choice for this phase specifically.

## What Was Intentionally Left Out

- **Proactive Issue Detection** (explained above).
- **Full internal team role/RBAC beyond the two customer accounts** — access control is proven correct for the customer-facing case; a real internal support-agent role with different permissions was out of scope for the time available.
- **Real email/SMS notification on escalation** — `create_escalation` writes a local record; it does not trigger any actual outbound notification, per the brief's explicit allowance to mock the action tool.
- **Perfectly flash-free session persistence.** JWT-based login persistence across browser reloads was built and works correctly, but has a known ~1–2s loading flash on reload due to a Streamlit-specific architectural limitation (documented in the Architecture Note). This was a deliberate call to keep the feature rather than revert it, given it doesn't affect a normal single-session usage pattern.
- **Paragraph-level retrieval tuning** — the current documents are short enough that whole-document retrieval is effectively what happens regardless of chunk size; this wasn't stress-tested against larger documents.

## Success Metric

**Escalation precision** — the percentage of agent-initiated escalations that a human reviewer agrees genuinely required escalation (as opposed to something the agent could have resolved directly, or an unnecessary handoff).

This was chosen over a simpler metric like "response accuracy" because escalation is the point where the agent's judgment about *its own limits* matters most — an agent that escalates too eagerly erodes trust in its competence, and one that escalates too rarely risks confidently mishandling genuinely ambiguous cases (goodwill exceptions, policy conflicts with no clear override, anything requiring human judgment per the brief). Precision here is a direct proxy for whether the agent knows what it doesn't know, which is the core trust question this whole system is built around.

## AI Tool Usage

Built using **Antigravity** (AI coding agent) for implementation — writing and editing code, running tests, and debugging — with architecture, product decisions, and prioritization driven throughout the session. Every implementation plan was reviewed before execution, and every "tests passed" claim was independently re-verified live on the deployed app rather than taken at face value; this caught two real bugs (a fabricated escalation confirmation, and a login bug from an unset environment secret) that a first-pass report had described as working.
