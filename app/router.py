"""Tool routing: one LLM call that picks which skill answers the question."""

from typing import TypedDict

from app import config, identity, llm, memory

VALID_TOOLS: set[str] = {
    "query_database",
    "search_documents",
    "timetable",
    "reminders",
    "smalltalk",
}

SYSTEM_PROMPT = """You are the router for a college voice assistant. You choose \
exactly one tool to answer the user's question.

Tools:
1. "query_database" - a SQLite database of college records. Anything countable,
   filterable or comparable: numbers of students, marks and scores, averages,
   attendance percentages, per-subject or per-semester breakdowns, rankings
   ("who has the highest/lowest ...", "top three students"), and a signed-in
   student's own marks or attendance ("what is my attendance?").
   Tables: students(id, name, roll_no, semester, branch),
           subjects(id, name, semester, credits),
           marks(id, student_id, subject_id, exam_type, score, max_score),
           attendance(id, student_id, subject_id, date, present)

2. "search_documents" - college policy and FAQ documents. Rules, definitions,
   explanations and procedures: the attendance policy, grading and exam rules,
   re-evaluation, fees, library, hostel, "how do I ..." questions.

3. "timetable" - the class schedule. "What is my next class?", "what are my
   classes today?", "what is my timetable on Friday?", "when is Compiler
   Design this week?", "which room am I in?".

4. "reminders" - the user's personal to-do reminders. "Remind me to ...",
   "what are my reminders?", "mark ... as done". Only for reminders the user
   is creating or reading, never for college deadlines written in policy
   documents.

5. "smalltalk" - greetings ("hi", "hello", "thanks"), questions about the
   assistant itself ("what can you do?", "who are you?"), anything clearly
   outside college records, policies, timetables and reminders (weather,
   sport, general knowledge), AND questions about what you just said.
   A question that can be answered by re-reading your previous answer needs
   no lookup at all:
     "is that a percentage or a CGPA?"   -> smalltalk
     "sorry, what was that number?"      -> smalltalk
     "did you say Computer Networks?"    -> smalltalk
     "what do you mean?"                 -> smalltalk
   Only pick a data tool when NEW information has to be looked up.

Rules:
- Counting or aggregating student records -> query_database.
- A written rule, policy or procedure -> search_documents.
- A class time, day or room -> timetable.
- Creating or reading the user's own reminders -> reminders.
- Greeting, capability question, or off-topic -> smalltalk.
- "My marks" and "my attendance" are query_database, not timetable.
- Use the conversation history to resolve follow-ups such as "and for
  semester 4?" - a follow-up usually needs the SAME tool as the previous turn.
- Reply with ONLY a JSON object, no prose and no markdown fences:
{"tool": "<one of the five names>", "reason": "<one short sentence>"}"""


class RouteDecision(TypedDict):
    tool: str
    reason: str


DEFAULT_DECISION: RouteDecision = {
    "tool": "search_documents",
    "reason": "Router fell back to document search because the routing response "
    "could not be parsed.",
}


def route(question: str, principal=None, memory_key: str = "") -> RouteDecision:
    """Classify the question. Never raises; defaults to search_documents."""
    history = memory.history_text(memory_key)
    user_prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"{identity.context_line(principal)}\n\n"
        f"New user question: {question}\n\n"
        "Which tool should handle it? JSON only."
    )

    result = llm.chat_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        fallback=dict(DEFAULT_DECISION),
        model=config.ROUTER_MODEL,
    )

    tool = str(result.get("tool", "")).strip()
    reason = str(result.get("reason", "")).strip()

    if tool not in VALID_TOOLS:
        return dict(DEFAULT_DECISION)  # type: ignore[return-value]

    return {"tool": tool, "reason": reason or "No reason given by the router."}
