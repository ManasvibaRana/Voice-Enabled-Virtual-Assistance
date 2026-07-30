"""Final answer composition.

Raw SQL rows and raw document chunks are never spoken directly - they always
pass through this step, which turns them into 1-2 conversational sentences
suitable for text-to-speech.
"""

import re

from app import config, llm, memory, rag_tool, sql_tool
from app.rag_tool import RagResult
from app.sql_tool import SQLResult

SPOKEN_STYLE = (
    "You are a friendly college voice assistant. Answer in 1 to 2 short "
    "sentences that sound natural when read aloud. Plain text only: no "
    "markdown, no bullet points, no code, no tables, no emoji. If the "
    "result set is large, summarise it instead of listing everything.\n\n"
    "State the answer directly, as a person would say it out loud. Never "
    "mention SQL, databases, queries, results, rows, records, columns, IDs, "
    "documents or passages, and never explain where the answer came from.\n"
    "Never begin with filler such as 'Based on the results', 'According to "
    "the data', 'It appears that' or 'The query shows'.\n"
    "Say 'students', not 'student IDs' or 'unique student IDs'.\n\n"
    "Good: There are 7 students who scored above 80 in AI.\n"
    "Bad: Based on the query results, it appears there are 7 unique student "
    "IDs that scored above 80.\n\n"
    "Round numbers to at most one decimal place. Never read out a long "
    "decimal: say '74.4 percent', not '74.42857142857143 percent'.\n"
    "A mark is meaningless without its total and its subject. If the results "
    "give a score and a max_score, say '30 out of 50' or '60 percent' - never "
    "a bare '30'. If they give a subject name, always say which subject the "
    "mark belongs to. If the results give a raw score with NO max_score and "
    "no percentage column, say the number plainly - do not call it a "
    "percentage, because you do not know what it is out of.\n"
    "Refer to people by name, or as 'they' - never 'he' or 'she'.\n\n"
    "Ties: say two people are tied ONLY when their numbers are exactly equal. "
    "Compare the digits before using the word.\n"
    "  87.7, 84.7, 83.0 -> not tied. 'Nikhil Reddy is highest, then Kabir "
    "Nair, then Ananya Verma.'\n"
    "  100.0, 100.0     -> tied. 'Sneha Pillai and Kabir Nair are tied on 100 "
    "percent.'"
)

# Added to the SYSTEM prompt (not the user turn) when the caller is a student,
# because the results they get are deliberately restricted to themselves. Left
# to itself the model reads its single row as the whole class and answers
# "you have the highest marks", which is both wrong and misleading.
PERSONAL_ONLY_RULES = (
    "\n\nACCESS RULE - the person asking is a student, and the results contain "
    "ONLY their own record, by design. Other students' data is not missing or "
    "incomplete; it is simply not theirs to see.\n\n"
    "Decide which kind of question it is:\n"
    "(a) ABOUT THEMSELVES - answer it completely and normally, including "
    "totals and averages OF THEIR OWN marks or attendance. Their own average "
    "is not a class average.\n"
    "  'what is my attendance?'          -> answer it.\n"
    "  'what is my overall percentage?'  -> answer it.\n"
    "  'what are my marks in AI?'        -> answer it.\n"
    "  'what is my CGPA / total / average?' -> answer it.\n"
    "(b) ABOUT OTHER STUDENTS, or comparing them to others - refuse, then "
    "give their own figure if the results contain one.\n"
    "  'who has the lowest attendance?'  -> 'I can only show you your own "
    "records. Your attendance is 100 percent.'\n"
    "  'how many students are in semester 6?' -> 'I can only see your own "
    "record, so I can't tell you the class size.'\n"
    "  'what rank am I?' / 'am I the top student?' -> refuse: a rank needs "
    "everyone else's marks.\n\n"
    "Only refuse when answering would need somebody else's data. Never state "
    "a rank or position, and never call them top, highest, lowest or best - "
    "one row cannot be ranked, so any rank in the results is an artefact. "
    "Never invent other students."
)

FALLBACK_DB = (
    "Sorry, I couldn't answer that from the data. Try rephrasing the question."
)
FALLBACK_RAG = (
    "I couldn't find anything about that in the college documents I have."
)
FALLBACK_TOOL = "Sorry, I couldn't work that out. Try asking a different way."


def _sanitise_for_speech(text: str) -> str:
    """Strip stray markdown the model may still emit."""
    cleaned = llm.strip_fences(text or "")
    cleaned = re.sub(r"[*_`#>]+", "", cleaned)
    cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _sql_messages(
    question: str, result: SQLResult, memory_key: str, personal_only: bool
) -> list[dict[str, str]]:
    """Shared by the buffered and streaming variants, so they cannot drift."""
    if result["row_count"] == 0:
        table = "(the query ran successfully but matched no records)"
    else:
        table = sql_tool.rows_as_text(result["columns"], result["rows"])

    history = memory.history_text(memory_key)
    return [
        {
            "role": "system",
            "content": SPOKEN_STYLE + (PERSONAL_ONLY_RULES if personal_only else ""),
        },
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{history}\n\n"
                f"User question: {question}\n\n"
                f"Query results:\n{table}\n\n"
                "Answer the question using only these results."
            ),
        },
    ]


def _rag_messages(question: str, result: RagResult, memory_key: str) -> list[dict[str, str]]:
    passages = rag_tool.chunks_as_text(result["chunks"])
    history = memory.history_text(memory_key)
    return [
        {"role": "system", "content": SPOKEN_STYLE},
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{history}\n\n"
                f"User question: {question}\n\n"
                f"Reference passages:\n{passages}\n\n"
                "Answer using ONLY the passages above. If they do not contain "
                "the answer, say that the college documents do not cover it."
            ),
        },
    ]


def _tool_messages(question: str, rendered: str, memory_key: str) -> list[dict[str, str]]:
    history = memory.history_text(memory_key)
    return [
        {"role": "system", "content": SPOKEN_STYLE},
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{history}\n\n"
                f"User question: {question}\n\n"
                f"Information available:\n{rendered}\n\n"
                "Answer the question using only this information."
            ),
        },
    ]


def stream_answer(messages: list[dict[str, str]], fallback: str, max_tokens: int = 200):
    """Yield speech-safe pieces of the answer; return the finished text.

    Each piece is cleaned as it goes so markdown never flashes up in the UI,
    and the accumulated result is sanitised once more at the end so the spoken
    text is identical to the buffered path.
    """
    collected: list[str] = []
    for piece in llm.chat_stream(
        messages, temperature=0.2, max_tokens=max_tokens, model=config.COMPOSE_MODEL
    ):
        collected.append(piece)
        visible = re.sub(r"[*_`#>]+", "", piece)
        if visible:
            yield visible
    return _sanitise_for_speech("".join(collected)) or fallback


def compose_from_sql_stream(
    question: str,
    result: SQLResult,
    memory_key: str = "",
    personal_only: bool = False,
):
    if not result["ok"]:
        return FALLBACK_DB
    return (
        yield from stream_answer(
            _sql_messages(question, result, memory_key, personal_only), FALLBACK_DB
        )
    )


def compose_from_rag_stream(question: str, result: RagResult, memory_key: str = ""):
    if not result["ok"] or not result["chunks"]:
        return FALLBACK_RAG
    return (
        yield from stream_answer(
            _rag_messages(question, result, memory_key), FALLBACK_RAG, max_tokens=220
        )
    )


def compose_from_tool_stream(question: str, rendered: str, memory_key: str = ""):
    if not rendered or not rendered.strip():
        return FALLBACK_TOOL
    fallback = rendered.replace("\n", " ")
    return (
        yield from stream_answer(
            _tool_messages(question, rendered, memory_key), fallback
        )
    )


def compose_from_sql(
    question: str,
    result: SQLResult,
    memory_key: str = "",
    personal_only: bool = False,
) -> str:
    if not result["ok"]:
        return FALLBACK_DB

    if result["row_count"] == 0:
        table = "(the query ran successfully but matched no records)"
    else:
        table = sql_tool.rows_as_text(result["columns"], result["rows"])

    history = memory.history_text(memory_key)
    user_prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"User question: {question}\n\n"
        f"Query results:\n{table}\n\n"
        "Answer the question using only these results."
    )

    reply = llm.chat(
        [
            {
                "role": "system",
                "content": SPOKEN_STYLE
                + (PERSONAL_ONLY_RULES if personal_only else ""),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=200,
        model=config.COMPOSE_MODEL,
    )
    return _sanitise_for_speech(reply) or FALLBACK_DB


def compose_from_tool(question: str, rendered: str, memory_key: str = "") -> str:
    """Compose a spoken answer from a tool that produced its own text.

    Used by the timetable and reminder tools, whose results are already
    structured sentences rather than rows or passages.
    """
    if not rendered or not rendered.strip():
        return FALLBACK_TOOL

    history = memory.history_text(memory_key)
    user_prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"User question: {question}\n\n"
        f"Information available:\n{rendered}\n\n"
        "Answer the question using only this information."
        # This prompt used to add "if it says the user needs to sign in, tell
        # them that". Mentioning the phrase primed the model to produce it even
        # for a signed-in student with a perfectly good timetable. The
        # unauthenticated case never reaches here - main.py returns the tool's
        # error directly - so the clause was both wrong and unnecessary.
    )
    reply = llm.chat(
        [
            {"role": "system", "content": SPOKEN_STYLE},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=200,
        model=config.COMPOSE_MODEL,
    )
    return _sanitise_for_speech(reply) or rendered.replace("\n", " ")


def compose_from_rag(question: str, result: RagResult, memory_key: str = "") -> str:
    if not result["ok"] or not result["chunks"]:
        return FALLBACK_RAG

    passages = rag_tool.chunks_as_text(result["chunks"])
    history = memory.history_text(memory_key)
    user_prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"User question: {question}\n\n"
        f"Reference passages:\n{passages}\n\n"
        "Answer using ONLY the passages above. If they do not contain the "
        "answer, say that the college documents do not cover it."
    )

    reply = llm.chat(
        [
            {"role": "system", "content": SPOKEN_STYLE},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=220,
        model=config.COMPOSE_MODEL,
    )
    return _sanitise_for_speech(reply) or FALLBACK_RAG

