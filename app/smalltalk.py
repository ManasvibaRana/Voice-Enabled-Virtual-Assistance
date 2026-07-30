"""Greetings, capability questions and anything outside the assistant's scope.

Before this route existed, "hello" and "what can you do?" were forced into
document search, which retrieved irrelevant policy text and answered from it.
"""

from app import config, identity, llm, memory

CAPABILITIES = """You are a college voice assistant. You can:
- answer questions about student records: counts, marks, averages, attendance,
  rankings such as the top students in a semester
- look things up in the college policy documents: attendance rules, grading and
  exam rules, re-evaluation, fees, library and hostel rules
- tell a signed-in student their own marks, attendance and eligibility
- read out a class timetable: the next class, or a whole day
- keep simple reminders and read them back"""

SYSTEM_PROMPT = """You are a friendly college voice assistant handling a \
greeting, a question about what you can do, or something outside your scope.

{capabilities}

{identity}

Conversation so far:
{history}

Reply in 1 to 2 short sentences that sound natural read aloud. Plain text only:
no markdown, no bullet points, no lists, no emoji.

If they are asking about something you just said - "was that a percentage or a
CGPA?", "what was that number again?", "did you say Friday?" - answer from the
conversation above and nothing else. Repeat the figure exactly as you gave it;
never recalculate it and never quote a different number.
A percentage is out of 100 and a CGPA is out of 10: if a figure was given as a
percentage, say so plainly rather than agreeing it was a CGPA.

If the question is outside everything listed above, say plainly that you only
help with college records, policies, timetables and reminders. Never invent
facts about students, marks or policies - you have not looked anything up."""

FALLBACK = (
    "I can help with student records, college policies, your timetable and "
    "reminders. What would you like to know?"
)


def answer(question: str, principal=None, memory_key: str = "") -> str:
    reply = llm.chat(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    capabilities=CAPABILITIES,
                    identity=identity.context_line(principal),
                    history=memory.history_text(memory_key),
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.4,
        max_tokens=140,
        model=config.COMPOSE_MODEL,
    )
    from app.compose import _sanitise_for_speech

    return _sanitise_for_speech(reply) or FALLBACK
