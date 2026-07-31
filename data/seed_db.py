"""Create and populate ./data/app.db with reproducible sample college data.

Run:  python data/seed_db.py          (recreates the database from scratch)

A fixed random seed means every machine gets byte-identical data, so the
example questions in the README always return the same answers.
"""

import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "app.db"

RANDOM_SEED = 42
N_STUDENTS = 30
N_ATTENDANCE = 200
N_MARKS = 150

SCHEMA = """
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS library_loans;
DROP TABLE IF EXISTS reminders;
DROP TABLE IF EXISTS timetable;
DROP TABLE IF EXISTS attendance;
DROP TABLE IF EXISTS marks;
DROP TABLE IF EXISTS subjects;
DROP TABLE IF EXISTS students;

CREATE TABLE students (
    id       INTEGER PRIMARY KEY,
    name     TEXT    NOT NULL,
    roll_no  TEXT    NOT NULL UNIQUE,
    semester INTEGER NOT NULL,
    branch   TEXT    NOT NULL
);

CREATE TABLE subjects (
    id       INTEGER PRIMARY KEY,
    name     TEXT    NOT NULL,
    semester INTEGER NOT NULL,
    credits  INTEGER NOT NULL
);

CREATE TABLE marks (
    id         INTEGER PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES students(id),
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    exam_type  TEXT    NOT NULL,
    score      INTEGER NOT NULL,
    max_score  INTEGER NOT NULL
);

CREATE TABLE attendance (
    id         INTEGER PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES students(id),
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    date       TEXT    NOT NULL,
    present    INTEGER NOT NULL
);

CREATE TABLE timetable (
    id         INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL REFERENCES subjects(id),
    semester   INTEGER NOT NULL,
    day        TEXT    NOT NULL,   -- Monday .. Friday
    start_time TEXT    NOT NULL,   -- HH:MM, 24 hour
    end_time   TEXT    NOT NULL,
    room       TEXT    NOT NULL
);

-- Books currently or recently borrowed. due_date comes from the loan period
-- in the student FAQ (14 days undergraduate); returned_on is NULL while the
-- book is still out.
CREATE TABLE library_loans (
    id          INTEGER PRIMARY KEY,
    student_id  INTEGER NOT NULL REFERENCES students(id),
    title       TEXT    NOT NULL,
    issued_on   TEXT    NOT NULL,
    due_date    TEXT    NOT NULL,
    returned_on TEXT
);

-- Written to by the reminder tool using bound parameters only. The
-- text-to-SQL tool never touches this table: its connection is read-only.
CREATE TABLE reminders (
    id         INTEGER PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES students(id),
    text       TEXT    NOT NULL,
    due_date   TEXT,
    created_at TEXT    NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0
);

-- Login accounts. student_id is UNIQUE so two people cannot claim the same
-- student record. Passwords are bcrypt hashes; nothing here is reversible.
CREATE TABLE users (
    id              INTEGER PRIMARY KEY,
    username        TEXT    NOT NULL UNIQUE,   -- roll number, or a staff login
    password_hash   TEXT    NOT NULL,
    role            TEXT    NOT NULL CHECK (role IN ('student', 'staff')),
    student_id      INTEGER UNIQUE REFERENCES students(id),  -- NULL for staff
    token_version   INTEGER NOT NULL DEFAULT 1,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT,
    created_at      TEXT    NOT NULL,
    last_login      TEXT
);
"""

# Demo logins, documented in the README so a reviewer can sign in immediately.
DEMO_STUDENT_PASSWORD = "demo1234"
DEMO_STAFF_PASSWORD = "staff1234"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SLOTS = [("09:00", "10:00"), ("10:15", "11:15"), ("11:30", "12:30"), ("14:00", "15:00")]

FIRST_NAMES = [
    "Aarav", "Diya", "Rohan", "Ananya", "Kabir", "Meera", "Arjun", "Isha",
    "Vikram", "Priya", "Nikhil", "Sneha", "Rahul", "Tara", "Aditya", "Kavya",
    "Manav", "Riya", "Siddharth", "Neha", "Karan", "Pooja", "Aryan", "Sanya",
    "Devansh", "Ira", "Yash", "Anika", "Harsh", "Nandini",
]
LAST_NAMES = [
    "Sharma", "Verma", "Iyer", "Nair", "Reddy", "Gupta", "Mehta", "Kulkarni",
    "Bose", "Chopra", "Desai", "Joshi", "Menon", "Pillai", "Rao",
]

BRANCHES = ["Computer Science", "Electronics", "Mechanical"]

SUBJECTS = [
    # (name, semester, credits)
    ("Data Structures", 4, 4),
    ("Database Systems", 4, 3),
    ("Operating Systems", 4, 4),
    ("Compiler Design", 6, 4),
    ("Artificial Intelligence", 6, 3),
    ("Computer Networks", 6, 3),
]

BOOK_TITLES = [
    "Introduction to Algorithms", "Compilers: Principles and Techniques",
    "Artificial Intelligence: A Modern Approach", "Computer Networks",
    "Operating System Concepts", "Database System Concepts",
    "Clean Code", "The Pragmatic Programmer",
]
LOAN_DAYS = 14  # undergraduate loan period, per docs/student_faq.md

EXAM_TYPES = ["midterm", "final", "quiz"]
MAX_SCORES = {"midterm": 50, "final": 100, "quiz": 20}


def build_students(rng: random.Random) -> list[tuple]:
    """30 students across semesters 4 and 6, unique names and roll numbers."""
    students: list[tuple] = []
    used_names: set[str] = set()

    for i in range(N_STUDENTS):
        while True:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            if name not in used_names:
                used_names.add(name)
                break
        semester = 4 if i < N_STUDENTS // 2 else 6
        branch = BRANCHES[i % len(BRANCHES)]
        roll_no = f"CS{semester:02d}{i + 1:03d}"
        students.append((i + 1, name, roll_no, semester, branch))
    return students


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)

        students = build_students(rng)
        conn.executemany(
            "INSERT INTO students (id, name, roll_no, semester, branch) "
            "VALUES (?, ?, ?, ?, ?)",
            students,
        )

        subjects = [
            (i + 1, name, semester, credits)
            for i, (name, semester, credits) in enumerate(SUBJECTS)
        ]
        conn.executemany(
            "INSERT INTO subjects (id, name, semester, credits) VALUES (?, ?, ?, ?)",
            subjects,
        )

        # Map semester -> subject ids so students only get their own subjects.
        by_semester: dict[int, list[int]] = {}
        for sid, _name, semester, _credits in subjects:
            by_semester.setdefault(semester, []).append(sid)

        # --- marks ---------------------------------------------------------
        marks: list[tuple] = []
        for i in range(N_MARKS):
            student = rng.choice(students)
            subject_id = rng.choice(by_semester[student[3]])
            exam_type = rng.choice(EXAM_TYPES)
            max_score = MAX_SCORES[exam_type]
            # Skew towards a believable class distribution (mean ~68%).
            fraction = min(1.0, max(0.25, rng.gauss(0.68, 0.16)))
            score = round(fraction * max_score)
            marks.append((i + 1, student[0], subject_id, exam_type, score, max_score))

        conn.executemany(
            "INSERT INTO marks (id, student_id, subject_id, exam_type, score, "
            "max_score) VALUES (?, ?, ?, ?, ?, ?)",
            marks,
        )

        # --- attendance ----------------------------------------------------
        term_start = date(2025, 1, 6)
        class_days = [
            term_start + timedelta(days=offset)
            for offset in range(0, 84)
            if (term_start + timedelta(days=offset)).weekday() < 5
        ]

        attendance: list[tuple] = []
        # Each student has their own attendance habit, so "lowest attendance"
        # has a stable, meaningful answer.
        habit = {s[0]: rng.uniform(0.55, 0.98) for s in students}
        for i in range(N_ATTENDANCE):
            student = rng.choice(students)
            subject_id = rng.choice(by_semester[student[3]])
            day = rng.choice(class_days)
            present = 1 if rng.random() < habit[student[0]] else 0
            attendance.append(
                (i + 1, student[0], subject_id, day.isoformat(), present)
            )

        conn.executemany(
            "INSERT INTO attendance (id, student_id, subject_id, date, present) "
            "VALUES (?, ?, ?, ?, ?)",
            attendance,
        )

        # --- timetable ------------------------------------------------------
        # Each semester's subjects repeat through the week in fixed slots, so
        # "what's my next class?" always has a sensible answer.
        rows: list[tuple] = []
        next_id = 1
        for semester, subject_ids in sorted(by_semester.items()):
            slot = 0
            for day in DAYS:
                # Three lectures a day, cycling through that semester's subjects.
                for _ in range(3):
                    subject_id = subject_ids[slot % len(subject_ids)]
                    start, end = SLOTS[slot % len(SLOTS)]
                    room = f"{'A' if semester == 4 else 'B'}-{100 + (slot % 5)}"
                    rows.append((next_id, subject_id, semester, day, start, end, room))
                    next_id += 1
                    slot += 1

        # Keep each day's lectures in ascending time order.
        rows.sort(key=lambda r: (r[2], DAYS.index(r[3]), r[4]))
        rows = [(i + 1,) + r[1:] for i, r in enumerate(rows)]
        conn.executemany(
            "INSERT INTO timetable (id, subject_id, semester, day, start_time, "
            "end_time, room) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        timetable_rows = rows

        # --- library loans --------------------------------------------------
        # A spread of states so "do I have a book due?" has a real answer:
        # some returned, some due soon, some already overdue.
        loans: list[tuple] = []
        today = date(2026, 7, 29)
        for i, student in enumerate(rng.sample(students, 14)):
            title = BOOK_TITLES[i % len(BOOK_TITLES)]
            # offset -21..+10 days from today, so roughly a third are overdue
            offset = rng.choice([-21, -12, -5, -3, -1, 0, 2, 4, 6, 9, 10])
            due = today + timedelta(days=offset)
            issued = due - timedelta(days=LOAN_DAYS)
            # A quarter are already back on the shelf.
            returned = (issued + timedelta(days=rng.randint(3, LOAN_DAYS))).isoformat()                 if rng.random() < 0.25 else None
            loans.append((len(loans) + 1, student[0], title,
                          issued.isoformat(), due.isoformat(), returned))

        conn.executemany(
            "INSERT INTO library_loans (id, student_id, title, issued_on, "
            "due_date, returned_on) VALUES (?, ?, ?, ?, ?, ?)",
            loans,
        )

        # --- demo accounts --------------------------------------------------
        # Imported lazily so `python data/seed_db.py` still works if bcrypt is
        # somehow missing; the rest of the seed is more important than these.
        try:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from app.auth import hash_password

            now = datetime.now().isoformat(timespec="seconds")
            demo_student = students[19]  # id 20, semester 6
            conn.executemany(
                "INSERT INTO users (username, password_hash, role, student_id, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        demo_student[2],  # roll_no
                        hash_password(DEMO_STUDENT_PASSWORD),
                        "student",
                        demo_student[0],
                        now,
                    ),
                    (
                        "staff",
                        hash_password(DEMO_STAFF_PASSWORD),
                        "staff",
                        None,
                        now,
                    ),
                ],
            )
            seeded_accounts = [
                f"{demo_student[2]} / {DEMO_STUDENT_PASSWORD}  (student: {demo_student[1]})",
                f"staff / {DEMO_STAFF_PASSWORD}  (staff, sees everything)",
            ]
        except ImportError as exc:
            print(f"  ! could not seed demo accounts, bcrypt/PyJWT missing: {exc}")
            print("  ! run 'pip install -r requirements.txt' then re-seed")
            seeded_accounts = []

        conn.commit()
    finally:
        conn.close()

    print(f"Seeded {DB_PATH}")
    print(
        f"  students   : {len(students)}\n"
        f"  subjects   : {len(subjects)}\n"
        f"  marks      : {len(marks)}\n"
        f"  attendance : {len(attendance)}\n"
        f"  timetable  : {len(timetable_rows)}\n"
        f"  library    : {len(loans)} loans\n"
        f"  reminders  : 0 (created at runtime)"
    )
    if seeded_accounts:
        print("\nDemo logins:")
        for line in seeded_accounts:
            print(f"  {line}")


if __name__ == "__main__":
    main()
