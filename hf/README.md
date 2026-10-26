---
title: College Voice Assistant
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 8000
pinned: false
short_description: Voice assistant with LLM tool routing, text-to-SQL and RAG
---

# College Voice Assistant

Ask a small college anything, by voice or by typing. An LLM router picks the right
tool (student records via safe text-to-SQL, policy documents via RAG, timetable,
reminders or small talk) and answers in a short spoken reply.

## Try it

Sign in with one of the demo accounts:

| Login | Password | Sees |
| --- | --- | --- |
| `CS06020` | `demo1234` | one student's own records |
| `staff` | `staff1234` | the whole college |

All data is made up. This is a free public demo, so each visitor gets a limited
number of questions every few minutes.

Source code and full documentation:
https://github.com/ManasvibaRana/Voice-Enabled-Virtual-Assistance
