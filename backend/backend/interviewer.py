import json
import re
import os
import cohere

# Old/Legacy Syntax (v4.x)
client = cohere.Client(os.environ.get("COHERE_API_KEY"))

SYSTEM_PROMPT = """You are an expert technical interviewer conducting a live interview.

You receive:
- The job description
- The candidate's resume
- The full conversation history (all previous Q&A pairs)
- The candidate's latest answer

Your job: evaluate the answer and return a JSON object:
{
  "feedback": "2-3 sentence evaluation of the answer",
  "score": 0.0-1.0,
  "follow_up": "a deeper follow-up if score >= 0.7, else a hint",
  "next_question": "your next interview question",
  "difficulty": "easier|same|harder"
}

STRICT RULES:
- NEVER repeat a question that already appears in the conversation history
- NEVER ask a question similar to one already asked
- Each question must cover a NEW topic or skill not yet discussed
- Cover different areas: technical skills, projects, problem solving, system design, behaviour
- Score >= 0.75: ask harder question on a new topic
- Score 0.4-0.74: ask same difficulty on a new topic
- Score < 0.4: give a hint in follow_up, ask an easier question on the same topic
- Always return raw JSON only. No markdown, no backticks, no explanation."""


def evaluate_and_advance(session: dict) -> dict:
    already_asked = [h["question"] for h in session["history"]]
    already_asked.append(session["current_question"])
    already_asked_str = "\n".join(f"- {q}" for q in already_asked) if already_asked else "None yet"

    user_message = f"""Job Description: {session['jd']}

Resume: {session['resume']}

Missing skills to probe: {', '.join(session['missing_skills'])}

Questions already asked — DO NOT repeat or paraphrase any of these:
{already_asked_str}

Full conversation history:
{format_history(session['history'])}

Current question asked: {session['current_question']}

Candidate's latest answer: {session['latest_answer']}

Instructions:
1. Evaluate the candidate's answer honestly
2. Generate a next_question completely different from all questions listed above
3. The next question should probe a skill or topic not yet covered
4. Return raw JSON only — no markdown, no backticks"""

    try:
        response = client.chat(
            model="command-r-plus",
            messages=user_message,
            preamble=SYSTEM_PROMPT,
        )

        raw = response.text

        # Strip ```json fences if Cohere wraps output
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)

    except json.JSONDecodeError as e:
        result = {
            "feedback": "Could not parse response. Please try again.",
            "score": 0.5,
            "follow_up": "",
            "next_question": session["current_question"],
            "difficulty": "same",
            "parse_error": str(e),
        }
    except Exception as e:
        result = {
            "feedback": f"AI error: {str(e)}. Please try again.",
            "score": 0.5,
            "follow_up": "",
            "next_question": session["current_question"],
            "difficulty": "same",
        }

    return result


def format_history(history: list) -> str:
    if not history:
        return "No prior exchanges."
    return "\n".join(
        f"Q{i+1}: {h['question']}\nA{i+1}: {h['answer']} (score: {h['score']:.1f})"
        for i, h in enumerate(history)
    )