from groq import Groq

from . import config

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def generate_reply(system_prompt: str, history: list[dict], message: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    completion = _get_client().chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0.6,
        max_tokens=500,
    )
    return completion.choices[0].message.content
