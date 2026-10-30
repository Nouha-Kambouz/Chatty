import requests


# ============================================================
# QWEN3 SERVER CONFIGURATION
# ============================================================

BASE_URL = "http://127.0.0.1:8080"

MODEL_NAME = "QuantFactory/Qwen3-0.6B-GGUF:Q4_K_M"

CHAT_ENDPOINT = f"{BASE_URL}/v1/chat/completions"
MODELS_ENDPOINT = f"{BASE_URL}/v1/models"


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt(
    response_language="Automatic",
    response_style="Balanced",
    custom_instructions=""
):

    language_instructions = {
        "Automatic": (
            "Respond in the same language as the user's latest message. "
            "Support English, French and Arabic."
        ),
        "English": "Always respond in English.",
        "French": "Always respond in French.",
        "Arabic": "Always respond in Arabic."
    }

    style_instructions = {
        "Balanced": "Be clear, natural and balanced.",
        "Friendly": "Use a friendly and approachable tone.",
        "Professional": "Use a professional and precise tone.",
        "Concise": "Keep responses short and direct.",
        "Detailed": "Provide detailed explanations and examples.",
        "Creative": "Use creative examples and engaging explanations."
    }

    system_prompt = f"""
You are Chatty, a helpful conversational AI assistant.

Your responsibilities:
- Answer the user's actual questions.
- Follow instructions carefully.
- Explain difficult concepts clearly.
- Be honest when you do not know something.
- Never claim to execute computer commands or access files.
- Do not invent capabilities or information.

Language:
{language_instructions.get(response_language, language_instructions["Automatic"])}

Response style:
{style_instructions.get(response_style, style_instructions["Balanced"])}

You are a conversational assistant, not an encyclopedia text-completion engine.
"""

    if custom_instructions.strip():
        system_prompt += (
            "\nAdditional user preferences:\n"
            + custom_instructions[:800]
        )

    return system_prompt.strip()


# ============================================================
# CHECK SERVER CONNECTION
# ============================================================

def check_server():

    try:

        response = requests.get(
            MODELS_ENDPOINT,
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        models = data.get("data") or data.get("models", [])

        if not models:
            raise RuntimeError("No model is available on the server.")

        return {
            "connected": True,
            "models": models
        }

    except requests.RequestException as error:

        raise ConnectionError(
            "Cannot connect to the Qwen3 server. "
            "Make sure llama-server is running on port 8080."
        ) from error


# ============================================================
# PREPARE CONVERSATION HISTORY
# ============================================================

def prepare_messages(
    messages,
    response_language="Automatic",
    response_style="Balanced",
    custom_instructions=""
):

    system_message = {
        "role": "system",
        "content": build_system_prompt(
            response_language,
            response_style,
            custom_instructions
        )
    }

    valid_messages = []

    # Keep recent context to fit the 1024-token context window.
    for message in messages[-8:]:

        role = message.get("role")
        content = message.get("content", "")

        if role not in ["user", "assistant"]:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        valid_messages.append({
            "role": role,
            "content": content[-400:]
        })

    if not valid_messages:
        raise ValueError("The conversation contains no valid messages.")

    if valid_messages[-1]["role"] != "user":
        raise ValueError(
            "The latest conversation message must be from the user."
        )

    return [system_message] + valid_messages


# ============================================================
# GENERATE CONVERSATIONAL RESPONSE
# ============================================================

def chat_completion(
    messages,
    temperature=0.7,
    top_p=0.8,
    max_tokens=160,
    response_language="Automatic",
    response_style="Balanced",
    custom_instructions="",
    timeout=120
):

    if not 0.1 <= temperature <= 1.5:
        raise ValueError("Temperature must be between 0.1 and 1.5.")

    if not 0 < top_p <= 1:
        raise ValueError("Top-p must be between 0 and 1.")

    if not 16 <= max_tokens <= 256:
        raise ValueError("Maximum tokens must be between 16 and 256.")

    conversation = prepare_messages(
        messages,
        response_language,
        response_style,
        custom_instructions
    )

    payload = {
        "model": MODEL_NAME,
        "messages": conversation,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 20,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    try:

        response = requests.post(
            CHAT_ENDPOINT,
            json=payload,
            timeout=timeout
        )

        response.raise_for_status()

        data = response.json()

        choices = data.get("choices", [])

        if not choices:
            raise RuntimeError(
                "The Qwen3 server returned no response choices."
            )

        content = choices[0]["message"].get("content", "")

        if not content or not content.strip():
            raise RuntimeError(
                "Qwen3 returned an empty response."
            )

        return {
            "content": content.strip(),
            "model": data.get("model", MODEL_NAME),
            "usage": data.get("usage", {}),
            "finish_reason": choices[0].get("finish_reason")
        }

    except requests.RequestException as error:

        raise ConnectionError(
            f"Qwen3 API request failed: {error}"
        ) from error