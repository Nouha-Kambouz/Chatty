
import os
import requests
import streamlit as st


# ============================================================
# BACKEND CONFIGURATION
# ============================================================

LOCAL_BASE_URL = "http://127.0.0.1:8080"

MODEL_NAME = "QuantFactory/Qwen3-0.6B-GGUF:Q4_K_M"

LOCAL_CHAT_URL = f"{LOCAL_BASE_URL}/v1/chat/completions"
LOCAL_MODELS_URL = f"{LOCAL_BASE_URL}/v1/models"


# ============================================================
# SELECT BACKEND
# ============================================================

def get_backend_mode():

    # Streamlit Cloud uses an app secret.
    try:
        cloud_setting = st.secrets.get("CHATTY_BACKEND", "")
    except Exception:
        cloud_setting = ""

    # Local development can use an environment variable.
    environment_setting = os.getenv("CHATTY_BACKEND", "")

    mode = (
        cloud_setting
        or environment_setting
        or "local"
    ).strip().lower()

    if mode not in {"local", "cloud"}:
        raise ValueError(
            "CHATTY_BACKEND must be either 'local' or 'cloud'."
        )

    return mode


# ============================================================
# CHECK CONNECTION
# ============================================================

def check_server():

    mode = get_backend_mode()

    if mode == "cloud":

        # Import only when the cloud backend is selected.
        import cloud_assistant_backend

        result = cloud_assistant_backend.check_model()
        result["backend"] = "cloud"

        return result

    try:

        response = requests.get(
            LOCAL_MODELS_URL,
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        models = data.get("data") or data.get("models", [])

        if not models:
            raise RuntimeError("No local model is loaded.")

        return {
            "connected": True,
            "backend": "local",
            "models": models
        }

    except requests.RequestException as error:

        raise ConnectionError(
            "The local Qwen server is unavailable. "
            "Start llama-server on port 8080."
        ) from error


# ============================================================
# SYSTEM PROMPT FOR LOCAL BACKEND
# ============================================================

def build_system_prompt(
    response_language="Automatic",
    response_style="Balanced",
    custom_instructions=""
):

    languages = {
        "Automatic": (
            "Respond in the same language as the user's latest message. "
            "Support English, French and Arabic."
        ),
        "English": "Always respond in English.",
        "French": "Always respond in French.",
        "Arabic": "Always respond in Arabic."
    }

    styles = {
        "Balanced": "Be clear, natural and balanced.",
        "Friendly": "Use a friendly and approachable tone.",
        "Professional": "Use a professional and precise tone.",
        "Concise": "Keep answers short and direct.",
        "Detailed": "Provide explanations and useful examples.",
        "Creative": "Use creative examples and engaging explanations."
    }

    prompt = f"""
You are Chatty, a friendly and helpful conversational assistant.

Answer the user's actual questions.
Follow instructions carefully.
Explain difficult concepts clearly.
Be honest when uncertain.
Do not invent information.
Do not claim to access the user's computer or execute commands.
Maintain context from previous messages.

Language:
{languages.get(response_language, languages["Automatic"])}

Style:
{styles.get(response_style, styles["Balanced"])}

Respond naturally and directly.
"""

    if custom_instructions.strip():
        prompt += (
            "\nAdditional preferences:\n"
            + custom_instructions[:800]
        )

    return prompt.strip()


# ============================================================
# LOCAL CHAT COMPLETION
# ============================================================

def local_chat_completion(
    messages,
    temperature=0.7,
    top_p=0.8,
    max_tokens=160,
    response_language="Automatic",
    response_style="Balanced",
    custom_instructions=""
):

    valid_messages = []

    for item in messages[-8:]:

        role = item.get("role")
        content = item.get("content", "")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        valid_messages.append({
            "role": role,
            "content": content[-1200:]
        })

    if not valid_messages:
        raise ValueError("No valid messages were provided.")

    if valid_messages[-1]["role"] != "user":
        raise ValueError("The latest message must be from the user.")

    prepared_messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                response_language,
                response_style,
                custom_instructions
            )
        }
    ] + valid_messages

    payload = {
        "model": MODEL_NAME,
        "messages": prepared_messages,
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
            LOCAL_CHAT_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        choices = data.get("choices", [])

        if not choices:
            raise RuntimeError("The local model returned no response.")

        content = choices[0]["message"].get("content", "")

        if not content.strip():
            raise RuntimeError("The local model returned an empty answer.")

        return {
            "content": content.strip(),
            "model": data.get("model", MODEL_NAME),
            "usage": data.get("usage", {}),
            "finish_reason": choices[0].get("finish_reason"),
            "backend": "local"
        }

    except requests.RequestException as error:

        raise ConnectionError(
            f"Local Qwen request failed: {error}"
        ) from error


# ============================================================
# MAIN CHAT COMPLETION INTERFACE
# ============================================================

def chat_completion(
    messages,
    temperature=0.7,
    top_p=0.8,
    max_tokens=160,
    response_language="Automatic",
    response_style="Balanced",
    custom_instructions=""
):

    mode = get_backend_mode()

    if mode == "local":

        return local_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_language=response_language,
            response_style=response_style,
            custom_instructions=custom_instructions
        )

    # Cloud backend is loaded only when needed.
    import cloud_assistant_backend

    result = cloud_assistant_backend.chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_language=response_language,
        response_style=response_style,
        custom_instructions=custom_instructions
    )

    result["backend"] = "cloud"

    return result
