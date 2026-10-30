
import threading
import streamlit as st


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_REPO = "QuantFactory/Qwen3-0.6B-GGUF"

MODEL_FILENAME = "Qwen3-0.6B.Q4_K_M.gguf"

MODEL_CONTEXT = 2048

MODEL_MAX_THREADS = 2

MAX_HISTORY_MESSAGES = 8


# Prevent simultaneous requests from using the same model.
MODEL_LOCK = threading.Lock()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading Chatty's conversational model...")
def load_chat_model():
    from llama_cpp import Llama
    print("Loading Qwen3-0.6B...")

    llm = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,

        # CPU configuration for Streamlit Community Cloud.
        n_gpu_layers=0,
        n_ctx=MODEL_CONTEXT,
        n_threads=MODEL_MAX_THREADS,
        n_threads_batch=MODEL_MAX_THREADS,
        n_batch=128,
        n_ubatch=64,

        # Reduce unnecessary memory locking.
        use_mmap=True,
        use_mlock=False,

        verbose=False
    )

    print("Qwen3 loaded successfully.")

    return llm


# ============================================================
# CHECK MODEL
# ============================================================

def check_model():

    model = load_chat_model()

    return {
        "connected": model is not None,
        "model": MODEL_REPO,
        "quantization": "Q4_K_M",
        "context_length": MODEL_CONTEXT
    }


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt(
    response_language="Automatic",
    response_style="Balanced",
    custom_instructions=""
):

    language_settings = {
        "Automatic": (
            "Respond in the same language as the user's latest message. "
            "Support English, French and Arabic."
        ),
        "English": "Always respond in English.",
        "French": "Always respond in French.",
        "Arabic": "Always respond in Arabic."
    }

    style_settings = {
        "Balanced": "Be clear, helpful and natural.",
        "Friendly": "Use a friendly and approachable tone.",
        "Professional": "Use a professional and precise tone.",
        "Concise": "Keep answers short and direct.",
        "Detailed": "Explain topics carefully and give useful examples.",
        "Creative": "Use creative examples and engaging explanations."
    }

    prompt = f"""
You are Chatty, a helpful and friendly conversational AI assistant.

Your responsibilities:
- Answer the user's actual questions.
- Follow instructions carefully.
- Explain difficult concepts clearly.
- Be honest when you are uncertain.
- Do not invent facts.
- Do not claim to access the user's computer or execute commands.
- Maintain context from previous messages.

Language instructions:
{language_settings.get(response_language, language_settings["Automatic"])}

Response style:
{style_settings.get(response_style, style_settings["Balanced"])}

Respond naturally without displaying internal reasoning.
Use /no_think mode.
"""

    if custom_instructions.strip():
        prompt += (
            "\nAdditional conversation preferences:\n"
            + custom_instructions[:800]
        )

    return prompt.strip()


# ============================================================
# PREPARE CONVERSATION HISTORY
# ============================================================

def prepare_messages(
    llm,
    messages,
    response_language,
    response_style,
    custom_instructions,
    max_tokens
):

    system_message = {
        "role": "system",
        "content": build_system_prompt(
            response_language,
            response_style,
            custom_instructions
        )
    }

    # Keep only valid recent messages.
    history = []

    for item in messages[-MAX_HISTORY_MESSAGES:]:

        role = item.get("role")
        content = item.get("content", "")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        history.append({
            "role": role,
            "content": content[-1200:]
        })

    if not history:
        raise ValueError("No valid conversation messages were provided.")

    if history[-1]["role"] != "user":
        raise ValueError(
            "The latest message must be from the user."
        )

    prepared = [system_message] + history

    # Estimate actual tokenizer usage and remove oldest messages
    # until sufficient space remains for the response.
    def estimate_tokens(items):

        total = 0

        for item in items:

            encoded = llm.tokenize(
                item["content"].encode("utf-8"),
                add_bos=False
            )

            total += len(encoded) + 12

        return total

    available_prompt_tokens = (
        MODEL_CONTEXT - max_tokens - 64
    )

    while (
        len(prepared) > 2
        and estimate_tokens(prepared) > available_prompt_tokens
    ):
        # Preserve the system message and latest user request.
        prepared.pop(1)

    # If the latest prompt alone is too large, shorten it.
    if estimate_tokens(prepared) > available_prompt_tokens:

        last_message = prepared[-1]

        while (
            len(last_message["content"]) > 100
            and estimate_tokens(prepared) > available_prompt_tokens
        ):
            last_message["content"] = (
                last_message["content"][len(last_message["content"]) // 2:]
            )

    if estimate_tokens(prepared) > available_prompt_tokens:
        raise ValueError(
            "The prompt is too long for the available context window."
        )

    # Add Qwen3's non-thinking instruction.
    # This is a supported prompt-level switch for Qwen3.
    prepared[-1]["content"] += "\n/no_think"

    return prepared


# ============================================================
# GENERATE RESPONSE
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

    if not 0.1 <= temperature <= 1.5:
        raise ValueError("Temperature must be between 0.1 and 1.5.")

    if not 0.1 <= top_p <= 1.0:
        raise ValueError("Top-p must be between 0.1 and 1.0.")

    if not 16 <= max_tokens <= 256:
        raise ValueError("Maximum tokens must be between 16 and 256.")

    llm = load_chat_model()

    prepared = prepare_messages(
        llm=llm,
        messages=messages,
        response_language=response_language,
        response_style=response_style,
        custom_instructions=custom_instructions,
        max_tokens=max_tokens
    )

    # llama.cpp's model object is shared, so protect generation.
    with MODEL_LOCK:

        response = llm.create_chat_completion(
        messages=prepared,

        temperature=temperature,
        top_p=top_p,
        top_k=20,

        max_tokens=max_tokens,

        repeat_penalty=1.1,

        stream=False
)

    choices = response.get("choices", [])

    if not choices:
        raise RuntimeError("Qwen3 returned an empty response.")

    import re

    content = re.sub(
        r"<think>.*?</think>",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Qwen3 generated an empty response.")

    return {
        "content": content.strip(),
        "model": MODEL_REPO,
        "usage": response.get("usage", {}),
        "finish_reason": choices[0].get("finish_reason")
    }
