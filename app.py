
import json
import uuid
import html
from datetime import datetime

import streamlit as st
from langdetect import detect, DetectorFactory, LangDetectException

import assistant_backend

DetectorFactory.seed = 0

st.set_page_config(
    page_title="Chatty | Your AI companion",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inline SVG mascot: no separate image file is required.
ROBOT = r'''<svg viewBox="0 0 240 245" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="shell" x1="0" y1="0" x2="1" y2="1">
<stop stop-color="#FFFFFF"/><stop offset=".55" stop-color="#DCEAFF"/>
<stop offset="1" stop-color="#8DB8F4"/></linearGradient>
<linearGradient id="blue" x1="0" y1="0" x2="1" y2="1">
<stop stop-color="#69C8FF"/><stop offset="1" stop-color="#176BEE"/></linearGradient>
<linearGradient id="screen" x1="0" y1="0" x2="1" y2="1">
<stop stop-color="#142B50"/><stop offset="1" stop-color="#07152D"/></linearGradient>
<filter id="glow"><feGaussianBlur stdDeviation="5" result="b"/>
<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<ellipse cx="120" cy="226" rx="67" ry="9" fill="#78AFFF" opacity=".22"/>
<path d="M120 12v18" stroke="#54B9FF" stroke-width="5" stroke-linecap="round"/>
<circle cx="120" cy="9" r="8" fill="url(#blue)" filter="url(#glow)"/>
<rect x="29" y="60" width="182" height="126" rx="48" fill="url(#shell)" stroke="#A8CBFF" stroke-width="4"/>
<rect x="42" y="70" width="156" height="103" rx="35" fill="url(#screen)" stroke="#438CFB" stroke-width="6"/>
<path d="M25 91q-13 22 0 43M215 91q13 22 0 43" fill="none" stroke="#9CC5FF" stroke-width="12" stroke-linecap="round"/>
<g fill="#78D5FF" filter="url(#glow)">
<ellipse cx="91" cy="119" rx="10" ry="15"/><ellipse cx="149" cy="119" rx="10" ry="15"/>
</g>
<path d="M105 145q15 10 30 0" fill="none" stroke="#78D5FF" stroke-width="5" stroke-linecap="round"/>
<rect x="91" y="185" width="58" height="13" rx="6.5" fill="url(#blue)"/>
<path d="M66 194q-24 17-22 42h152q2-25-22-42l-23-6h-62z" fill="url(#shell)" stroke="#A8CBFF" stroke-width="4"/>
<rect x="101" y="201" width="38" height="20" rx="10" fill="url(#blue)"/>
<circle cx="120" cy="211" r="6" fill="#DDF6FF"/>
<path d="M27 113h-9M222 113h-9" stroke="#6BB8FF" stroke-width="5" stroke-linecap="round"/>
<path d="M187 33l5 12 12 5-12 5-5 12-5-12-12-5 12-5z" fill="#55B8FF"/>
</svg>'''


def now():
    return datetime.now().isoformat(timespec="seconds")


def detect_language(text):
    letters = [c for c in text if c.isalpha()]
    arabic = sum("\u0600" <= c <= "\u06FF" for c in letters)

    if letters and arabic / len(letters) > 0.2:
        return "ar", "Arabic"

    if len(text.strip()) < 8:
        return "unknown", "Undetermined"

    try:
        code = detect(text)
    except LangDetectException:
        return "unknown", "Undetermined"

    return code, {"en": "English", "fr": "French", "ar": "Arabic"}.get(code, "Other")


def new_conversation():
    return {
        "id": str(uuid.uuid4()),
        "title": "New conversation",
        "created_at": now(),
        "updated_at": now(),
        "settings": {
            "response_language": "Automatic",
            "response_style": "Balanced",
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 160,
            "custom_instructions": "",
        },
        "messages": [],
    }


def export_markdown(chat):
    lines = [
        f"# {chat['title']}",
        "",
        f"Created: {chat['created_at']}",
        "",
    ]

    for item in chat["messages"]:
        who = "You" if item["role"] == "user" else "Chatty"
        lines.extend([f"## {who}", "", item["content"], ""])

    return "\n".join(lines)


def import_chat(upload):
    text = upload.getvalue().decode("utf-8", errors="replace")
    suffix = upload.name.lower().rsplit(".", 1)[-1]

    chat = new_conversation()
    chat["title"] = upload.name.rsplit(".", 1)[0]

    if suffix == "json":
        data = json.loads(text)
        raw = data.get("messages", []) if isinstance(data, dict) else data

        if isinstance(data, dict):
            chat["title"] = data.get("title", chat["title"])

        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue

            role = str(item.get("role", "")).lower()
            role = (
                "user" if role in {"user", "human"}
                else "assistant" if role in {"assistant", "bot", "chatty"}
                else ""
            )
            content = item.get("content", "")

            if role and isinstance(content, str) and content.strip():
                code, _ = detect_language(content)
                chat["messages"].append({
                    "role": role,
                    "content": content,
                    "language": item.get("language", code),
                    "timestamp": item.get("timestamp", now()),
                })
    else:
        if not text.strip():
            raise ValueError("The uploaded file is empty.")

        code, _ = detect_language(text)
        chat["messages"].append({
            "role": "user",
            "content": text,
            "language": code,
            "timestamp": now(),
        })

    if not chat["messages"]:
        raise ValueError("No valid conversation messages were found.")

    return chat


def reply(chat, prompt):
    prompt = prompt.strip()

    if not prompt:
        return

    code, language_name = detect_language(prompt)

    chat["messages"].append({
        "role": "user",
        "content": prompt,
        "language": code,
        "timestamp": now(),
    })

    if chat["title"] == "New conversation":
        chat["title"] = " ".join(prompt.split())[:42]

    settings = chat["settings"]

    try:
        result = assistant_backend.chat_completion(
            messages=chat["messages"],
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            max_tokens=int(settings["max_tokens"]),
            response_language=settings["response_language"],
            response_style=settings["response_style"],
            custom_instructions=settings["custom_instructions"],
        )

        answer = {
            "role": "assistant",
            "content": result["content"],
            "language": code,
            "timestamp": now(),
            "statistics": result.get("usage", {}),
        }

    except Exception as exc:
        answer = {
            "role": "assistant",
            "content": (
                "I couldn't connect to the assistant right now. "
                "Please make sure the Qwen server is running, then try again.\n\n"
                f"Technical detail: {exc}"
            ),
            "language": code,
            "timestamp": now(),
            "error": True,
        }

    chat["messages"].append(answer)
    chat["updated_at"] = now()
    st.session_state.last_language = language_name


# ---------------------------- Session state ----------------------------

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "conversations" not in st.session_state:
    first = new_conversation()
    st.session_state.conversations = {first["id"]: first}
    st.session_state.current_id = first["id"]

if "current_id" not in st.session_state:
    st.session_state.current_id = next(iter(st.session_state.conversations))

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "last_language" not in st.session_state:
    st.session_state.last_language = "Not detected"

if st.session_state.current_id not in st.session_state.conversations:
    first = new_conversation()
    st.session_state.conversations[first["id"]] = first
    st.session_state.current_id = first["id"]

chat = st.session_state.conversations[st.session_state.current_id]


# ---------------------------- Theme ----------------------------

if st.session_state.dark_mode:
    theme = {
        "bg": "#0B1220", "side": "#0F1A2D", "panel": "#142139",
        "panel2": "#1B2D49", "text": "#F3F7FF", "muted": "#A8B7CE",
        "border": "#2B4161", "accent": "#287CF5", "input": "#172943",
        "user": "#183C70",
    }
else:
    theme = {
        "bg": "#F2F8FF", "side": "#FFFFFF", "panel": "#FFFFFF",
        "panel2": "#EAF3FF", "text": "#17243A", "muted": "#65758E",
        "border": "#DCE8F7", "accent": "#2878F5", "input": "#FFFFFF",
        "user": "#DDEBFF",
    }


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {{
 --bg:{theme['bg']}; --side:{theme['side']}; --panel:{theme['panel']};
 --panel2:{theme['panel2']}; --text:{theme['text']}; --muted:{theme['muted']};
 --border:{theme['border']}; --accent:{theme['accent']};
 --input:{theme['input']}; --user:{theme['user']};
}}

html, body, [class*="css"], .stApp {{
 font-family:'Manrope','Segoe UI',sans-serif !important;
}}
.stApp,[data-testid="stAppViewContainer"] {{
 background:var(--bg) !important; color:var(--text) !important;
}}
[data-testid="stHeader"] {{ background:transparent !important; }}
[data-testid="stSidebar"] {{
 background:var(--side) !important;
 border-right:1px solid var(--border) !important;
}}
.block-container {{ max-width:1180px; padding-top:1.25rem; padding-bottom:2rem; }}
[data-testid="stMarkdownContainer"] {{ color:var(--text) !important; }}
[data-testid="stMarkdownContainer"] p {{ color:inherit !important; }}

.brand {{
 font-size:2.1rem; font-weight:800; letter-spacing:-1.5px; line-height:1.1;
 background:linear-gradient(100deg,#1B4C91,#2878F5,#62BFFF);
 -webkit-background-clip:text; background-clip:text;
 color:transparent !important;
}}
.tagline {{ color:var(--muted) !important; font-size:.88rem; }}

.hero {{
 border:1px solid var(--border); border-radius:25px; padding:1.8rem;
 background:linear-gradient(120deg,
 {'#142746' if st.session_state.dark_mode else '#E9F4FF'},
 {'#1B3760' if st.session_state.dark_mode else '#FFFFFF'});
}}
.hero-kicker {{ color:var(--accent) !important; font-size:.75rem; font-weight:800; letter-spacing:1.8px; }}
.hero-title {{ color:var(--text) !important; font-size:clamp(2rem,4vw,3.25rem); font-weight:800; letter-spacing:-1.8px; line-height:1.1; margin:.5rem 0; }}
.hero-title span {{ color:var(--accent) !important; }}
.hero-copy {{ color:var(--muted) !important; font-size:1rem; line-height:1.7; }}
.robot-box {{ max-width:250px; margin:auto; padding:1rem; border-radius:24px; background:radial-gradient(circle,#D9EBFF 0%,#9CC8FF 48%,#377FE6 100%); box-shadow:0 12px 36px rgba(40,120,245,.20); }}
.robot-box svg {{ width:100%; height:auto; display:block; }}
.robot-small {{ width:82px; }}
.robot-small svg {{ width:100%; height:auto; display:block; }}
.section-label {{ color:var(--muted) !important; font-size:.75rem; font-weight:800; letter-spacing:1.4px; text-transform:uppercase; margin:.7rem 0; }}

/* Buttons: explicit foreground and background colors. */
div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button {{
 background:var(--panel2) !important;
 color:var(--text) !important;
 -webkit-text-fill-color:var(--text) !important;
 border:1px solid var(--border) !important;
 border-radius:13px !important;
 min-height:2.65rem;
 font-weight:700 !important;
 opacity:1 !important;
}}
div[data-testid="stButton"] button *,
div[data-testid="stDownloadButton"] button * {{
 color:inherit !important;
 -webkit-text-fill-color:inherit !important;
}}
div[data-testid="stButton"] button[kind="primary"] {{
 background:var(--accent) !important;
 color:#FFFFFF !important;
 -webkit-text-fill-color:#FFFFFF !important;
 border-color:var(--accent) !important;
}}
div[data-testid="stButton"] button[kind="primary"] * {{
 color:#FFFFFF !important;
 -webkit-text-fill-color:#FFFFFF !important;
}}

/* Message composer contrast in both themes. */
[data-testid="stChatInput"] {{
 background:var(--input) !important;
 border:1px solid var(--border) !important;
 border-radius:20px !important;
 box-shadow:0 8px 24px rgba(30,70,130,.08) !important;
}}
[data-testid="stChatInput"] div {{ background-color:transparent !important; }}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] [contenteditable="true"] {{
 background:var(--input) !important;
 color:var(--text) !important;
 -webkit-text-fill-color:var(--text) !important;
 caret-color:var(--accent) !important;
 border:0 !important;
 box-shadow:none !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
 color:var(--muted) !important;
 -webkit-text-fill-color:var(--muted) !important;
 opacity:1 !important;
}}
input, textarea {{
 color:var(--text) !important;
 -webkit-text-fill-color:var(--text) !important;
 background:var(--input) !important;
 caret-color:var(--accent) !important;
}}
input::placeholder, textarea::placeholder {{ color:var(--muted) !important; opacity:1 !important; }}

[data-testid="stChatMessage"] {{
 background:var(--panel) !important;
 border:1px solid var(--border) !important;
 border-radius:18px !important;
 padding:.8rem 1rem !important;
}}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{ color:var(--text) !important; }}
[data-testid="stExpander"] {{ background:var(--panel) !important; border:1px solid var(--border) !important; border-radius:14px !important; }}
[data-testid="stFileUploader"] {{ background:var(--panel2) !important; border-radius:13px !important; }}
[data-testid="stFileUploader"] * {{ color:var(--text) !important; }}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p {{ color:var(--text) !important; }}
.arabic {{ direction:rtl; text-align:right; white-space:pre-wrap; line-height:1.9; color:var(--text); }}
hr {{ border-color:var(--border) !important; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------- Sidebar ----------------------------

with st.sidebar:
    col_img, col_brand = st.columns([.8, 1.7], vertical_alignment="center")

    with col_img:
        st.markdown(f'<div class="robot-small">{ROBOT}</div>', unsafe_allow_html=True)

    with col_brand:
        st.markdown(
            '<div class="brand">Chatty ✦</div><div class="tagline">Your AI companion</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    if st.button("＋  New chat", type="primary", use_container_width=True):
        fresh = new_conversation()
        st.session_state.conversations[fresh["id"]] = fresh
        st.session_state.current_id = fresh["id"]
        st.rerun()

    st.markdown('<div class="section-label">Your conversations</div>', unsafe_allow_html=True)

    for cid in reversed(list(st.session_state.conversations.keys())):
        item = st.session_state.conversations[cid]
        label = item["title"][:30] or "New conversation"

        if st.button(label, key=f"history_{cid}", use_container_width=True):
            st.session_state.current_id = cid
            st.rerun()

    st.divider()
    st.toggle("Dark mode", key="dark_mode")

    with st.expander("Personalize Chatty"):
        settings = chat["settings"]

        settings["response_language"] = st.selectbox(
            "Response language",
            ["Automatic", "English", "French", "Arabic"],
            index=["Automatic", "English", "French", "Arabic"].index(settings["response_language"]),
        )

        settings["response_style"] = st.selectbox(
            "Response style",
            ["Balanced", "Friendly", "Professional", "Concise", "Detailed", "Creative"],
            index=["Balanced", "Friendly", "Professional", "Concise", "Detailed", "Creative"].index(settings["response_style"]),
        )

        settings["temperature"] = st.slider(
            "Creativity", .1, 1.5, float(settings["temperature"]), .1
        )

        settings["top_p"] = st.slider(
            "Top-p", .1, 1.0, float(settings["top_p"]), .05
        )

        settings["max_tokens"] = st.slider(
            "Maximum response length", 32, 256, int(settings["max_tokens"]), 16
        )

        settings["custom_instructions"] = st.text_area(
            "Custom instructions",
            value=settings["custom_instructions"],
            placeholder="Example: Explain technical topics with simple examples.",
        )

    with st.expander("Import a conversation"):
        uploaded = st.file_uploader(
            "Upload JSON, Markdown or TXT",
            type=["json", "md", "txt"],
            key="chatty_import",
        )

        if uploaded and st.button("Import conversation", use_container_width=True):
            try:
                imported = import_chat(uploaded)
                st.session_state.conversations[imported["id"]] = imported
                st.session_state.current_id = imported["id"]
                st.rerun()
            except Exception as exc:
                st.error(f"Import failed: {exc}")

    with st.expander("Export and manage"):
        st.download_button(
            "Download JSON",
            data=json.dumps(chat, ensure_ascii=False, indent=2),
            file_name=f"chatty_{chat['id']}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.download_button(
            "Download Markdown",
            data=export_markdown(chat),
            file_name=f"chatty_{chat['id']}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        new_title = st.text_input("Conversation title", value=chat["title"])

        if st.button("Save title", use_container_width=True):
            chat["title"] = new_title.strip() or "New conversation"
            st.rerun()

        if st.button("Delete this conversation", use_container_width=True):
            del st.session_state.conversations[chat["id"]]

            if not st.session_state.conversations:
                fresh = new_conversation()
                st.session_state.conversations[fresh["id"]] = fresh

            st.session_state.current_id = next(iter(st.session_state.conversations))
            st.rerun()

    st.divider()
    st.caption("Conversations stay in this active browser session. Export them to keep a copy.")


# Re-fetch the current conversation after sidebar navigation.
chat = st.session_state.conversations[st.session_state.current_id]


# ---------------------------- Main header ----------------------------

head_robot, head_title, head_space = st.columns(
    [.75, 4.8, 1.1],
    vertical_alignment="center",
)

with head_robot:
    st.markdown(f'<div class="robot-small">{ROBOT}</div>', unsafe_allow_html=True)

with head_title:
    st.markdown(
        '<div class="brand">Chatty ✦</div><div class="tagline">Your conversational AI assistant</div>',
        unsafe_allow_html=True,
    )

with head_space:
    st.markdown(
        '<div style="color:#20B982;font-weight:700;text-align:right">● Ready</div>',
        unsafe_allow_html=True,
    )

st.divider()


# ---------------------------- Welcome ----------------------------

if not chat["messages"]:
    left, right = st.columns([1.45, 1], gap="large", vertical_alignment="center")

    with left:
        st.markdown(
            '<div class="hero"><div class="hero-kicker">YOUR AI COMPANION</div>'
            '<div class="hero-title">Ideas become<br><span>conversations.</span></div>'
            '<div class="hero-copy">Ask questions, explore ideas, learn something new, '
            'or get help with a task. Chatty is ready to talk with you.</div></div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(f'<div class="robot-box">{ROBOT}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Try asking</div>', unsafe_allow_html=True)

    suggestions = [
        "Explain machine learning with a simple example.",
        "Help me organize a study plan.",
        "What can I build with Python?",
    ]

    columns = st.columns(3)

    for col, suggestion in zip(columns, suggestions):
        with col:
            if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                st.session_state.pending_prompt = suggestion
                st.rerun()


# ---------------------------- Messages ----------------------------

for item in chat["messages"]:
    role = item.get("role", "assistant")

    if role not in {"user", "assistant"}:
        continue

    avatar = "👤" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar):
        if item.get("error"):
            st.error(item["content"])
        elif item.get("language") == "ar":
            st.markdown(
                f'<div class="arabic" dir="rtl">{html.escape(item["content"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(item["content"])

        if item.get("statistics"):
            with st.expander("Response details"):
                st.json(item["statistics"])


# ---------------------------- Composer ----------------------------

prompt = st.chat_input("Message Chatty...")

if not prompt:
    prompt = st.session_state.pending_prompt

if prompt:
    st.session_state.pending_prompt = None

    with st.spinner("Chatty is thinking..."):
        reply(chat, prompt)

    st.rerun()

st.caption("Chatty · Your conversational AI companion")
