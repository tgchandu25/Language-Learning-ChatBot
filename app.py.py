
import os
import sqlite3
import datetime
import pandas as pd
import gradio as gr

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ✅ Load API key securely
openai_api_key = os.environ["OPENAI_API_KEY"]

# ✅ Setup SQLite database (local to HF Space)
db_path = 'mistakes.db'
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS mistakes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT,
    mistake_type TEXT,
    correction TEXT,
    timestamp TEXT
)
""")
conn.commit()

# ✅ OpenAI model setup
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=openai_api_key
)

# 🧠 Global state
session_state = {
    "known_language": "",
    "target_language": "",
    "proficiency_level": "",
    "scene": "",
    "messages": [],
    "scene_prompt": ""
}

scenes = [
    "Ordering food at a restaurant",
    "Shopping at a clothing store",
    "Asking for directions in a new city",
    "Introducing yourself to someone",
    "Booking a hotel room"
]

def setup_profile(known_language, target_language, level, selected_scene):
    if not all([known_language, target_language, level, selected_scene]):
        return "❌ Please fill all fields before starting.", gr.update(interactive=False), gr.update(interactive=False)

    session_state["known_language"] = known_language
    session_state["target_language"] = target_language
    session_state["proficiency_level"] = level
    session_state["scene"] = selected_scene

    prompt = f"""
You are a friendly, multilingual language tutor helping a user learn {target_language}.

They are a {level.lower()} learner whose native language is {known_language}.
Set the scene: {selected_scene}.

Start the conversation in {target_language} only.
Use simple vocabulary and grammar suited for a {level.lower()} learner.
Do NOT include translations. Just speak in {target_language}.
"""
    session_state["scene_prompt"] = prompt
    session_state["messages"] = [SystemMessage(content=prompt)]

    try:
        first_msg = llm.invoke(session_state["messages"])
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(interactive=False), gr.update(interactive=False)

    session_state["messages"].append(first_msg)
    return f"Scene set: {selected_scene}\n\n🤖 Bot ({target_language}): {first_msg.content}", gr.update(interactive=True), gr.update(interactive=True)

def handle_message(user_input, chat_history):
    if user_input.strip().lower() in ["exit", "quit"]:
        return chat_history + [[user_input, "🛑 Session ended."]], gr.update(interactive=False)

    session_state["messages"].append(HumanMessage(content=user_input))

    correction_prompt = f"""
The user is learning {session_state['target_language']} and just said: "{user_input}"

Please do two things:
1. Reply in {session_state['target_language']} ONLY, continuing the {session_state['scene']} scene.
2. If there's a mistake, give a short explanation in English and provide the correction.

Use this format:
[Bot reply in {session_state['target_language']}]
[Mistake] (optional)
[Correction] (optional)
"""

    session_state["messages"].append(SystemMessage(content=correction_prompt))
    reply = llm.invoke(session_state["messages"])
    session_state["messages"].append(AIMessage(content=reply.content))

    reply_lines = reply.content.strip().split("\n")
    bot_reply = reply_lines[0]
    correction_info = "\n".join(reply_lines[1:]).strip()

    if correction_info:
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO mistakes (user_input, mistake_type, correction, timestamp)
            VALUES (?, ?, ?, ?)
        """, (user_input, "General", correction_info, timestamp))
        conn.commit()

    full_bot_reply = bot_reply
    if correction_info:
        full_bot_reply += f"\n\n🔧 {correction_info}"

    chat_history.append([user_input, full_bot_reply])
    return chat_history, gr.update()

def show_summary():
    try:
        thread_safe_conn = sqlite3.connect(db_path, check_same_thread=False)
        df = pd.read_sql_query("SELECT * FROM mistakes", thread_safe_conn)
        thread_safe_conn.close()

        if df.empty:
            return "✅ No mistakes found in this session. Great job!"
        
        summary = "📚 Summary of Mistakes:\n\n"
        for i, row in df.iterrows():
            summary += f"{i+1}. ❌ '{row['user_input']}'\n"
            summary += f"   🔧 Correction: {row['correction']}\n"
            summary += f"   🕒 {row['timestamp']}\n\n"
        
        summary += f"📌 Total Mistakes: {len(df)}\n"
        summary += "🧠 Tip: Focus on sentence structure, vocabulary, and grammar usage."
        return summary

    except Exception as e:
        return f"❌ Error while generating summary: {str(e)}"

# 🎛️ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🌍 Multilingual Language Learning Chatbot")

    with gr.Accordion("Set Up Your Learning Profile", open=True):
        known_lang = gr.Textbox(label="What is your native/known language?", placeholder="e.g., English")
        target_lang = gr.Textbox(label="Which language do you want to learn?", placeholder="e.g., Spanish")
        level = gr.Dropdown(["Beginner", "Intermediate", "Advanced"], label="Your proficiency level")
        scene = gr.Radio(scenes, label="Choose a practice scene")
        start_btn = gr.Button("Start Conversation")
        profile_output = gr.Textbox(label="Bot's First Message")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message", placeholder="Type here and press Enter...")
    send_btn = gr.Button("Send")
    summary_btn = gr.Button("Show Mistake Summary")
    summary_output = gr.Textbox(label="Session Summary")

    msg.interactive = False
    send_btn.interactive = False

    def start_chat(known, target, lvl, scn):
        return setup_profile(known, target, lvl, scn)

    start_btn.click(fn=start_chat, inputs=[known_lang, target_lang, level, scene], outputs=[profile_output, msg, send_btn])
    send_btn.click(handle_message, inputs=[msg, chatbot], outputs=[chatbot, msg])
    msg.submit(handle_message, inputs=[msg, chatbot], outputs=[chatbot, msg])
    summary_btn.click(show_summary, outputs=[summary_output])

demo.launch()
