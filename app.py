import streamlit as st

from ai_engine import generate_bot_response
from knowledge_base import search_knowledge
from safety import find_emergency_flag

st.set_page_config(
    page_title="Healthcare Assistant | স্বাস্থ্য সহায়ক",
    page_icon="🩺",
    layout="centered",
)

TEXT = {
    "English": {
        "title": "Healthcare Assistant",
        "subtitle": "A bilingual learning prototype for general health information.",
        "disclaimer_title": "Important safety notice",
        "disclaimer": (
            "This is an educational demo, not a doctor. It cannot diagnose conditions, "
            "recommend a personalised treatment plan, or rule out an emergency. "
            "Do not delay medical care to use this bot."
        ),
        "privacy": "Please do not enter your name, address, phone number, ID, or other identifying information.",
        "lang": "Response language",
        "placeholder": "Ask a general health question (for example: What are dengue symptoms?)",
        "intro": "Try asking about dengue, dengue warning signs, or fever in adults.",
        "clear": "Clear conversation",
        "sources": "Sources",
        "unknown_title": "I could not find a supported answer",
        "unknown": (
            "This demo only answers a small set of topics covered by its cited knowledge base. "
            "I should not guess about medical questions. Please consult a licensed healthcare professional. "
            "Seek urgent help if symptoms are severe, sudden, or getting worse."
        ),
        "emergency_title": "Possible emergency warning sign",
        "emergency": (
            "Your message contains a phrase that can be associated with a medical emergency. "
            "If this is happening now, seek emergency medical help immediately. If you are in Bangladesh, "
            "call **999** for emergency services; otherwise call your local emergency number. "
            "Do not wait for another chatbot response. Follow the emergency operator's instructions. "
            "This keyword check is imperfect and cannot assess or rule out an emergency."
        ),
        "kb_note": "This answer is retrieved from the demo's local knowledge base, not generated from a diagnosis.",
        "scope_title": "What this demo can do",
        "scope": [
            "Answer a few general questions using linked public health sources.",
            "Show a conservative warning when certain emergency phrases appear.",
            "Display answers in English or Bangla.",
        ],
        "not_scope_title": "What it cannot do",
        "not_scope": [
            "Diagnose a disease or estimate your personal risk.",
            "Recommend medicine doses or personalised treatment.",
            "Replace a doctor, clinic, or emergency service.",
        ],
    },
    "বাংলা": {
        "title": "স্বাস্থ্য সহায়ক",
        "subtitle": "সাধারণ স্বাস্থ্যতথ্যের জন্য একটি দ্বিভাষিক শিক্ষামূলক প্রোটোটাইপ।",
        "disclaimer_title": "গুরুত্বপূর্ণ নিরাপত্তা বার্তা",
        "disclaimer": (
            "এটি একটি শিক্ষামূলক ডেমো, ডাক্তার নয়। এটি রোগ নির্ণয় করতে, ব্যক্তিগত চিকিৎসা পরিকল্পনা দিতে, "
            "বা জরুরি অবস্থা নেই বলে নিশ্চিত করতে পারে না। এই বট ব্যবহার করতে গিয়ে চিকিৎসা নিতে দেরি করবেন না।"
        ),
        "privacy": "আপনার নাম, ঠিকানা, ফোন নম্বর, পরিচয়পত্র নম্বর বা অন্য কোনো পরিচয় শনাক্তকারী তথ্য লিখবেন না।",
        "lang": "উত্তরের ভাষা",
        "placeholder": "সাধারণ স্বাস্থ্য সম্পর্কে প্রশ্ন করুন (যেমন: ডেঙ্গুর লক্ষণ কী?)",
        "intro": "ডেঙ্গু, ডেঙ্গুর বিপদসংকেত বা প্রাপ্তবয়স্কদের জ্বর সম্পর্কে প্রশ্ন করতে পারেন।",
        "clear": "কথোপকথন মুছুন",
        "sources": "তথ্যসূত্র",
        "unknown_title": "সমর্থিত উত্তর খুঁজে পাইনি",
        "unknown": (
            "এই ডেমো কেবল তার সূত্রযুক্ত তথ্যভান্ডারে থাকা কয়েকটি বিষয় সম্পর্কে উত্তর দেয়। "
            "চিকিৎসাবিষয়ক প্রশ্নে অনুমান করে উত্তর দেওয়া ঠিক নয়। একজন নিবন্ধিত স্বাস্থ্য পেশাজীবীর পরামর্শ নিন। "
            "লক্ষণ গুরুতর, হঠাৎ শুরু হওয়া বা ক্রমে খারাপ হলে দ্রুত চিকিৎসা নিন।"
        ),
        "emergency_title": "সম্ভাব্য জরুরি অবস্থার সতর্কসংকেত",
        "emergency": (
            "আপনার বার্তায় এমন একটি শব্দ বা বাক্যাংশ আছে যা চিকিৎসাজনিত জরুরি অবস্থার সঙ্গে সম্পর্কিত হতে পারে। "
            "এটি যদি এখন ঘটছে, অবিলম্বে জরুরি চিকিৎসা সহায়তা নিন। বাংলাদেশে থাকলে জরুরি সেবার জন্য **৯৯৯** নম্বরে কল করুন; "
            "অন্য দেশে থাকলে সেখানকার স্থানীয় জরুরি নম্বরে কল করুন। বটের পরবর্তী উত্তরের জন্য অপেক্ষা করবেন না। "
            "জরুরি সেবার কর্মীর নির্দেশনা অনুসরণ করুন। এই শব্দভিত্তিক পরীক্ষা নিখুঁত নয় এবং জরুরি অবস্থা আছে বা নেই তা নিশ্চিত করতে পারে না।"
        ),
        "kb_note": "এই উত্তরটি ডেমোর স্থানীয় তথ্যভান্ডার থেকে নেওয়া হয়েছে; এটি রোগ নির্ণয়ের ফল নয়।",
        "scope_title": "এই বট যা করতে পারে",
        "scope": [
            "সূত্রযুক্ত জনস্বাস্থ্য তথ্য ব্যবহার করে সাধারণ প্রশ্নের উত্তর দিতে পারে।",
            "বাংলা বা ইংরেজিতে উত্তর দেখাতে পারে।",
        ],
        "not_scope_title": "এই বট যা করতে পারে না",
        "not_scope": [
            "রোগ নির্ণয় বা ব্যক্তিগত ঝুঁকি নির্ধারণ করতে পারে না।",
            "ওষুধের ডোজ বা ব্যক্তিগত চিকিৎসা পরিকল্পনা দিতে পারে না।",
            "ডাক্তার, ক্লিনিক বা জরুরি সেবার বিকল্প নয়।",
        ],
    },
}

st.title("🩺 " + "Healthcare Assistant | স্বাস্থ্য সহায়ক")

with st.sidebar:
    language = st.selectbox("Response language / উত্তরের ভাষা", ["Auto (স্বয়ংক্রিয়)", "বাংলা", "English"])
    ui_lang = "English" if language == "English" else "বাংলা"
    t = TEXT[ui_lang]
    st.subheader(t["scope_title"])
    for item in t["scope"]:
        st.write(f"• {item}")
    st.subheader(t["not_scope_title"])
    for item in t["not_scope"]:
        st.write(f"• {item}")
    if st.button(t["clear"], use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.caption(TEXT[ui_lang]["subtitle"])
st.warning(f"**{TEXT[ui_lang]['disclaimer_title']}**\n\n{TEXT[ui_lang]['disclaimer']}")
st.caption("🔒 " + TEXT[ui_lang]["privacy"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.messages:
    st.info(TEXT[ui_lang]["intro"])

question = st.chat_input(TEXT[ui_lang]["placeholder"])

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    bot_output = generate_bot_response(question, language)
    resp_lang = bot_output.get("language", ui_lang)
    if resp_lang not in TEXT:
        resp_lang = "বাংলা" if "বাংলা" in resp_lang else "English"
    
    if bot_output["type"] == "emergency":
        answer = f"**{TEXT[resp_lang]['emergency_title']}**\n\n{TEXT[resp_lang]['emergency']}"
        if bot_output.get("triage", {}).get("category"):
            answer += f"\n\n*Identified category:* `{bot_output['triage']['category']}`"
    elif bot_output["type"] == "unsupported":
        answer = f"**{TEXT[resp_lang]['unknown_title']}**\n\n{TEXT[resp_lang]['unknown']}"
        if bot_output.get("gemini_error") and bot_output["gemini_error"] != "no_key":
            answer += f"\n\n*(Notice: {bot_output['gemini_error']})*"
    else:
        answer = bot_output["answer"]
        if bot_output.get("sources"):
            answer += "\n\n**" + TEXT[resp_lang]["sources"] + ":**"
            for source in bot_output["sources"]:
                answer += f"\n- [{source['title']}]({source['url']})"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
