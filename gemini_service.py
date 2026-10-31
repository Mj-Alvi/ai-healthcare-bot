"""
Gemini API Service for Healthcare Assistant
===========================================
Provides fallback generative AI answering when a user question is NOT covered
by the local curated knowledge base, symptom predictor, or dialogue dataset.

Safety Guardrails:
- Strictly educational and informative
- Explicit diagnostic and prescriptive disclaimer
- Responds in the user's selected language (English or বাংলা)
"""

import os
import json
import urllib.request
import urllib.error

SYSTEM_PROMPT_EN = (
    "You are a helpful, empathetic, educational healthcare assistant. "
    "Your duty is to provide clear, helpful, evidence-based general medical and health information. "
    "Note: The user may ask in English or Banglish (Bengali transliterated in English letters, e.g. 'amar pete betha', 'ki korbo'). "
    "Accurately interpret their medical intent and respond in English.\n"
    "CRITICAL RULES:\n"
    "1. Start directly with the answer to the user's question without unnecessary meta-preamble.\n"
    "2. Never claim to make a definitive medical diagnosis.\n"
    "3. Never prescribe specific drug dosages or individualized treatment plans.\n"
    "4. Always advise consulting a qualified clinician for personal symptoms.\n"
    "5. If symptoms sound severe, sudden, or life-threatening, urge immediate emergency medical care.\n"
    "6. Respond clearly in English."
)

SYSTEM_PROMPT_BN = (
    "আপনি একজন আন্তরিক, নির্ভরযোগ্য ও শিক্ষামূলক স্বাস্থ্য সহকারী। "
    "ব্যবহারকারী বাংলা হরফে কিংবা বাংলিশে (ইংরেজি হরফে বাংলা, যেমন: 'amar matha betha korche', 'ghumer jonno ki khabo', 'ki korbo') প্রশ্ন করতে পারেন। "
    "বাংলিশ ও বাংলা উভয় ভাষার প্রশ্ন সঠিকভাবে বুঝে সুন্দর, প্রাঞ্জল ও আকর্ষণীয় বাংলায় উত্তর দিন।\n"
    "নির্দেশিকা:\n"
    "১. কোনো প্রকার ভূমিকা বা ভণিতা ছাড়া সরাসরি প্রশ্নের উত্তর দেওয়া শুরু করুন।\n"
    "২. কোনো সুনির্দিষ্ট ওষুধের মাত্রা বা নিশ্চিত রোগ নির্ণয় করবেন না।\n"
    "৩. প্রয়োজন অনুযায়ী চিকিৎসকের সরাসরি পরামর্শ নেওয়ার কথা স্মরণ করিয়ে দিন।\n"
    "৪. লক্ষণ গুরুতর বা জরুরি মনে হলে অবিলম্বে নিকটস্থ হাসপাতাল বা জরুরি সেবা (৯৯৯) এ যোগাযোগ করতে বলুন।\n"
    "৫. উত্তরটি অবশ্যই প্রাঞ্জল ও আকর্ষণীয় বাংলায় দিন।"
)


def _get_gemini_api_key(explicit_key: str = None) -> str:
    """Securely resolve the Gemini API key from environment, .env, or Streamlit secrets."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    # 1. Check process environment
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 2. Check local .env file (hidden from git & browser)
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except Exception:
            pass

    # 3. Check Streamlit native secrets (.streamlit/secrets.toml)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    return None


def ask_gemini_medical(query: str, language: str = "English", api_key: str = None) -> dict:
    """
    Call Gemini API to answer an open-ended medical question not covered in local datasets.
    Uses direct REST with multiple model fallbacks for maximum speed and reliability.
    """
    resolved_key = _get_gemini_api_key(api_key)
    if not resolved_key:
        return {
            "success": False,
            "answer": None,
            "error": "no_key",
            "model_name": None
        }

    sys_instruction = SYSTEM_PROMPT_BN if language == "বাংলা" else SYSTEM_PROMPT_EN
    
    # gemini-3.5-flash-lite is proven active and responsive on this key; fallback to gemini-3.8-flash
    candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-lite-latest"]
    last_error = None

    for model_name in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={resolved_key}"
        
        payload = {
            "system_instruction": {
                "parts": [{"text": sys_instruction}]
            },
            "contents": [
                {
                    "parts": [{"text": query}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3
            }
        }
        
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=20) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            return {
                                "success": True,
                                "answer": parts[0]["text"].strip(),
                                "error": None,
                                "model_name": model_name
                            }
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
                last_error = f"HTTP {e.code}: {err_body}"
            except Exception:
                last_error = f"HTTP {e.code}"
            continue
        except Exception as e:
            last_error = str(e)
            continue

    return {
        "success": False,
        "answer": None,
        "error": last_error or "Unknown Gemini API error",
        "model_name": None
    }
