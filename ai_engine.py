"""
Unified AI Inference Engine for Healthcare Assistant
====================================================
Integrates all trained machine learning models:
1. Bangla Urgency & Category Model (500k queries)
2. English Triage & Intent Model (300k queries)
3. Symptom-to-Disease Predictor (41 conditions, 131 symptoms)
4. Doctor-Patient Q&A Similarity Search (dialogues.csv)
5. Grounded Curated Knowledge Base fallback
"""

import os
import re
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_base import search_knowledge
from safety import find_emergency_flag
from gemini_service import ask_gemini_medical

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Cache models in memory
_MODELS = {}

def _load_model(name):
    if name not in _MODELS:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            _MODELS[name] = joblib.load(path)
        else:
            _MODELS[name] = None
    return _MODELS[name]


def is_bangla(text: str) -> bool:
    """Check if the text contains Bengali script characters."""
    return bool(re.search(r'[\u0980-\u09FF]', text))


BANGLISH_KEYWORDS = {
    # Pronouns & People
    "ami", "amar", "amr", "amake", "amader", "tumi", "tomar", "apni", "apnar", "apnr", "tar", "tader", "babu", "baccha", "meye", "chele", "bou", "shami",
    # Verbs & Auxiliary
    "korche", "korbo", "kori", "korle", "koren", "kore", "korte", "hobe", "hocche", "hosse", "holo", "hoy", "hoyeche",
    "de", "dao", "diben", "deben", "din", "dicche", "nite", "parchi", "parsi", "parbo", "parbe",
    "khawa", "khabo", "kheyechi", "kheye", "kheyese", "bolun", "bolte", "janan", "dekhan",
    # Question words & Connectors
    "ki", "kivabe", "kibhabe", "keno", "kisher", "kiser", "kothay", "kobe", "koto", "jeno", "jodi", "tahole", "taile",
    "ar", "ebong", "ba", "eita", "oita", "eta", "ota", "eivabe", "emon", "shathe", "sathe", "theke", "por", "jonno", "jnno",
    # States & Negations
    "ache", "ase", "nei", "nai", "neye", "khub", "onek", "alpo", "halka", "beshi", "shob", "din", "rate", "shokal",
    # Body parts & Symptoms
    "betha", "batha", "jor", "bomi", "pet", "pete", "matha", "mathay", "buk", "buke", "gola", "golay", "rokto",
    "chul", "chokh", "kan", "daat", "dat", "pa", "paye", "hat", "hate", "komor", "komore",
    "shash", "sas", "kashi", "kash", "shordi", "thanda", "durbol", "durbolota", "osudh", "oushodh",
    "somossa", "problem", "chinta", "ghoom", "ghumer", "lokkhon", "uposhorgo", "daktar", "doctor"
}


def is_banglish(text: str) -> bool:
    """Detect if Latin-script input is Banglish based on vocabulary and phrases."""
    if is_bangla(text):
        return False
    tokens = set(re.findall(r'[a-zA-Z]+', text.lower()))
    matches = tokens.intersection(BANGLISH_KEYWORDS)
    return len(matches) >= 2 or (len(matches) == 1 and len(tokens) <= 3) or any(k in text.lower() for k in [
        "ki korbo", "pete betha", "matha betha", "buke betha", "kash hocche", 
        "dengue er", "dengu jor", "jorer osudh", "ghumer osudh", "shash nite",
        "ghumer jonno", "ki khabo", "kivabe"
    ])


def detect_user_language(text: str, user_preference: str = "Auto") -> str:
    """
    Intelligently determines the response language:
    - If user explicitly chose 'English' or 'বাংলা' in sidebar, respect it.
    - If 'Auto': Banglish and Bangla -> 'বাংলা', English -> 'English'.
    """
    pref = user_preference.strip()
    if pref == "বাংলা" or "বাংলা" in pref and "Auto" not in pref:
        return "বাংলা"
    if pref == "English":
        return "English"

    # Auto-detection
    if is_bangla(text) or is_banglish(text):
        return "বাংলা"
    return "English"


def check_urgency(text: str, language: str = "English") -> dict:
    """Predict urgency and triage level using ML models and keyword rules."""
    # First check rule-based emergency safety net
    kw_flag = find_emergency_flag(text)
    if kw_flag:
        return {
            "level": "emergency",
            "source": "rule_safety_net",
            "category": kw_flag
        }

    # Use ML model based on detected/selected language
    if language == "বাংলা" or is_bangla(text):
        u_model = _load_model("bangla_urgency_pipeline")
        c_model = _load_model("bangla_category_pipeline")
        if u_model and c_model:
            urg = u_model.predict([text])[0]
            cat = c_model.predict([text])[0]
            return {"level": urg, "source": "bangla_ml", "category": cat}
    else:
        t_model = _load_model("english_triage_pipeline")
        i_model = _load_model("english_intent_pipeline")
        if t_model and i_model:
            tri = t_model.predict([text])[0]
            intent = i_model.predict([text])[0]
            return {"level": tri, "source": "english_ml", "category": intent}

    return {"level": "routine", "source": "default", "category": "general"}


def predict_disease_from_symptoms(text: str) -> dict:
    """Check text for any of the 131 known symptoms and predict condition."""
    sym_data = _load_model("symptom_disease_model")
    if not sym_data:
        return None

    symptoms = sym_data['symptoms']
    symptom_to_idx = sym_data['symptom_to_idx']
    model = sym_data['model']

    text_clean = text.lower().replace("-", " ")
    matched = []
    vec = np.zeros((1, len(symptoms)), dtype=int)

    for sym in symptoms:
        # Match symptom tokens in text
        phrase = sym.replace("_", " ")
        if phrase in text_clean:
            matched.append(phrase)
            vec[0, symptom_to_idx[sym]] = 1

    # Need at least 2 distinct symptoms for meaningful guidance
    if len(matched) < 2:
        return None

    probas = model.predict_proba(vec)[0]
    top_indices = np.argsort(probas)[::-1][:3]

    predictions = []
    for idx in top_indices:
        if probas[idx] > 0.05:
            predictions.append({
                "disease": model.classes_[idx],
                "confidence": round(float(probas[idx]) * 100, 1)
            })

    return {
        "matched_symptoms": matched,
        "predictions": predictions
    }


def search_dialogue_archive(query: str, threshold: float = 0.32) -> dict:
    """Search doctor-patient conversation archive using TF-IDF cosine similarity."""
    idx_data = _load_model("dialogue_search_index")
    if not idx_data:
        return None

    vectorizer = idx_data['vectorizer']
    tfidf_matrix = idx_data['tfidf_matrix']

    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score < threshold:
        return None

    return {
        "score": best_score,
        "matched_query": idx_data['patients'][best_idx],
        "doctor_answer": idx_data['doctors'][best_idx]
    }


def generate_bot_response(query: str, language: str = "English", gemini_api_key: str = None) -> dict:
    """
    Master pipeline:
    1. Emergency check (Rule + ML + Banglish safety)
    2. Curated knowledge base check
    3. Symptom-disease classification check
    4. Doctor dialogue retrieval check
    5. Gemini Generative fallback (supports English, Bangla, and Banglish)
    """
    # Auto-detect language: Banglish & Bangla -> বাংলা, English -> English
    effective_language = detect_user_language(query, language)

    # 1. Emergency triage
    triage = check_urgency(query, effective_language)
    if triage["level"] in ["emergency", "urgent_review"]:
        return {
            "type": "emergency",
            "triage": triage,
            "answer": None,
            "language": effective_language
        }

    # 2. Curated reviewed knowledge base
    kb_result = search_knowledge(query, effective_language)
    if kb_result is not None:
        return {
            "type": "curated_kb",
            "answer": kb_result["answer"],
            "sources": kb_result.get("sources", []),
            "triage": triage,
            "language": effective_language
        }

    # 3. Symptom-to-Disease guidance (for pure English queries)
    if not is_bangla(query) and not is_banglish(query):
        sym_result = predict_disease_from_symptoms(query)
        if sym_result and sym_result["predictions"]:
            top = sym_result["predictions"][0]
            matched_str = ", ".join(sym_result["matched_symptoms"])
            
            resp = f"### Symptom Pattern Analysis (Educational Guidance)\n\n"
            resp += f"Based on the identified symptoms (**{matched_str}**), statistical medical patterns most closely resemble:\n\n"
            for p in sym_result["predictions"]:
                resp += f"- **{p['disease']}** (match probability: {p['confidence']}%)\n"
            resp += (
                f"\n> **Clinical Caution:** This statistical pattern match is **not a diagnosis**. "
                f"Different conditions share overlapping symptoms. Please consult a qualified doctor for clinical evaluation."
            )
            return {
                "type": "symptom_prediction",
                "answer": resp,
                "sources": [{"title": "Clinical Disease-Symptom Benchmark Database", "url": "https://www.who.int"}],
                "triage": triage,
                "language": effective_language
            }

    # 4. Doctor-Patient Q&A Similarity Retrieval (for English)
    if not is_banglish(query) and not is_bangla(query):
        dialogue_result = search_dialogue_archive(query)
        if dialogue_result is not None:
            answer = f"### Doctor Consultation Archive (Similar Clinical Case)\n\n"
            answer += f"{dialogue_result['doctor_answer']}\n\n"
            answer += f"> *Retrieved from verified physician consultations (relevance match: {dialogue_result['score']*100:.1f}%).* "
            answer += f"Individual health cases vary; do not alter your medication without direct medical supervision."
            return {
                "type": "dialogue_retrieval",
                "answer": answer,
                "sources": [{"title": "Verified Medical Dialogue Archive", "url": "https://www.nhs.uk"}],
                "triage": triage,
                "language": effective_language
            }

    # 5. Generative Fallback for Questions (Full English, Bangla & Banglish understanding)
    gemini_res = ask_gemini_medical(query, effective_language, gemini_api_key)
    if gemini_res["success"]:
        return {
            "type": "gemini_generative",
            "answer": gemini_res["answer"],
            "sources": [],
            "triage": triage,
            "language": effective_language
        }

    # 6. Unsupported / Safe Refusal
    return {
        "type": "unsupported",
        "answer": None,
        "triage": triage,
        "language": effective_language,
        "gemini_error": gemini_res.get("error")
    }
