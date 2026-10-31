"""Conservative keyword-based emergency warning checks.

These checks are a simple safety net, not a medical triage model. They can miss
emergencies and can also trigger on general questions. Clinical review and much
stronger evaluation are required before any real-world use.
"""

EMERGENCY_PHRASES = {
    "breathing": [
        "trouble breathing", "difficulty breathing", "can't breathe", "cannot breathe",
        "shortness of breath", "severe breathlessness", "শ্বাসকষ্ট", "শ্বাস নিতে কষ্ট",
        "শ্বাস নিতে পারছি না", "shash koshto", "shash nite koshto", "shas kosto",
        "shash nite parchi na", "shash nite parchhi na", "shash bondho",
    ],
    "chest_pain": [
        "chest pain", "chest pressure", "বুকে ব্যথা", "বুকে চাপ",
        "buke betha", "buke batha", "buke chap", "buke prochondo betha", "chest betha"
    ],
    "stroke": [
        "face drooping", "facial droop", "slurred speech", "weakness on one side",
        "one-sided weakness", "মুখ বেঁকে", "কথা জড়িয়ে", "এক পাশ দুর্বল", "হাত-পা অবশ",
        "mukh beke", "kotha joriye", "ek pash obosh", "hat pa obosh"
    ],
    "consciousness_or_seizure": [
        "unconscious", "not responding", "passed out", "seizure", "convulsion",
        "অজ্ঞান", "সাড়া দিচ্ছে না", "খিঁচুনি", "oggan", "shara dicche na", "khichuni",
        "senseless", "gyan harano"
    ],
    "bleeding_or_dengue_warning": [
        "severe bleeding", "heavy bleeding", "vomiting blood", "blood in vomit", "blood in stool",
        "severe abdominal pain", "persistent vomiting", "rapid breathing",
        "cold and pale skin", "ঠান্ডা ফ্যাকাশে", "প্রচুর রক্তপাত", "অনেক রক্তপাত",
        "রক্ত বমি", "পায়খানায় রক্ত", "তীব্র পেট ব্যথা", "বারবার বমি", "দ্রুত শ্বাস",
        "rokto bomi", "paykhanay rokto", "tibro pet betha", "bar bar bomi", "druto shash"
    ],
    "severe_allergy_or_poisoning": [
        "swelling of the tongue", "tongue swelling", "swelling of the throat", "throat swelling",
        "anaphylaxis", "severe allergic reaction", "poisoning", "overdose",
        "জিহ্বা ফুলে", "গলা ফুলে", "বিষক্রিয়া", "বিষ খেয়েছে",
        "bish kheyeche", "bish kheyese", "gola fule", "jibba fule"
    ],
}

NEGATION_BEFORE = [
    "no ", "not ", "without ", "don't have ", "do not have ", "does not have ",
    "doesn't have ", "denies ", "not experiencing ", "নেই ", "নাই ", "nei ", "nai ",
]
NEGATION_AFTER = [" নেই", " নাই", " হচ্ছে না", " হয়নি", " হচ্ছে না।", " nei", " nai", " hocche na"]


def _normalise(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def _is_negated(text: str, start: int, end: int) -> bool:
    """Catch simple nearby negation. Complex phrasing needs human/clinical review."""
    before = text[max(0, start - 35):start]
    after = text[end:min(len(text), end + 18)]
    if any(marker in before for marker in NEGATION_BEFORE):
        return True
    if any(marker in after for marker in NEGATION_AFTER):
        return True
    return False


def find_emergency_flag(message: str):
    """Return the category of a matched warning phrase, otherwise None."""
    text = _normalise(message)
    for category, phrases in EMERGENCY_PHRASES.items():
        for phrase in phrases:
            phrase = _normalise(phrase)
            start = text.find(phrase)
            while start != -1:
                end = start + len(phrase)
                if not _is_negated(text, start, end):
                    return category
                start = text.find(phrase, end)
    return None
