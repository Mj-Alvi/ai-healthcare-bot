"""Tiny curated knowledge base for the healthcare assistant MVP.

Keep answers short, reviewable, and linked to reliable sources. Do not add
medical content without checking its source and having it reviewed.
"""

KNOWLEDGE = [
    {
        "id": "dengue",
        "keywords": [
            "dengue", "dengue fever", "dengu", "ডেঙ্গু", "ডেঙ্গু জ্বর", "dengue symptoms",
            "symptoms of dengue", "ডেঙ্গুর লক্ষণ", "ডেঙ্গু হলে", "ডেঙ্গুর উপসর্গ",
            "dengur lokkhon", "dengu jor", "dengue hole ki hoy",
        ],
        "answer_en": (
            "### Dengue: general information\n\n"
            "Dengue is a viral infection spread by infected mosquitoes. Some people have no symptoms; "
            "others may have fever, headache, pain behind the eyes, muscle or joint pain, nausea, vomiting, or a rash. "
            "These symptoms do not prove that someone has dengue. A healthcare professional can assess symptoms and decide whether testing is needed.\n\n"
            "**Warning signs that need immediate medical care** include severe abdominal pain, persistent vomiting, "
            "rapid breathing, bleeding from the nose or gums, blood in vomit or stool, unusual tiredness or restlessness, "
            "marked thirst, or pale and cold skin. Warning signs can appear even when the fever is going down.\n\n"
            "WHO advises rest and fluids for people with dengue and says aspirin and ibuprofen should be avoided because "
            "they can increase bleeding risk. This bot cannot decide which medicine is suitable for you. Contact a clinician for personal advice."
        ),
        "answer_bn": (
            "### ডেঙ্গু: সাধারণ তথ্য\n\n"
            "ডেঙ্গু একটি ভাইরাসজনিত সংক্রমণ, যা আক্রান্ত মশার কামড়ে ছড়ায়। কারও কোনো লক্ষণ নাও থাকতে পারে; "
            "আবার জ্বর, মাথাব্যথা, চোখের পেছনে ব্যথা, পেশি বা জয়েন্টে ব্যথা, বমিভাব, বমি বা ত্বকে র‍্যাশ হতে পারে। "
            "শুধু এসব লক্ষণ দেখে ডেঙ্গু নিশ্চিত করা যায় না। স্বাস্থ্য পেশাজীবী লক্ষণ মূল্যায়ন করে পরীক্ষার প্রয়োজন আছে কি না নির্ধারণ করতে পারেন।\n\n"
            "**যেসব বিপদসংকেতে অবিলম্বে চিকিৎসা দরকার:** তীব্র পেটব্যথা, বারবার বমি, দ্রুত শ্বাস, নাক বা মাড়ি দিয়ে রক্তপাত, "
            "বমি বা পায়খানায় রক্ত, অস্বাভাবিক ক্লান্তি বা অস্থিরতা, অতিরিক্ত তৃষ্ণা, অথবা ত্বক ফ্যাকাশে ও ঠান্ডা হয়ে যাওয়া। "
            "জ্বর কমতে থাকলেও বিপদসংকেত দেখা দিতে পারে।\n\n"
            "WHO ডেঙ্গু হলে বিশ্রাম ও পর্যাপ্ত তরল গ্রহণের পরামর্শ দেয় এবং রক্তপাতের ঝুঁকি বাড়তে পারে বলে অ্যাসপিরিন ও আইবুপ্রোফেন এড়াতে বলে। "
            "আপনার জন্য কোন ওষুধ উপযুক্ত, এই বট তা নির্ধারণ করতে পারে না। ব্যক্তিগত পরামর্শের জন্য চিকিৎসকের সঙ্গে যোগাযোগ করুন।"
        ),
        "sources": [
            {
                "title": "WHO: Dengue and severe dengue",
                "url": "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
            }
        ],
    },
    {
        "id": "fever",
        "keywords": [
            "fever", "high temperature", "temperature", "chills", "জ্বর", "উচ্চ তাপমাত্রা", "গা গরম", "কাঁপুনি",
            "jor", "ga gorom", "kapuni", "jorer osudh", "high fever",
        ],
        "answer_en": (
            "### Fever: general information\n\n"
            "Fever can have many causes, so this bot cannot identify the cause from a message. For general self-care, "
            "the NHS advises rest and drinking fluids to help avoid dehydration. Seek advice from a healthcare professional "
            "if the fever is not improving, is getting worse, or you are worried. Do not rely on this demo to assess a child, "
            "pregnant person, or someone with a serious underlying condition.\n\n"
            "If fever occurs with a possible emergency warning sign, seek urgent medical help."
        ),
        "answer_bn": (
            "### জ্বর: সাধারণ তথ্য\n\n"
            "জ্বরের অনেক কারণ থাকতে পারে, তাই শুধু বার্তা দেখে এই বট কারণ নির্ধারণ করতে পারে না। সাধারণ যত্নের জন্য NHS বিশ্রাম নেওয়া "
            "এবং পানিশূন্যতা এড়াতে পর্যাপ্ত তরল পান করার পরামর্শ দেয়। জ্বর না কমলে, খারাপ হলে বা আপনি উদ্বিগ্ন হলে স্বাস্থ্য পেশাজীবীর পরামর্শ নিন। "
            "শিশু, গর্ভবতী ব্যক্তি বা গুরুতর পূর্ববর্তী অসুস্থতা আছে এমন কারও অবস্থা মূল্যায়নে এই ডেমোর ওপর নির্ভর করবেন না।\n\n"
            "জ্বরের সঙ্গে সম্ভাব্য জরুরি বিপদসংকেত থাকলে দ্রুত চিকিৎসা সহায়তা নিন।"
        ),
        "sources": [
            {
                "title": "NHS: High temperature (fever) in adults",
                "url": "https://www.nhs.uk/symptoms/fever-in-adults/",
            }
        ],
    },
]


def _normalise(text: str) -> str:
    """Normalise case and whitespace while preserving Bangla characters."""
    return " ".join(text.casefold().strip().split())


def search_knowledge(question: str, language: str = "English"):
    """Return the best matching curated entry, or None if evidence is too weak.

    This intentionally uses transparent keyword matching rather than pretending
    to understand every question. It is not a clinical decision system.
    """
    text = _normalise(question)
    if not text:
        return None

    best_entry = None
    best_score = 0
    for entry in KNOWLEDGE:
        score = 0
        for keyword in entry["keywords"]:
            key = _normalise(keyword)
            if key and key in text:
                # Exact, multi-word topics count a little more than short terms.
                score += 2 if " " in key or len(key) > 8 else 1
        if score > best_score:
            best_score = score
            best_entry = entry

    # Require at least one meaningful match.
    if best_entry is None or best_score < 1:
        return None

    answer_key = "answer_bn" if language == "বাংলা" else "answer_en"
    return {
        "id": best_entry["id"],
        "answer": best_entry[answer_key],
        "sources": best_entry["sources"],
        "score": best_score,
    }
