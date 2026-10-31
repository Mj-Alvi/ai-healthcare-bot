# Healthcare Assistant (English + Bangla)

A beginner-friendly Streamlit web-app prototype for general health information and conservative emergency phrase warnings.

> **Medical safety:** This is a learning project, not a medical device and not a substitute for a qualified clinician. Do not use it for diagnosis, prescribing, or real patient triage. Its phrase rules can miss emergencies and can trigger on general questions. Get clinical review and safety evaluation before any real-world use.

## What is included

- English and Bangla response language selection.
- A small, source-linked local knowledge base covering dengue and fever.
- Simple transparent keyword matching. Unknown topics receive a refusal-to-guess response.
- A separate warning layer for selected possible emergency signs.
- No external AI API, no database, and no conversation upload. The chat history only remains in the current Streamlit session.
- Unit tests for basic content retrieval and safety phrase rules.

## Architecture

```text
Browser
  |
  v
Streamlit UI (app.py)
  |
  +--> safety.py --------- possible emergency phrase? --> emergency warning
  |
  +--> knowledge_base.py - find a matching reviewed topic --> answer + source link
  |                         no confident match ------------> do not guess
  v
Browser displays answer and source
```

The first version is intentionally retrieval-based rather than a free-form generative AI. This makes the answers easier for a beginner to inspect and source-check. Keyword matching is not a clinical triage method and is not robust natural-language understanding.

## Setup on Windows

1. Install Python 3.11 or 3.12 from <https://www.python.org/downloads/>. During installation, enable **Add Python to PATH**.
2. Extract the ZIP file and open PowerShell in the `healthcare_medical_bot` folder.
3. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install the packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the app:

   ```powershell
   streamlit run app.py
   ```

6. Streamlit will show a local URL, usually `http://localhost:8501`. Open it in your browser.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Test it

Run tests from the project folder after installing Python:

```bash
python -m unittest discover -s tests -v
```

Try these demo messages:

- `What are dengue symptoms?`
- `ডেঙ্গুর লক্ষণ কী?`
- `I have a fever`
- `আমার শ্বাসকষ্ট হচ্ছে` (should display an emergency warning)
- `Can you tell me my blood group?` (should not guess)

The emergency test is intentionally conservative, but it does **not** prove that the bot can safely detect or triage emergencies.

## Where to edit the bot

- Add/review public-health topics in `knowledge_base.py`. Each answer should have a credible source URL. Do not add treatment claims without expert review.
- Add carefully evaluated emergency phrases in `safety.py`. Do not treat this list as complete.
- Change labels, warning messages, and user-interface elements in `app.py`.
- Add a test for every new content item or safety rule in `tests/test_bot.py`.

## Testing and safety plan before showing this to real users

1. **Functional tests:** test each supported English and Bangla question, unknown questions, blank input, and source links.
2. **Emergency tests:** test direct descriptions, misspellings, Bangla spelling variants, negation, third-person reports, and educational questions. Have a clinician review the alert list. Do not rely on keyword matching as the only safety barrier.
3. **Clinical review:** ask a qualified clinician to review every health answer, red flag, translation, and escalation instruction.
4. **Coverage limits:** make the bot clearly state which topics it supports. It must not invent answers when no trusted content matches.
5. **Privacy/security:** collect no unnecessary personal information, don't log symptom text, protect any future stored data, and review local law before deploying.
6. **Usability and accessibility:** test on phone and desktop; ask Bangla and English speakers to review wording.
7. **Pre-deployment review:** use a written test set, measure missed emergency warnings as well as false alarms, document failures, and do not deploy for patient use until appropriate clinical, legal, security, and regulatory review is complete.

## Trusted sources used in this starter

- WHO, Dengue and severe dengue: <https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue>
- NHS, High temperature (fever) in adults: <https://www.nhs.uk/symptoms/fever-in-adults/>
- Bangladesh National Emergency Service 999: <https://telecom-police.portal.gov.bd/pages/static-pages/695e3b0cc4774958d7b72321>
- WHO, Ethics and governance of AI for health: <https://www.who.int/publications/i/item/9789240084759>

## Suggested next stages

1. Expand the reviewed bilingual knowledge base using a documented editorial process.
2. Measure how well retrieval finds the right passage for both English and Bangla.
3. Only after that, consider adding a language model with retrieval-augmented generation (RAG), strict source-grounding, output checks, and human clinical review. A prompt saying "do not hallucinate" is not enough to make a medical bot safe.
4. Keep symptom guidance separate from diagnosis. Any future triage feature should be designed and validated with qualified clinicians.
