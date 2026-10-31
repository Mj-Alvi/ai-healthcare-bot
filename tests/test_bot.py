import unittest

from ai_engine import generate_bot_response, is_bangla, is_banglish
from knowledge_base import search_knowledge
from safety import find_emergency_flag


class SafetyTests(unittest.TestCase):
    def test_detects_breathing_emergency(self):
        self.assertEqual(find_emergency_flag("I am having trouble breathing"), "breathing")

    def test_detects_bangla_emergency(self):
        self.assertEqual(find_emergency_flag("আমার শ্বাসকষ্ট হচ্ছে"), "breathing")

    def test_detects_dengue_warning_sign(self):
        self.assertEqual(
            find_emergency_flag("I have severe abdominal pain"),
            "bleeding_or_dengue_warning",
        )

    def test_simple_english_negation_does_not_trigger(self):
        self.assertIsNone(find_emergency_flag("I do not have chest pain"))

    def test_unknown_message_has_no_emergency_flag(self):
        self.assertIsNone(find_emergency_flag("Can you explain photosynthesis?"))


class KnowledgeTests(unittest.TestCase):
    def test_dengue_returns_source(self):
        result = search_knowledge("What are dengue symptoms?", "English")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "dengue")
        self.assertIn("who.int", result["sources"][0]["url"])

    def test_bangla_dengue_answer(self):
        result = search_knowledge("ডেঙ্গুর লক্ষণ কী?", "বাংলা")
        self.assertIsNotNone(result)
        self.assertIn("ডেঙ্গু", result["answer"])

    def test_fever_returns_nhs_source(self):
        result = search_knowledge("I have a fever", "English")
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], "fever")
        self.assertIn("nhs.uk", result["sources"][0]["url"])

    def test_unsupported_question_returns_none(self):
        self.assertIsNone(search_knowledge("Can you tell me my blood group?", "English"))


class AIEngineTests(unittest.TestCase):
    def test_bangla_language_detector(self):
        self.assertTrue(is_bangla("আমার জ্বর হয়েছে"))
        self.assertFalse(is_bangla("I have a fever"))

    def test_banglish_detector(self):
        self.assertTrue(is_banglish("amar matha betha korche ki korbo"))
        self.assertFalse(is_banglish("what is the capital of France"))

    def test_ai_engine_banglish_emergency(self):
        res = generate_bot_response("amar buke prochondo betha shash nite parchi na")
        self.assertEqual(res["type"], "emergency")

    def test_ai_engine_bangla_emergency(self):
        res = generate_bot_response("আমার শ্বাস নিতে খুব কষ্ট হচ্ছে", "বাংলা")
        self.assertEqual(res["type"], "emergency")

    def test_ai_engine_curated_knowledge(self):
        res = generate_bot_response("What are dengue symptoms?", "English")
        self.assertEqual(res["type"], "curated_kb")
        self.assertIn("Dengue", res["answer"])

    def test_ai_engine_dialogue_retrieval(self):
        res = generate_bot_response("How to reduce weight with hypothyroidism?", "English")
        self.assertEqual(res["type"], "dialogue_retrieval")
        self.assertIsNotNone(res["answer"])

    def test_ai_engine_identity_banglish(self):
        res = generate_bot_response("tumi ke")
        self.assertEqual(res["type"], "identity")
        self.assertIn("Alvi", res["answer"])
        self.assertNotIn("Gemini", res["answer"])

    def test_ai_engine_identity_english(self):
        res = generate_bot_response("Who are you?", "English")
        self.assertEqual(res["type"], "identity")
        self.assertIn("Alvi", res["answer"])
        self.assertNotIn("Gemini", res["answer"])

    def test_ai_engine_identity_bangla_model(self):
        res = generate_bot_response("kon model eta")
        self.assertEqual(res["type"], "identity")
        self.assertIn("Alvi", res["answer"])

    def test_ai_engine_identity_developer(self):
        res = generate_bot_response("kar dara developed")
        self.assertEqual(res["type"], "identity")
        self.assertIn("Alvi", res["answer"])

    def test_ai_engine_unsupported_abstain(self):
        res = generate_bot_response("Can you teach me how to bake chocolate cookies?", "English")
        self.assertEqual(res["type"], "unsupported")


if __name__ == "__main__":
    unittest.main()
