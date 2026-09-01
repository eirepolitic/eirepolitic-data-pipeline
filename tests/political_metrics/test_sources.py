import unittest

import pandas as pd

from political_metrics.sources import canonical_speeches


class CanonicalSpeechSourceTests(unittest.TestCase):
    def test_speaker_member_code_is_exposed_as_member_code(self):
        source = pd.DataFrame({
            "speech_id": ["s1"],
            "debate_date": ["2026-01-10"],
            "speaker_member_code": ["m1"],
        })
        result = canonical_speeches(source)
        self.assertEqual(result.loc[0, "member_code"], "m1")

    def test_conflicting_existing_member_code_fails(self):
        source = pd.DataFrame({
            "speech_id": ["s1"],
            "debate_date": ["2026-01-10"],
            "speaker_member_code": ["m1"],
            "member_code": ["m2"],
        })
        with self.assertRaises(ValueError):
            canonical_speeches(source)

    def test_required_source_columns_are_enforced(self):
        source = pd.DataFrame({"speech_id": ["s1"]})
        with self.assertRaises(ValueError):
            canonical_speeches(source)


if __name__ == "__main__":
    unittest.main()
