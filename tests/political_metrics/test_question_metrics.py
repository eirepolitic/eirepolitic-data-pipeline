import unittest

import pandas as pd

from political_metrics.calculators.questions import (
    grouped_question_metrics,
    member_question_metrics,
    prepare_eligible_td_questions,
    question_type_distribution,
    recipient_distribution,
)


class QuestionMetricTests(unittest.TestCase):
    def _histories(self):
        memberships = pd.DataFrame({
            "member_code": ["m1", "m2"],
            "membership_id": ["mem1", "mem2"],
            "house_uri": ["house:dail:34", "house:dail:34"],
            "house_no": ["34", "34"],
            "chamber": ["dail", "dail"],
            "membership_start": ["2025-01-01", "2025-01-01"],
            "membership_end": [None, None],
        })
        parties = pd.DataFrame({
            "member_code": ["m1", "m1", "m2"],
            "party_start": ["2025-01-01", "2026-07-15", "2025-01-01"],
            "party_end": ["2026-07-15", None, None],
            "party_uri": ["party:A", "party:B", "party:A"],
            "party_name": ["Party A", "Party B", "Party A"],
        })
        constituencies = pd.DataFrame({
            "member_code": ["m1", "m2"],
            "represent_start": ["2025-01-01", "2025-01-01"],
            "represent_end": [None, None],
            "constituency_uri": ["const:C1", "const:C2"],
            "constituency_name": ["C1", "C2"],
        })
        return memberships, parties, constituencies

    def test_questions_use_party_on_question_date(self):
        memberships, parties, constituencies = self._histories()
        questions = pd.DataFrame({
            "question_id": ["q1", "q2"],
            "question_date": ["2026-07-10", "2026-07-20"],
            "asked_by_member_code": ["m1", "m1"],
            "question_type": ["Written", "Written"],
            "to_minister_or_department": ["Health", "Education"],
        })
        result = prepare_eligible_td_questions(questions, memberships, parties, constituencies)
        self.assertEqual(result["party_uri"].tolist(), ["party:A", "party:B"])

    def test_member_question_metrics_count_days_types_and_recipients(self):
        questions = pd.DataFrame({
            "question_id": ["q1", "q2", "q3"],
            "question_date": ["2026-07-01", "2026-07-01", "2026-07-02"],
            "member_code": ["m1", "m1", "m1"],
            "question_type": ["Written", "Written", "Oral"],
            "to_minister_or_department": ["Health", "Education", "Health"],
        })
        row = member_question_metrics(questions).iloc[0]
        self.assertEqual(row["question_count"], 3)
        self.assertEqual(row["question_day_count"], 2)
        self.assertEqual(row["question_type_count"], 2)
        self.assertEqual(row["recipient_count"], 2)

    def test_grouped_question_metrics(self):
        questions = pd.DataFrame({
            "question_id": ["q1", "q2", "q3"],
            "question_date": ["2026-07-01", "2026-07-01", "2026-07-02"],
            "member_code": ["m1", "m2", "m1"],
            "party_uri": ["A", "A", "A"],
            "question_type": ["Written", "Written", "Oral"],
            "to_minister_or_department": ["Health", "Health", "Education"],
        })
        row = grouped_question_metrics(questions, group_col="party_uri").iloc[0]
        self.assertEqual(row["question_count"], 3)
        self.assertEqual(row["asking_member_count"], 2)
        self.assertEqual(row["recipient_count"], 2)

    def test_question_type_share(self):
        questions = pd.DataFrame({
            "question_id": ["q1", "q2", "q3", "q4"],
            "question_type": ["Written", "Written", "Written", "Oral"],
        })
        result = question_type_distribution(questions).set_index("question_type")
        self.assertAlmostEqual(float(result.loc["Written", "question_type_share"]), 0.75)

    def test_recipient_distribution_by_party(self):
        questions = pd.DataFrame({
            "question_id": ["q1", "q2", "q3"],
            "party_uri": ["A", "A", "A"],
            "to_minister_or_department": ["Health", "Health", "Education"],
        })
        result = recipient_distribution(questions, group_col="party_uri").set_index("to_minister_or_department")
        self.assertAlmostEqual(float(result.loc["Health", "question_share"]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
