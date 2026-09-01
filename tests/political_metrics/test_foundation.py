import unittest
from datetime import date

import pandas as pd

from political_metrics.calculators.speeches import member_speech_metrics, national_speech_metrics
from political_metrics.eligibility import member_debate_day_exposure
from political_metrics.periods import resolve_period
from political_metrics.temporal_joins import attach_event_constituency, attach_event_party
from political_metrics.validators import validate_temporal_history


class PeriodTests(unittest.TestCase):
    def test_calendar_month(self):
        period = resolve_period("2026-07")
        self.assertEqual(period.start, date(2026, 7, 1))
        self.assertEqual(period.end, date(2026, 7, 31))

    def test_last_completed_month(self):
        period = resolve_period("last_completed_month", today="2026-08-31")
        self.assertEqual(period.start, date(2026, 7, 1))
        self.assertEqual(period.end, date(2026, 7, 31))

    def test_rolling_period_is_inclusive(self):
        period = resolve_period("rolling_7d", today="2026-08-31")
        self.assertEqual(period.start, date(2026, 8, 25))
        self.assertEqual(period.end, date(2026, 8, 31))


class TemporalJoinTests(unittest.TestCase):
    def test_party_switch_is_period_correct(self):
        events = pd.DataFrame({
            "speech_id": ["s1", "s2"],
            "member_code": ["m1", "m1"],
            "event_date": ["2026-01-10", "2026-01-20"],
        })
        history = pd.DataFrame({
            "member_code": ["m1", "m1"],
            "party_start": ["2025-01-01", "2026-01-15"],
            "party_end": ["2026-01-14", None],
            "party_uri": ["party:A", "party:B"],
            "party_name": ["Party A", "Party B"],
        })
        joined = attach_event_party(events, history)
        self.assertEqual(joined["party_uri"].tolist(), ["party:A", "party:B"])

    def test_constituency_switch_is_period_correct(self):
        events = pd.DataFrame({
            "speech_id": ["s1", "s2"],
            "member_code": ["m1", "m1"],
            "event_date": ["2026-01-10", "2026-01-20"],
        })
        history = pd.DataFrame({
            "member_code": ["m1", "m1"],
            "represent_start": ["2025-01-01", "2026-01-15"],
            "represent_end": ["2026-01-14", None],
            "constituency_uri": ["constituency:C1", "constituency:C2"],
            "constituency_name": ["Old", "New"],
        })
        joined = attach_event_constituency(events, history)
        self.assertEqual(joined["constituency_uri"].tolist(), ["constituency:C1", "constituency:C2"])

    def test_overlapping_party_history_raises(self):
        events = pd.DataFrame({
            "speech_id": ["s1"],
            "member_code": ["m1"],
            "event_date": ["2026-01-15"],
        })
        history = pd.DataFrame({
            "member_code": ["m1", "m1"],
            "party_start": ["2025-01-01", "2026-01-01"],
            "party_end": [None, None],
            "party_uri": ["party:A", "party:B"],
            "party_name": ["Party A", "Party B"],
        })
        with self.assertRaises(ValueError):
            attach_event_party(events, history)


class ExposureTests(unittest.TestCase):
    def test_partial_period_member_only_gets_eligible_days(self):
        memberships = pd.DataFrame({
            "member_code": ["m1"],
            "membership_start": ["2026-01-15"],
            "membership_end": [None],
        })
        debate_days = ["2026-01-10", "2026-01-20", "2026-01-21"]
        exposure = member_debate_day_exposure(memberships, debate_days)
        self.assertEqual(exposure.loc[0, "eligible_debate_days"], 2)


class SpeechMetricTests(unittest.TestCase):
    def test_member_counts_and_rates(self):
        speeches = pd.DataFrame({
            "speech_id": ["s1", "s2", "s3"],
            "member_code": ["m1", "m1", "m2"],
            "debate_date": ["2026-01-10", "2026-01-10", "2026-01-11"],
        })
        exposure = pd.DataFrame({
            "member_code": ["m1", "m2"],
            "eligible_debate_days": [2, 2],
        })
        result = member_speech_metrics(speeches, exposure).set_index("member_code")
        self.assertEqual(result.loc["m1", "speech_count"], 2)
        self.assertEqual(result.loc["m1", "speaking_day_count"], 1)
        self.assertAlmostEqual(float(result.loc["m1", "share_of_dail_speeches"]), 2 / 3)
        self.assertAlmostEqual(float(result.loc["m1", "speeches_per_eligible_debate_day"]), 1.0)

    def test_eligible_member_with_no_speeches_is_zero_not_missing(self):
        speeches = pd.DataFrame({
            "speech_id": ["s1"],
            "member_code": ["m1"],
            "debate_date": ["2026-01-10"],
        })
        exposure = pd.DataFrame({
            "member_code": ["m1", "m2"],
            "eligible_debate_days": [1, 1],
        })
        result = member_speech_metrics(speeches, exposure).set_index("member_code")
        self.assertEqual(result.loc["m2", "speech_count"], 0)
        self.assertEqual(result.loc["m2", "speaking_day_count"], 0)
        self.assertEqual(float(result.loc["m2", "share_of_dail_speeches"]), 0.0)

    def test_national_metrics(self):
        speeches = pd.DataFrame({
            "speech_id": ["s1", "s2", "s3"],
            "member_code": ["m1", "m1", "m2"],
            "debate_date": ["2026-01-10", "2026-01-10", "2026-01-11"],
        })
        result = national_speech_metrics(speeches)
        self.assertEqual(result["speech_count"], 3)
        self.assertEqual(result["unique_speaker_count"], 2)
        self.assertEqual(result["debate_day_count"], 2)
        self.assertEqual(result["speeches_per_debate_day"], 1.5)


class ValidationTests(unittest.TestCase):
    def test_history_overlap_is_reported(self):
        history = pd.DataFrame({
            "member_code": ["m1", "m1"],
            "party_start": ["2026-01-01", "2026-01-10"],
            "party_end": ["2026-01-15", None],
        })
        errors = validate_temporal_history(
            history,
            entity_col="member_code",
            start_col="party_start",
            end_col="party_end",
        )
        self.assertTrue(any("overlapping" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
