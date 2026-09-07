import unittest

import pandas as pd

from political_metrics.bill_content_snapshot import audit_bill_content_snapshot, build_bill_content_snapshot


class BillContentSnapshotTests(unittest.TestCase):
    def _bills(self, count=8):
        rows=[]
        for i in range(1,count+1):
            rows.append({
                "bill_id":f"b{i}","bill_no":str(i),"bill_year":"2026","title":f"Bill {i}","short_title":f"Bill {i}",
                "origin_house_name":"Dáil Éireann","bill_type":"Public","status":"Current","introduced_date":"2026-01-01",
                "last_event_date":"2026-07-01","snapshot_date":"2026-09-06",
            })
        return pd.DataFrame(rows)

    def _stages(self, count=8):
        rows=[]
        for i in range(1,count+1):
            rows.extend([
                {"bill_id":f"b{i}","stage_name":"First Stage","stage_date":"2026-01-01","house_name":"Dáil Éireann","stage_outcome":"","order_in_bill":"1"},
                {"bill_id":f"b{i}","stage_name":"Second Stage","stage_date":"2026-02-01","house_name":"Dáil Éireann","stage_outcome":"","order_in_bill":"2"},
            ])
        return pd.DataFrame(rows)

    def _sponsors(self, count=8):
        return pd.DataFrame([
            {"bill_id":f"b{i}","sponsor_name":f"Sponsor {i}","sponsor_role_name":"Deputy","sponsor_uri":f"member:{i}","is_primary":"true","sponsor_order":"1"}
            for i in range(1,count+1)
        ])

    def _empty_context(self):
        bridge=pd.DataFrame(columns=["bill_id","debate_section_id"])
        speeches=pd.DataFrame(columns=["speech_id","debate_section_id"])
        divisions=pd.DataFrame(columns=["division_id","debate_section_id","division_date","subject","outcome"])
        votes=pd.DataFrame(columns=["division_id","vote_label"])
        return bridge,speeches,divisions,votes

    def test_latest_stage_and_six_bill_batches(self):
        bridge,speeches,divisions,votes=self._empty_context()
        result=build_bill_content_snapshot(
            bills=self._bills(),stages=self._stages(),sponsors=self._sponsors(),bill_debate_sections=bridge,
            speeches=speeches,divisions=divisions,member_votes=votes,batch_size=6,generated_at_utc="2026-09-06T00:00:00+00:00",
        )
        self.assertEqual(set(result["current_stage_name"]),{"Second Stage"})
        self.assertEqual(set(result["series_bucket"]),{"second_stage"})
        self.assertEqual(result["series_batch_id"].nunique(),2)
        self.assertEqual(result.groupby("series_batch_id").size().tolist(),[6,2])
        self.assertTrue((result["support_opposition_status"]=="do_not_infer_from_speeches").all())
        self.assertTrue(audit_bill_content_snapshot(result,batch_size=6)["ready"])

    def test_cream_list_is_public_returned_amendments_bucket(self):
        bills=self._bills(1)
        stages=pd.DataFrame([
            {"bill_id":"b1","stage_name":"Fifth Stage","stage_date":"2026-06-01","house_name":"Seanad Éireann","stage_outcome":"","order_in_bill":"5"},
            {"bill_id":"b1","stage_name":"Cream List","stage_date":"2026-07-01","house_name":"Dáil Éireann","stage_outcome":"","order_in_bill":"6"},
        ])
        bridge,speeches,divisions,votes=self._empty_context()
        result=build_bill_content_snapshot(
            bills=bills,stages=stages,sponsors=self._sponsors(1),bill_debate_sections=bridge,
            speeches=speeches,divisions=divisions,member_votes=votes,generated_at_utc="2026-09-06T00:00:00+00:00",
        )
        self.assertEqual(result.iloc[0]["series_bucket"],"returned_amendments")
        self.assertEqual(result.iloc[0]["series_bucket_label"],"Returned amendments")
        self.assertEqual(result.iloc[0]["house_badge"],"Dáil Éireann")

    def test_certified_division_enables_vote_evidence_only_for_linked_bill(self):
        bills=self._bills(2)
        stages=self._stages(2)
        sponsors=self._sponsors(2)
        bridge=pd.DataFrame([{"bill_id":"b1","debate_section_id":"s1"}])
        speeches=pd.DataFrame([{"speech_id":"p1","debate_section_id":"s1"},{"speech_id":"p2","debate_section_id":"s1"}])
        divisions=pd.DataFrame([{"division_id":"d1","debate_section_id":"s1","division_date":"2026-03-01","subject":"Question put","outcome":"carried"}])
        votes=pd.DataFrame([
            {"division_id":"d1","vote_label":"yes"},{"division_id":"d1","vote_label":"yes"},
            {"division_id":"d1","vote_label":"no"},{"division_id":"d1","vote_label":"abstain"},
        ])
        result=build_bill_content_snapshot(
            bills=bills,stages=stages,sponsors=sponsors,bill_debate_sections=bridge,speeches=speeches,
            divisions=divisions,member_votes=votes,generated_at_utc="2026-09-06T00:00:00+00:00",
        ).set_index("bill_id")
        self.assertEqual(result.loc["b1","certified_speech_count"],2)
        self.assertEqual(result.loc["b1","certified_division_count"],1)
        self.assertEqual(result.loc["b1","latest_division_ta"],2)
        self.assertEqual(result.loc["b1","latest_division_nil"],1)
        self.assertEqual(result.loc["b1","latest_division_abstain"],1)
        self.assertEqual(result.loc["b1","support_opposition_status"],"recorded_vote_evidence_available")
        self.assertEqual(result.loc["b2","support_opposition_status"],"do_not_infer_from_speeches")


if __name__ == "__main__":
    unittest.main()
