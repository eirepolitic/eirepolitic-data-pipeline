import unittest
from datetime import date

import pandas as pd

from political_metrics.bill_editorial_series import audit_editorial_bill_series, build_editorial_bill_series


class BillEditorialSeriesTests(unittest.TestCase):
    def _snapshot(self):
        rows=[]
        states=[
            ("b1","Current","Second Stage","2026-08-20"),
            ("b2","Current","Committee Stage","2026-07-01"),
            ("b3","Enacted","Fifth Stage","2026-08-01"),
            ("b4","Lapsed","Second Stage","2026-08-15"),
            ("b5","Withdrawn","First Stage","2026-08-10"),
            ("b6","Current","Cream List","2026-08-25"),
            ("b7","Current","Second Stage","2025-12-01"),
        ]
        for i,(bid,status,stage,last_event) in enumerate(states,1):
            rows.append({
                "bill_id":bid,"status":status,"current_stage_name":stage,"current_stage_date":last_event,
                "current_stage_house_name":"Dáil Éireann","current_state_key":f"{status}|{stage}|Dáil Éireann|{last_event}",
                "last_event_date":last_event,"bill_year":"2026","bill_no":str(i),"short_title":f"Bill {i}","title":f"Bill {i}",
            })
        return pd.DataFrame(rows)

    def test_baseline_recent_excludes_stale_and_non_core_statuses(self):
        result=build_editorial_bill_series(
            self._snapshot(),batch_size=6,as_of_date=date(2026,9,6),lookback_days=180
        )
        self.assertEqual(set(result["bill_id"]),{"b1","b2","b3","b6"})
        self.assertNotIn("Lapsed",set(result["status"]))
        self.assertNotIn("Withdrawn",set(result["status"]))
        self.assertEqual(result.set_index("bill_id").loc["b6","editorial_bucket"],"returned_amendments")
        self.assertTrue(audit_editorial_bill_series(result,batch_size=6)["ready"])

    def test_delta_includes_only_new_or_changed_current_or_enacted(self):
        current=self._snapshot()
        previous=current.copy()
        previous.loc[previous["bill_id"]=="b1","current_state_key"]="Current|First Stage|Dáil Éireann|2026-01-01"
        previous=previous[previous["bill_id"]!="b3"].copy()
        result=build_editorial_bill_series(
            current,batch_size=6,as_of_date=date(2026,9,6),lookback_days=180,previous_snapshot=previous
        ).set_index("bill_id")
        self.assertEqual(set(result.index),{"b1","b3"})
        self.assertEqual(result.loc["b1","change_type"],"state_changed")
        self.assertEqual(result.loc["b3","change_type"],"new_bill")
        self.assertTrue((result["editorial_scope_mode"]=="snapshot_delta").all())


if __name__ == "__main__":
    unittest.main()
