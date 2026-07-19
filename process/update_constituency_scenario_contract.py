from pathlib import Path

pilot_path = Path("instagram/factory/constituency_pilot.py")
text = pilot_path.read_text(encoding="utf-8")

start = text.index("def build_scenarios(")
end = text.index("\ndef write_cover_asset", start)
scenario_block = '''def _bounded_issue_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in record["issue_rows"][:7]]


def _synthetic_scenario(
    *,
    scenario: str,
    display_record: dict[str, Any],
    result_record: dict[str, Any],
) -> dict[str, Any]:
    rows = _bounded_issue_rows(result_record)
    return {
        "constituency": display_record["constituency"],
        "constituency_key": display_record["constituency_key"],
        "display_constituency": display_record["constituency"],
        "display_constituency_key": display_record["constituency_key"],
        "result_constituency": result_record["constituency"],
        "result_constituency_key": result_record["constituency_key"],
        "member_names": list(result_record.get("member_names", [])),
        "member_count": result_record.get("member_count", 0),
        "issue_rows": rows,
        "issue_count": len(rows),
        "speech_count": sum(int(row["value"]) for row in rows),
        "result_issue_count": result_record["issue_count"],
        "result_speech_count": result_record["speech_count"],
        "max_issue_label_length": max(len(row["label"]) for row in rows),
        "scenario": scenario,
        "synthetic": True,
        "no_publication": True,
        "source_fields": {
            "display_constituency": display_record["constituency"],
            "result_constituency": result_record["constituency"],
            "issue_rows": result_record["constituency"],
        },
    }


def build_scenarios(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    shortest_name = min(records, key=lambda row: (len(row["constituency"]), row["constituency"]))
    longest_name = max(records, key=lambda row: (len(row["constituency"]), row["constituency"]))
    smallest_results = min(
        records,
        key=lambda row: (row["speech_count"], row["issue_count"], row["constituency"]),
    )
    biggest_results = max(
        records,
        key=lambda row: (row["speech_count"], row["issue_count"], row["constituency"]),
    )

    complexity_values = sorted(_complexity(record) for record in records)
    target = median(complexity_values)
    real_source = min(records, key=lambda row: (abs(_complexity(row) - target), row["constituency"]))
    real_rows = _bounded_issue_rows(real_source)

    return {
        "minimum": _synthetic_scenario(
            scenario="minimum",
            display_record=shortest_name,
            result_record=smallest_results,
        ),
        "maximum": _synthetic_scenario(
            scenario="maximum",
            display_record=longest_name,
            result_record=biggest_results,
        ),
        "real_example": {
            **real_source,
            "display_constituency": real_source["constituency"],
            "display_constituency_key": real_source["constituency_key"],
            "result_constituency": real_source["constituency"],
            "result_constituency_key": real_source["constituency_key"],
            "issue_rows": real_rows,
            "issue_count": len(real_rows),
            "speech_count": sum(int(row["value"]) for row in real_rows),
            "result_issue_count": real_source["issue_count"],
            "result_speech_count": real_source["speech_count"],
            "scenario": "real_example",
            "synthetic": False,
            "no_publication": True,
            "source_fields": {
                "display_constituency": real_source["constituency"],
                "result_constituency": real_source["constituency"],
                "issue_rows": real_source["constituency"],
            },
        },
    }

'''
text = text[:start] + scenario_block + text[end + 1 :]

old_cover = '''    detail = f"{scenario.get('member_count', 0)} current members · {scenario.get('speech_count', 0)} classified speeches"
    draw.text((width // 2, 560), detail, font=body_font, fill="#cbbf9f", anchor="mm")
    marker = "SYNTHETIC TEST" if scenario["synthetic"] else "REAL DATA EXAMPLE"
'''
new_cover = '''    if scenario["synthetic"]:
        detail = f"Results source: {scenario['result_constituency']}"
        marker = f"SYNTHETIC {scenario['scenario'].upper()} TEST"
    else:
        detail = f"{scenario.get('member_count', 0)} current members · {scenario.get('result_speech_count', 0)} classified speeches"
        marker = "REAL DATA EXAMPLE"
    draw.text((width // 2, 560), detail, font=body_font, fill="#cbbf9f", anchor="mm")
'''
if old_cover not in text:
    raise SystemExit("Cover detail block not found")
text = text.replace(old_cover, new_cover, 1)

old_sample = '''        "grouping": {"grain": "constituency", "key": scenario["constituency_key"]},
        "source_note": "Synthetic validation context" if scenario["synthetic"] else "Joined Oireachtas member and classified debate data",
'''
new_sample = '''        "grouping": {"grain": "constituency", "key": scenario["result_constituency_key"]},
        "source_note": (
            f"Synthetic validation context using results from {scenario['result_constituency']}"
            if scenario["synthetic"]
            else "Joined Oireachtas member and classified debate data"
        ),
'''
if old_sample not in text:
    raise SystemExit("Visual grouping block not found")
text = text.replace(old_sample, new_sample, 1)

old_manifest = '''            "scenario": scenario_name,
            "constituency": scenario["constituency"],
            "synthetic": scenario["synthetic"],
'''
new_manifest = '''            "scenario": scenario_name,
            "constituency": scenario["constituency"],
            "display_constituency": scenario["display_constituency"],
            "result_constituency": scenario["result_constituency"],
            "result_issue_count": scenario["result_issue_count"],
            "result_speech_count": scenario["result_speech_count"],
            "synthetic": scenario["synthetic"],
'''
if old_manifest not in text:
    raise SystemExit("Scenario manifest block not found")
text = text.replace(old_manifest, new_manifest, 1)
pilot_path.write_text(text, encoding="utf-8")

validation_path = Path("instagram/projects/constituency_issue_profile_v1/validation_plan.yml")
validation = validation_path.read_text(encoding="utf-8")
validation = validation.replace(
    '''    selection:\n      constituency: lowest combined issue-count and speech-count complexity\n      issue_rows: two smallest non-empty issue values\n''',
    '''    selection:\n      display_constituency: shortest constituency name\n      result_constituency: smallest total classified-speech result set\n      issue_rows: up to seven top issue rows from the result constituency\n''',
)
validation = validation.replace(
    '''    selection:\n      constituency: longest constituency label\n      issue_labels: seven distinct longest issue labels\n      issue_values: seven largest available issue values\n''',
    '''    selection:\n      display_constituency: longest constituency name\n      result_constituency: largest total classified-speech result set\n      issue_rows: up to seven top issue rows from the result constituency\n''',
)
validation = validation.replace(
    "  - maximum labels are distinct\n",
    "  - minimum display constituency is the shortest available name\n  - maximum display constituency is the longest available name\n  - minimum results come from the smallest constituency result set\n  - maximum results come from the largest constituency result set\n",
)
validation_path.write_text(validation, encoding="utf-8")

test_path = Path("tests/test_instagram_factory.py")
tests = test_path.read_text(encoding="utf-8")
anchor = '''    def test_local_complete_slide_render(self) -> None:\n'''
regression = '''    def test_synthetic_name_and_result_extremes_are_selected_independently(self) -> None:\n        records = [\n            {\n                "constituency": "Mayo",\n                "constituency_key": "mayo",\n                "member_names": ["A"],\n                "member_count": 1,\n                "issue_rows": [{"label": "Housing", "value": 5}],\n                "issue_count": 1,\n                "speech_count": 5,\n                "max_issue_label_length": 7,\n            },\n            {\n                "constituency": "Dublin Bay North",\n                "constituency_key": "dublin-bay-north",\n                "member_names": ["B"],\n                "member_count": 1,\n                "issue_rows": [{"label": "Health", "value": 1}],\n                "issue_count": 1,\n                "speech_count": 1,\n                "max_issue_label_length": 6,\n            },\n            {\n                "constituency": "Limerick City",\n                "constituency_key": "limerick-city",\n                "member_names": ["C"],\n                "member_count": 1,\n                "issue_rows": [\n                    {"label": "Housing", "value": 20},\n                    {"label": "Transport", "value": 10},\n                ],\n                "issue_count": 2,\n                "speech_count": 30,\n                "max_issue_label_length": 9,\n            },\n        ]\n\n        scenarios = build_scenarios(records)\n\n        self.assertEqual(scenarios["minimum"]["display_constituency"], "Mayo")\n        self.assertEqual(scenarios["minimum"]["result_constituency"], "Dublin Bay North")\n        self.assertEqual(scenarios["minimum"]["issue_rows"], [{"label": "Health", "value": 1}])\n        self.assertEqual(scenarios["maximum"]["display_constituency"], "Dublin Bay North")\n        self.assertEqual(scenarios["maximum"]["result_constituency"], "Limerick City")\n        self.assertEqual(scenarios["maximum"]["result_speech_count"], 30)\n\n'''
if anchor not in tests:
    raise SystemExit("Test insertion anchor not found")
tests = tests.replace(anchor, regression + anchor, 1)
test_path.write_text(tests, encoding="utf-8")

progress_path = Path("instagram/projects/constituency_issue_profile_v1/progress_log.md")
progress = progress_path.read_text(encoding="utf-8")
progress += "\n- 2026-07-19: Corrected synthetic scenario selection: shortest/longest constituency names now independently pair with smallest/largest constituency result sets; provenance is explicit in manifests and covers.\n"
progress_path.write_text(progress, encoding="utf-8")
