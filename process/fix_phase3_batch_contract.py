from pathlib import Path

batch_path = Path("instagram/factory/constituency_batch.py")
text = batch_path.read_text(encoding="utf-8")
old = '''def _batch_id(source_manifest: dict[str, Any]) -> str:\n    for source_name in ("members", "speeches"):\n        source = source_manifest.get(source_name, {})\n        batch_id = source.get("resolution", {}).get("batch_id")\n        if batch_id:\n            return str(batch_id)\n    return "local-fixture"\n'''
new = '''def _batch_id(source_manifest: dict[str, Any]) -> str:\n    for source_name in ("members", "speeches"):\n        source = source_manifest.get(source_name, {})\n        if not isinstance(source, dict):\n            continue\n        resolution = source.get("resolution", {})\n        if not isinstance(resolution, dict):\n            continue\n        batch_id = resolution.get("batch_id")\n        if batch_id:\n            return str(batch_id)\n    return "local-fixture"\n'''
if old not in text:
    raise SystemExit("batch id block not found")
batch_path.write_text(text.replace(old, new, 1), encoding="utf-8")

test_path = Path("tests/test_instagram_factory.py")
tests = test_path.read_text(encoding="utf-8")
old_test = '''            self.assertEqual(first["run_id"], second["run_id"])\n            self.assertEqual(first["status"], "succeeded")\n            self.assertGreater(first["item_count_succeeded"], 0)\n            self.assertEqual(first["item_count_failed"], 0)\n            self.assertFalse(first["approved"])\n            self.assertFalse(first["publishing_allowed"])\n            run_root = Path(temp_dir) / first["project_id"] / "runs" / first["run_id"]\n            self.assertTrue((run_root / "run_manifest.json").is_file())\n            self.assertTrue((run_root / "review/review_state.json").is_file())\n            for item in first["items"]:\n                self.assertTrue((run_root / item["manifest"]).is_file())\n'''
new_test = '''            self.assertEqual(first["run_id"], second["run_id"])\n            self.assertEqual(first["state"], "succeeded")\n            self.assertGreater(first["succeeded_item_count"], 0)\n            self.assertEqual(first["failed_item_count"], 0)\n            self.assertFalse(first["approved"])\n            self.assertFalse(first["publishing_allowed"])\n            run_root = Path(first["output_root"])\n            self.assertTrue((run_root / "run_manifest.json").is_file())\n            self.assertTrue((run_root / "review/review_state.json").is_file())\n            for item in first["items"].values():\n                self.assertTrue((run_root / item["manifest"]).is_file())\n'''
if old_test not in tests:
    raise SystemExit("batch test block not found")
test_path.write_text(tests.replace(old_test, new_test, 1), encoding="utf-8")

workflow_path = Path(".github/workflows/instagram_factory_constituency_batch.yml")
workflow = workflow_path.read_text(encoding="utf-8")
old_workflow = '''          RUN_ID=$(python - <<'PY'\n          import json\n          print(json.load(open('batch-result.json', encoding='utf-8'))['run_id'])\n          PY\n          )\n          RUN_ROOT="$OUTPUT_ROOT/$PROJECT_ID/runs/$RUN_ID"\n          test -f "$RUN_ROOT/run_manifest.json"\n          echo "run_id=$RUN_ID" >> "$GITHUB_OUTPUT"\n          echo "run_root=$RUN_ROOT" >> "$GITHUB_OUTPUT"\n'''
new_workflow = '''          readarray -t VALUES < <(python - <<'PY'\n          import json\n          report = json.load(open('batch-result.json', encoding='utf-8'))\n          print(report['run_id'])\n          print(report['output_root'])\n          PY\n          )\n          RUN_ID="${VALUES[0]}"\n          RUN_ROOT="${VALUES[1]}"\n          test -f "$RUN_ROOT/run_manifest.json"\n          echo "run_id=$RUN_ID" >> "$GITHUB_OUTPUT"\n          echo "run_root=$RUN_ROOT" >> "$GITHUB_OUTPUT"\n'''
if old_workflow not in workflow:
    raise SystemExit("workflow resolve block not found")
workflow_path.write_text(workflow.replace(old_workflow, new_workflow, 1), encoding="utf-8")
