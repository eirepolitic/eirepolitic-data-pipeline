from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from instagram.factory.oireachtas_source import load_csv_tables, resolve_validated_production_batch
from instagram.factory.package import deterministic_zip
from instagram.factory.render_primitives import contact_sheet
from instagram.projects.bill_tracker_factory_v1 import adapter as enacted_adapter
from instagram.projects.bill_tracker_factory_v1.first_stage_renderers import (
    render_first_stage_bill,
    render_first_stage_cover,
    render_first_stage_explainer,
    render_first_stage_process_glossary,
)

PROJECT_ID = "bill_tracker_factory_v1"
FIRST_STAGE_PROTOTYPE = "first_stage_prototype"
FIRST_STAGE_POST1 = "first_stage_post1"
FIRST_STAGE_POST2 = "first_stage_post2"


def _assert_image(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing rendered slide: {path}")
    with Image.open(path) as image:
        if image.size != (1080, 1350):
            raise RuntimeError(f"Unexpected dimensions for {path}: {image.size}")


def _load_payload() -> dict[str, Any]:
    return yaml.safe_load(Path("instagram/projects/bill_tracker_factory_v1/first_stage_content.yml").read_text(encoding="utf-8")) or {}


def _bill_uri(bill: dict[str, Any]) -> str:
    number, year = str(bill["bill_number"]).split("/", 1)
    return f"https://data.oireachtas.ie/ie/oireachtas/bill/{year}/{number}"


def _production_counts(bills: list[dict[str, Any]], frames: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for bill in bills:
        uri = _bill_uri(bill)
        table_counts: dict[str, int] = {}
        for table_name, frame in frames.items():
            table_counts[table_name] = int(frame[frame["bill_id"].astype(str).eq(uri)].shape[0]) if "bill_id" in frame.columns else 0
        if table_counts.get("silver_bills", 0) != 1:
            raise RuntimeError(f"Expected exactly one production Bill row for {uri}: {table_counts}")
        counts[bill["key"]] = table_counts
    return counts


def _prepare_root(output_root: Path, period: str) -> tuple[Path, Path, Path, Path]:
    root = output_root / f"period={period}"
    if root.exists(): shutil.rmtree(root)
    slides_dir = root / "slides"; metadata_dir = root / "metadata"; contact_dir = root / "contact_sheets"
    for directory in (slides_dir, metadata_dir, contact_dir): directory.mkdir(parents=True, exist_ok=True)
    return root, slides_dir, metadata_dir, contact_dir


def _generate_first_stage_prototype(*, output_root: Path) -> dict[str, Any]:
    payload = _load_payload(); bill = (payload.get("prototype") or {}).get("bill") or {}
    if not bill: raise RuntimeError("First Stage prototype content is missing its Bill payload")
    batch = resolve_validated_production_batch()
    frames, lineage = load_csv_tables(batch, ["silver_bills", "silver_bill_stages", "silver_bill_sponsors", "silver_bill_debates"])
    production_rows = _production_counts([bill], frames)
    root, slides_dir, metadata_dir, contact_dir = _prepare_root(output_root, FIRST_STAGE_PROTOTYPE)
    slide = slides_dir / "01_adult_safeguarding_first_stage.png"
    render_manifest = render_first_stage_bill(bill, slide); _assert_image(slide)
    contact_path = contact_dir / "first_stage_prototype_contact_sheet.jpg"; contact_sheet([("Adult Safeguarding · First Stage", slide)], contact_path, columns=1)
    caption_path = root / "caption.txt"; caption_path.write_text("Approved component prototype — publication remains disabled.\n", encoding="utf-8")
    manifest = {"project_id":PROJECT_ID,"period_key":FIRST_STAGE_PROTOTYPE,"review_state":"pending_human_review","publication_enabled":False,"publishing_allowed":False,"source_batch_id":batch.batch_id,"source_pointer":batch.pointer,"source_lineage":lineage,"production_bill_row_counts":production_rows,"editorial_verification_date":"2026-10-03","editorial_sources":bill.get("sources") or [],"slides":[str(slide)],"contact_sheets":{FIRST_STAGE_PROTOTYPE:str(contact_path)},"caption":str(caption_path),"render_manifests":{slide.stem:render_manifest},"qa":{"expected_slide_count":1,"actual_slide_count":1,"dimensions":[1080,1350],"source_footer_required":True,"publication_enabled":False,"publishing_allowed":False,"overflow_assertions_passed":True}}
    manifest_path = metadata_dir / "manifest.json"; manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    manifest["package"] = deterministic_zip(root, root / "bill_tracker_first_stage_prototype_review.zip"); manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    return manifest


def _generate_first_stage_post(*, period: str, output_root: Path) -> dict[str, Any]:
    payload = _load_payload(); series = payload["series"]; post = (payload.get("posts") or {}).get(period) or {}; bills = post.get("bills") or []
    if len(bills) != 3: raise RuntimeError(f"{period} requires exactly three current Bills; got {len(bills)}")
    batch = resolve_validated_production_batch()
    frames, lineage = load_csv_tables(batch,["silver_bills","silver_bill_stages","silver_bill_sponsors","silver_bill_debates"])
    production_rows = _production_counts(bills, frames)
    root, slides_dir, metadata_dir, contact_dir = _prepare_root(output_root, period)
    slides:list[Path]=[]; render_manifests:dict[str,Any]={}
    cover=slides_dir/"01_cover.png"; render_manifests[cover.stem]=render_first_stage_cover(post,series,cover); _assert_image(cover); slides.append(cover)
    for idx,bill in enumerate(bills,start=2):
        p=slides_dir/f"{idx:02d}_{bill['key']}.png"; render_manifests[p.stem]=render_first_stage_bill(bill,p); _assert_image(p); slides.append(p)
    process=slides_dir/"05_process_glossary.png"; render_manifests[process.stem]=render_first_stage_process_glossary(series["glossary"],process); _assert_image(process); slides.append(process)
    explainer=slides_dir/"06_first_stage_explainer.png"; render_manifests[explainer.stem]=render_first_stage_explainer(series["glossary"],explainer); _assert_image(explainer); slides.append(explainer)
    if len(slides)!=6: raise RuntimeError(f"{period} rendered {len(slides)} slides, expected 6")
    contact_path=contact_dir/f"{period}_contact_sheet.jpg"; labels=[p.stem.replace("_"," ").title() for p in slides]; contact_sheet(list(zip(labels,slides)),contact_path,columns=2)
    caption_path=root/"caption.txt"; caption_path.write_text(str(post["caption_draft"]).strip()+"\n",encoding="utf-8")
    editorial_sources=[source for bill in bills for source in (bill.get("sources") or [])]
    manifest={"project_id":PROJECT_ID,"period_key":period,"review_state":"pending_human_review","publication_enabled":False,"publishing_allowed":False,"source_batch_id":batch.batch_id,"source_pointer":batch.pointer,"source_lineage":lineage,"production_bill_row_counts":production_rows,"editorial_verification_date":"2026-10-03","editorial_sources":editorial_sources,"slides":[str(p) for p in slides],"contact_sheets":{period:str(contact_path)},"caption":str(caption_path),"render_manifests":render_manifests,"qa":{"expected_slide_count":6,"actual_slide_count":6,"dimensions":[1080,1350],"source_footer_required":True,"publication_enabled":False,"publishing_allowed":False,"overflow_assertions_passed":True,"live_stage_verification_date":"2026-10-03"}}
    manifest_path=metadata_dir/"manifest.json"; manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    manifest["package"]=deterministic_zip(root,root/f"bill_tracker_{period}_review.zip"); manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    return manifest


def generate(*, project: dict[str, Any], period_spec: str, output_root: Path) -> dict[str, Any]:
    period=(period_spec or "post1").strip().lower()
    if period==FIRST_STAGE_PROTOTYPE: return _generate_first_stage_prototype(output_root=output_root)
    if period in {FIRST_STAGE_POST1,FIRST_STAGE_POST2}: return _generate_first_stage_post(period=period,output_root=output_root)
    return enacted_adapter.generate(project=project,period_spec=period,output_root=output_root)
