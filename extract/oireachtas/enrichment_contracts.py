from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

DEFAULT_REGISTRY_PATH = Path("configs/oireachtas/enrichment_registry.yml")
DEFAULT_POLICY_PATH = Path("configs/oireachtas/enrichment_write_policies.yml")


class EnrichmentContractError(ValueError):
    pass


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise EnrichmentContractError(f"Missing enrichment contract file: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EnrichmentContractError(f"Expected YAML mapping in {resolved}")
    return payload


def load_enrichment_registry(
    path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, dict[str, Any]]:
    payload = _load_yaml_mapping(path)
    enrichments = payload.get("enrichments")
    if not isinstance(enrichments, dict) or not enrichments:
        raise EnrichmentContractError("Enrichment registry contains no enrichments")
    return enrichments


def load_enrichment_write_policies(
    path: str | Path = DEFAULT_POLICY_PATH,
) -> dict[str, dict[str, Any]]:
    payload = _load_yaml_mapping(path)
    policies = payload.get("policies")
    if not isinstance(policies, dict) or not policies:
        raise EnrichmentContractError("Enrichment write-policy file contains no policies")
    return policies


def get_enrichment_contract(
    table_name: str,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = load_enrichment_registry(registry_path)
    policies = load_enrichment_write_policies(policy_path)
    if table_name not in registry:
        raise EnrichmentContractError(f"Unknown enrichment table: {table_name}")
    if table_name not in policies:
        raise EnrichmentContractError(f"Missing write policy for enrichment: {table_name}")
    return registry[table_name], policies[table_name]


def validate_enrichment_manifest(
    table_name: str,
    manifest: Mapping[str, Any],
    *,
    require_candidate_artifacts: bool = False,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
) -> list[str]:
    contract, _ = get_enrichment_contract(
        table_name,
        registry_path=registry_path,
        policy_path=policy_path,
    )
    manifest_contract = contract.get("manifest")
    if not isinstance(manifest_contract, dict):
        raise EnrichmentContractError(f"Missing manifest contract for {table_name}")

    errors: list[str] = []
    if str(manifest.get("table") or "") != table_name:
        errors.append(f"manifest.table must equal {table_name}")

    required_fields = manifest_contract.get("required_fields") or []
    for field in required_fields:
        if field not in manifest:
            errors.append(f"manifest missing field: {field}")

    status = str(manifest.get("status") or "")
    allowed_statuses = set(manifest_contract.get("allowed_statuses") or [])
    if status not in allowed_statuses:
        errors.append(f"manifest status is not allowed: {status!r}")

    if require_candidate_artifacts:
        for field in manifest_contract.get("candidate_artifact_fields") or []:
            if not str(manifest.get(field) or "").strip():
                errors.append(f"manifest candidate artifact field is blank: {field}")

    checksums = manifest.get("artifact_checksums")
    if not isinstance(checksums, dict):
        errors.append("manifest.artifact_checksums must be a mapping")

    return errors


def validate_enrichment_table(
    table_name: str,
    frame: pd.DataFrame,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
) -> list[str]:
    contract, policy = get_enrichment_contract(
        table_name,
        registry_path=registry_path,
        policy_path=policy_path,
    )
    errors: list[str] = []

    required_columns = list(contract.get("required_columns") or [])
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")
        return errors

    primary_key = list(contract.get("primary_key") or [])
    if primary_key and bool(frame.duplicated(subset=primary_key).any()):
        errors.append(f"duplicate primary key rows: {primary_key}")

    for column in contract.get("required_non_blank") or []:
        blank_count = int(
            frame[column].fillna("").astype(str).str.strip().eq("").sum()
        )
        if blank_count:
            errors.append(f"blank required values in {column}: {blank_count}")

    enum_columns = contract.get("enum_columns") or {}
    for column, allowed in enum_columns.items():
        invalid = sorted(set(frame[column].fillna("").astype(str)) - set(allowed))
        if invalid:
            errors.append(f"invalid values in {column}: {invalid}")

    for conditional in contract.get("conditional_enums") or []:
        when = conditional.get("when") or {}
        column = str(conditional.get("column") or "")
        allowed = set(conditional.get("values") or [])
        mask = pd.Series(True, index=frame.index)
        for when_column, expected_value in when.items():
            mask &= frame[when_column].astype(str).eq(str(expected_value))
        invalid = sorted(
            set(frame.loc[mask, column].fillna("").astype(str)) - allowed
        )
        if invalid:
            errors.append(f"invalid conditional values in {column}: {invalid}")

    if policy.get("allow_duplicate_primary_keys") is False and primary_key:
        if bool(frame.duplicated(subset=primary_key).any()):
            duplicate_count = int(frame.duplicated(subset=primary_key).sum())
            errors.append(f"write policy rejects duplicate rows: {duplicate_count}")

    return errors


def validate_publish_contract(
    table_name: str,
    manifest: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
) -> list[str]:
    contract, policy = get_enrichment_contract(
        table_name,
        registry_path=registry_path,
        policy_path=policy_path,
    )
    errors = validate_enrichment_manifest(
        table_name,
        manifest,
        require_candidate_artifacts=True,
        registry_path=registry_path,
        policy_path=policy_path,
    )
    errors.extend(
        validate_enrichment_table(
            table_name,
            frame,
            registry_path=registry_path,
            policy_path=policy_path,
        )
    )

    publishable_status = str(
        (contract.get("manifest") or {}).get("publishable_status") or ""
    )
    if str(manifest.get("status") or "") != publishable_status:
        errors.append(
            f"manifest is not publishable: expected {publishable_status!r}, "
            f"got {manifest.get('status')!r}"
        )
    if policy.get("allow_stale_publication") is False and manifest.get("stale_reasons"):
        errors.append("write policy rejects stale publication")
    if policy.get("allow_partial_publication") is False:
        if str(manifest.get("candidate_validation_status") or "") == "validated_partial":
            errors.append("write policy rejects partial publication")
    if policy.get("validation_required") is True and manifest.get("dq_status") != "pass":
        errors.append("write policy requires dq_status=pass")
    if policy.get("atomic_pointer_publish_required") is not True:
        errors.append("write policy must require atomic pointer publication")
    if policy.get("allow_core_transaction_coupling") is not False:
        errors.append("write policy must prohibit core transaction coupling")

    return errors


def assert_valid_enrichment_manifest(
    table_name: str,
    manifest: Mapping[str, Any],
    *,
    require_candidate_artifacts: bool = False,
) -> None:
    errors = validate_enrichment_manifest(
        table_name,
        manifest,
        require_candidate_artifacts=require_candidate_artifacts,
    )
    if errors:
        raise EnrichmentContractError("; ".join(errors))


def assert_valid_publish_contract(
    table_name: str,
    manifest: Mapping[str, Any],
    frame: pd.DataFrame,
) -> None:
    errors = validate_publish_contract(table_name, manifest, frame)
    if errors:
        raise EnrichmentContractError("; ".join(errors))
