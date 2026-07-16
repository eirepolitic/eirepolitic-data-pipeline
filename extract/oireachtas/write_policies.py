"""Registry-driven write and relationship policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_POLICY_CONFIG = Path("configs/oireachtas/write_policies.yml")
VALID_STRATEGIES = {"snapshot_replace", "upsert", "append", "rebuild"}


@dataclass(frozen=True)
class ForeignKeyPolicy:
    columns: tuple[str, ...]
    references: str
    referenced_columns: tuple[str, ...]
    nullable: bool = False


@dataclass(frozen=True)
class WritePolicy:
    table: str
    write_strategy: str
    valid_from_column: str | None = None
    valid_to_column: str | None = None
    current_column: str | None = None
    foreign_keys: tuple[ForeignKeyPolicy, ...] = field(default_factory=tuple)


def load_write_policies(path: Path | str = DEFAULT_POLICY_CONFIG) -> dict[str, WritePolicy]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Write policy registry not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_tables = payload.get("tables") or {}
    policies: dict[str, WritePolicy] = {}
    for table, raw in raw_tables.items():
        raw = raw or {}
        strategy = str(raw.get("write_strategy") or "").strip()
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"Invalid write strategy for {table}: {strategy!r}")
        foreign_keys: list[ForeignKeyPolicy] = []
        for item in raw.get("foreign_keys") or []:
            columns = _as_tuple(item.get("columns"))
            referenced_columns = _as_tuple(item.get("referenced_columns"))
            if not columns or len(columns) != len(referenced_columns):
                raise ValueError(f"Invalid foreign key for {table}: {item!r}")
            foreign_keys.append(
                ForeignKeyPolicy(
                    columns=columns,
                    references=str(item.get("references") or "").strip(),
                    referenced_columns=referenced_columns,
                    nullable=bool(item.get("nullable", False)),
                )
            )
        policies[table] = WritePolicy(
            table=table,
            write_strategy=strategy,
            valid_from_column=_optional_text(raw.get("valid_from_column")),
            valid_to_column=_optional_text(raw.get("valid_to_column")),
            current_column=_optional_text(raw.get("current_column")),
            foreign_keys=tuple(foreign_keys),
        )
    return policies


def get_write_policy(table: str, path: Path | str = DEFAULT_POLICY_CONFIG) -> WritePolicy:
    policies = load_write_policies(path)
    try:
        return policies[table]
    except KeyError as exc:
        raise KeyError(f"No write policy configured for table {table!r}") from exc


def validate_policy_coverage(table_names: set[str], policies: dict[str, WritePolicy]) -> list[str]:
    """Return registry table names missing a write policy."""
    return sorted(table_names - set(policies))


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in (value or []) if str(item).strip())


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
