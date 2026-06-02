"""Schema helpers for unified Oireachtas tables.

The table registry lives in configs/oireachtas/tables.yml. These helpers keep
schema access small and deterministic for the early foundation packets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml


DEFAULT_TABLES_CONFIG = Path("configs/oireachtas/tables.yml")


@dataclass(frozen=True)
class TableSchema:
    """Minimal table contract loaded from tables.yml."""

    name: str
    layer: str
    status: str
    primary_key: List[str]
    columns: List[str]
    cadence: str = "manual"
    description: str = ""
    endpoint: Optional[str] = None

    @property
    def primary_key_display(self) -> str:
        return ",".join(self.primary_key)


def load_table_registry(path: Path | str = DEFAULT_TABLES_CONFIG) -> Dict[str, TableSchema]:
    """Load table definitions from a YAML registry."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Table registry not found: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_tables = data.get("tables") or {}
    registry: Dict[str, TableSchema] = {}

    for table_name, raw in raw_tables.items():
        raw = raw or {}
        columns = list(raw.get("columns") or [])
        primary_key = raw.get("primary_key") or []
        if isinstance(primary_key, str):
            primary_key = [primary_key]

        registry[table_name] = TableSchema(
            name=table_name,
            layer=str(raw.get("layer") or "").strip() or infer_layer(table_name),
            status=str(raw.get("status") or "planned"),
            primary_key=list(primary_key),
            columns=columns,
            cadence=str(raw.get("cadence") or "manual"),
            description=str(raw.get("description") or ""),
            endpoint=raw.get("endpoint"),
        )

    return registry


def infer_layer(table_name: str) -> str:
    """Infer layer from table-name prefix."""
    if table_name.startswith("silver_"):
        return "silver"
    if table_name.startswith("gold_"):
        return "gold"
    if table_name.startswith("control_"):
        return "control"
    return "unknown"


def get_table_schema(table_name: str, path: Path | str = DEFAULT_TABLES_CONFIG) -> TableSchema:
    """Return a table schema or raise a helpful KeyError."""
    registry = load_table_registry(path)
    try:
        return registry[table_name]
    except KeyError as exc:
        available = ", ".join(sorted(registry)) or "(none)"
        raise KeyError(f"Unknown table '{table_name}'. Available tables: {available}") from exc


def validate_required_columns(rows: Iterable[Mapping[str, Any]], schema: TableSchema) -> List[str]:
    """Return missing required columns for a sequence of row mappings.

    This is intentionally lightweight. Full data-quality checks are added in a
    later packet.
    """
    rows = list(rows)
    if not rows:
        return list(schema.columns)
    observed = set().union(*(row.keys() for row in rows))
    return [column for column in schema.columns if column not in observed]
