from __future__ import annotations

from pathlib import Path


REPLACEMENTS = {
    "extract/oireachtas/table_member_parties.py": (
        '        compared_fields=("membership_id", "party_name", "is_current"),\n',
        '        compared_fields=("party_name", "is_current"),\n',
    ),
    "extract/oireachtas/table_member_constituencies.py": (
        '        compared_fields=("membership_id", "constituency_name", "is_current"),\n',
        '        compared_fields=("constituency_name", "is_current"),\n',
    ),
}


def apply() -> None:
    for file_name, (old, new) in REPLACEMENTS.items():
        path = Path(file_name)
        text = path.read_text(encoding="utf-8")
        if old not in text:
            raise RuntimeError(f"comparison anchor missing in {file_name}")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")


if __name__ == "__main__":
    apply()
