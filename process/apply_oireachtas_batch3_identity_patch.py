from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match in {path}, found {count}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched {path}")


def main() -> int:
    replace_once(
        "extract/oireachtas/table_member_memberships.py",
        "membership_id = membership_uri or f\"generated:membership:{stable_hash([member_code, member_uri, house_uri, house_no, house_code, membership_start, membership_end])}\"",
        "membership_id = membership_uri or f\"generated:membership:{stable_hash([member_code, member_uri, house_uri, house_no, house_code, membership_start])}\"",
    )
    replace_once(
        "extract/oireachtas/table_member_parties.py",
        "membership_id = _first_text(membership, \"uri\", \"membershipUri\") or f\"generated:membership:{stable_hash([member_code, _membership_start(membership), _membership_end(membership)])}\"",
        "membership_id = _first_text(membership, \"uri\", \"membershipUri\") or f\"generated:membership:{stable_hash([member_code, _membership_start(membership)])}\"",
    )
    replace_once(
        "extract/oireachtas/table_member_parties.py",
        "party_uri = f\"generated:party:{stable_hash([party_name, party_start, party_end])}\"",
        "party_uri = f\"generated:party:{stable_hash([party_name])}\"",
    )
    replace_once(
        "extract/oireachtas/table_member_parties.py",
        "member_party_id = f\"generated:member_party:{stable_hash([membership_id, member_code, party_uri, party_start, party_end])}\"",
        "member_party_id = f\"generated:member_party:{stable_hash([membership_id, member_code, party_uri, party_start])}\"",
    )
    replace_once(
        "extract/oireachtas/table_member_constituencies.py",
        "f\"generated:membership:{stable_hash([member_code, membership_start, membership_end])}\"",
        "f\"generated:membership:{stable_hash([member_code, membership_start])}\"",
    )
    replace_once(
        "extract/oireachtas/table_member_constituencies.py",
        "f\"{stable_hash([membership_id, member_code, constituency_uri, represent_start, represent_end])}\"",
        "f\"{stable_hash([membership_id, member_code, constituency_uri, represent_start])}\"",
    )
    replace_once(
        "extract/oireachtas/table_member_offices.py",
        "membership_id = _first_text(membership, \"uri\", \"membershipUri\") or f\"generated:membership:{stable_hash([member_code, _date_start(membership), _date_end(membership)])}\"",
        "membership_id = _first_text(membership, \"uri\", \"membershipUri\") or f\"generated:membership:{stable_hash([member_code, _date_start(membership)])}\"",
    )
    replace_once(
        "extract/oireachtas/table_member_offices.py",
        "office_uri = f\"generated:office:{stable_hash([office_name, office_start, office_end])}\"",
        "office_uri = f\"generated:office:{stable_hash([office_name])}\"",
    )
    replace_once(
        "extract/oireachtas/table_member_offices.py",
        "\"member_office_id\": f\"generated:member_office:{stable_hash([membership_id, member_code, office_uri, office_start, office_end])}\"",
        "\"member_office_id\": f\"generated:member_office:{stable_hash([membership_id, member_code, office_uri, office_start])}\"",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
