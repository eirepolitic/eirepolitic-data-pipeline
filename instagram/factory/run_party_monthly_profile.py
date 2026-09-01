from __future__ import annotations

import boto3

from instagram.factory import party_monthly_profile as profile
from instagram.factory.oireachtas_production import resolve_production_key


def _name_aliases(value: str) -> set[str]:
    """Conservative aliases for Oireachtas display-name punctuation/title variants."""
    name = " ".join(str(value or "").strip().split())
    if not name:
        return set()
    straight = name.replace("’", "'").replace("‘", "'")
    curly = straight.replace("'", "’")
    aliases = {name, straight, curly}
    for variant in (name, straight, curly):
        if not variant.casefold().startswith("deputy "):
            aliases.add(f"Deputy {variant}")
    return aliases


def _install_member_name_aliases() -> None:
    original = profile._member_snapshot_for_period

    def wrapped(s3, period):
        rows, source = original(s3, period)
        augmented = list(rows)
        for row in rows:
            name = profile._field(row, ["Full Name", "Member Name", "Name", "full_name"])
            if not name:
                continue
            for alias in _name_aliases(name):
                if alias == name:
                    continue
                alias_row = dict(row)
                if "full_name" in alias_row:
                    alias_row["full_name"] = alias
                elif "Full Name" in alias_row:
                    alias_row["Full Name"] = alias
                else:
                    alias_row["full_name"] = alias
                augmented.append(alias_row)
        source = {**source, "name_alias_policy": "deputy-prefix-and-apostrophe-normalization"}
        return augmented, source

    profile._member_snapshot_for_period = wrapped


def main() -> None:
    s3 = boto3.client("s3", region_name="ca-central-1")
    resolved_key, pointer = resolve_production_key(
        s3,
        bucket=profile.S3_BUCKET,
        production_key=profile.CLASSIFIED_KEY,
    )
    print(f"Resolved unified production batch: {pointer.get('batch_id') or pointer.get('mode')}")
    print(f"Resolved classified source: {resolved_key}")
    profile.CLASSIFIED_KEY = resolved_key
    _install_member_name_aliases()
    profile.main()


if __name__ == "__main__":
    main()
