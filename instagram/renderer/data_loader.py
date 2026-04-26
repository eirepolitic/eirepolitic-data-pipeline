from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from .constants import DATASET_CANDIDATES, DEFAULT_BUCKET, DEFAULT_REGION, LOCAL_DATASET_FILENAMES

REQUIRED_DATASETS = {"members"}


@dataclass
class DatasetBundle:
    tables: Dict[str, pd.DataFrame]
    sources: Dict[str, str]


class BaseCSVLoader:
    def read_first_csv(self, label: str, keys: Iterable[str], required: bool = False) -> pd.DataFrame:
        raise NotImplementedError


class S3CSVLoader(BaseCSVLoader):
    def __init__(self, bucket: str = DEFAULT_BUCKET, region: str = DEFAULT_REGION) -> None:
        self.bucket = bucket
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)
        self.used_keys: Dict[str, str] = {}

    def _exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def read_first_csv(self, label: str, keys: Iterable[str], required: bool = False) -> pd.DataFrame:
        for key in keys:
            if not self._exists(key):
                continue
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            text = obj["Body"].read().decode("utf-8-sig", errors="replace")
            self.used_keys[label] = f"s3://{self.bucket}/{key}"
            return pd.read_csv(io.StringIO(text))
        if required:
            raise FileNotFoundError(f"Missing required dataset for label={label}. Checked: {list(keys)}")
        self.used_keys[label] = "missing"
        return pd.DataFrame()


class LocalCSVLoader(BaseCSVLoader):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.used_keys: Dict[str, str] = {}

    def read_first_csv(self, label: str, keys: Iterable[str], required: bool = False) -> pd.DataFrame:
        filename = LOCAL_DATASET_FILENAMES[label]
        path = self.root / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing local fixture for {label}: {path}")
            self.used_keys[label] = "missing"
            return pd.DataFrame()
        self.used_keys[label] = str(path)
        return pd.read_csv(path)


def load_datasets(loader: BaseCSVLoader) -> DatasetBundle:
    tables = {}
    sources = {}
    for label, keys in DATASET_CANDIDATES.items():
        required = label in REQUIRED_DATASETS
        tables[label] = loader.read_first_csv(label, keys, required=required)
        sources[label] = loader.used_keys[label]
    return DatasetBundle(tables=tables, sources=sources)
