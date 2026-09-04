"""Polling data ingestion for EirePolitic."""

from .ipi import prepare_ingestion, publish_ingestion

__all__ = ["prepare_ingestion", "publish_ingestion"]
