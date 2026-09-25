from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error, parse, request

from publishing.meta_provider import MetaContainerStatus


class MetaApiError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None, subcode: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.subcode = subcode
        self.retryable = retryable


@dataclass(frozen=True)
class MetaCredentials:
    ig_user_id: str
    access_token: str


Transport = Callable[[request.Request, float], tuple[int, bytes]]


def urllib_transport(req: request.Request, timeout: float) -> tuple[int, bytes]:
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except error.HTTPError as exc:
        return exc.code, exc.read()


class MetaInstagramHttpClient:
    """Minimal Meta Instagram Content Publishing HTTP client.

    Credentials are injected at runtime. Access tokens are sent in the
    Authorization header and are never added to URLs or exception messages.
    """

    def __init__(
        self,
        credentials: MetaCredentials,
        *,
        graph_version: str,
        base_url: str = "https://graph.facebook.com",
        timeout_seconds: float = 20.0,
        transport: Transport = urllib_transport,
    ) -> None:
        if not graph_version.startswith("v"):
            raise ValueError("graph_version must be explicit, for example vXX.X")
        self.credentials = credentials
        self.graph_version = graph_version
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def create_image_container(
        self,
        *,
        image_url: str,
        caption: str | None,
        alt_text: str | None,
        user_tags: tuple[dict[str, object], ...] = (),
        collaborators: tuple[str, ...] = (),
        location_id: str | None = None,
        is_carousel_item: bool = False,
    ) -> str:
        payload: dict[str, Any] = {"image_url": image_url}
        if caption is not None:
            payload["caption"] = caption
        if alt_text:
            payload["alt_text"] = alt_text
        if user_tags:
            payload["user_tags"] = json.dumps(user_tags, separators=(",", ":"))
        if collaborators:
            payload["collaborators"] = json.dumps(collaborators, separators=(",", ":"))
        if location_id:
            payload["location_id"] = location_id
        if is_carousel_item:
            payload["is_carousel_item"] = "true"
        data = self._request("POST", f"/{self.credentials.ig_user_id}/media", form=payload)
        return self._required_id(data, "create image container")

    def create_carousel_container(
        self,
        *,
        child_container_ids: tuple[str, ...],
        caption: str,
        collaborators: tuple[str, ...] = (),
        location_id: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "media_type": "CAROUSEL",
            "children": ",".join(child_container_ids),
            "caption": caption,
        }
        if collaborators:
            payload["collaborators"] = json.dumps(collaborators, separators=(",", ":"))
        if location_id:
            payload["location_id"] = location_id
        data = self._request("POST", f"/{self.credentials.ig_user_id}/media", form=payload)
        return self._required_id(data, "create carousel container")

    def get_container_status(self, container_id: str) -> MetaContainerStatus:
        data = self._request("GET", f"/{container_id}", query={"fields": "status_code,id"})
        return MetaContainerStatus(
            container_id=container_id,
            status_code=str(data.get("status_code", "")).upper(),
            media_id=data.get("media_id"),
            error_message=data.get("status"),
        )

    def publish_container(self, container_id: str) -> str | None:
        data = self._request(
            "POST",
            f"/{self.credentials.ig_user_id}/media_publish",
            form={"creation_id": container_id},
        )
        value = data.get("id")
        return str(value) if value else None

    def create_comment(self, media_id: str, message: str) -> str:
        data = self._request("POST", f"/{media_id}/comments", form={"message": message})
        return self._required_id(data, "create comment")

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        form: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{self.graph_version}{path}"
        if query:
            url = f"{url}?{parse.urlencode(query)}"
        body = parse.urlencode(form or {}).encode("utf-8") if form is not None else None
        req = request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.credentials.access_token}",
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "Eirepolitic-Instagram-Publisher/1.0",
            },
        )
        status, raw = self.transport(req, self.timeout_seconds)
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetaApiError(f"Meta returned invalid JSON with HTTP {status}", retryable=status >= 500) from exc
        if status >= 400 or "error" in data:
            self._raise_api_error(status, data)
        if not isinstance(data, dict):
            raise MetaApiError("Meta returned an unexpected response shape")
        return data

    @staticmethod
    def _required_id(data: dict[str, Any], operation: str) -> str:
        value = data.get("id")
        if not value:
            raise MetaApiError(f"Meta did not return an ID for {operation}")
        return str(value)

    @staticmethod
    def _raise_api_error(status: int, data: dict[str, Any]) -> None:
        value = data.get("error") if isinstance(data, dict) else None
        value = value if isinstance(value, dict) else {}
        code = value.get("code")
        subcode = value.get("error_subcode")
        message = str(value.get("message") or f"Meta API request failed with HTTP {status}")
        # Retry transport/server failures and common rate-limit responses; auth/input
        # failures require reconciliation or operator action instead of blind retries.
        retryable = status >= 500 or status == 429 or code in {1, 2, 4, 17, 32, 613}
        raise MetaApiError(message, code=code, subcode=subcode, retryable=retryable)
