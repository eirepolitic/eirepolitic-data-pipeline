from __future__ import annotations

import json
from urllib import parse

import pytest

from publishing.meta_http_client import MetaApiError, MetaCredentials, MetaInstagramHttpClient


class FakeTransport:
    def __init__(self, status=200, payload=None):
        self.status = status
        self.payload = payload if payload is not None else {"id": "result-1"}
        self.requests = []

    def __call__(self, req, timeout):
        self.requests.append((req, timeout))
        return self.status, json.dumps(self.payload).encode("utf-8")


def _client(transport):
    return MetaInstagramHttpClient(
        MetaCredentials("ig-user-123", "super-secret-token"),
        graph_version="vTEST",
        transport=transport,
    )


def _form(req):
    return parse.parse_qs(req.data.decode("utf-8"), keep_blank_values=True) if req.data else {}


def test_token_is_authorization_header_not_url_or_form():
    transport = FakeTransport()
    client = _client(transport)
    client.publish_container("container-1")
    req, _ = transport.requests[0]
    assert req.get_header("Authorization") == "Bearer super-secret-token"
    assert "super-secret-token" not in req.full_url
    assert "access_token" not in _form(req)


def test_image_container_encodes_supported_metadata():
    transport = FakeTransport(payload={"id": "container-1"})
    client = _client(transport)
    result = client.create_image_container(
        image_url="https://example.invalid/asset.jpg?signature=private",
        caption="Exact caption",
        alt_text="Accessible description",
        user_tags=({"username": "example", "x": 0.5, "y": 0.5},),
        collaborators=("collaborator",),
        location_id="location-1",
        is_carousel_item=True,
    )
    assert result == "container-1"
    req, _ = transport.requests[0]
    form = _form(req)
    assert req.full_url.endswith("/vTEST/ig-user-123/media")
    assert form["caption"] == ["Exact caption"]
    assert form["alt_text"] == ["Accessible description"]
    assert json.loads(form["user_tags"][0])[0]["username"] == "example"
    assert json.loads(form["collaborators"][0]) == ["collaborator"]
    assert form["location_id"] == ["location-1"]
    assert form["is_carousel_item"] == ["true"]


def test_carousel_container_uses_ordered_children():
    transport = FakeTransport(payload={"id": "parent-1"})
    client = _client(transport)
    client.create_carousel_container(
        child_container_ids=("child-1", "child-2"),
        caption="Caption",
        collaborators=(),
        location_id=None,
    )
    form = _form(transport.requests[0][0])
    assert form["media_type"] == ["CAROUSEL"]
    assert form["children"] == ["child-1,child-2"]


def test_container_status_is_parsed_without_guessing():
    transport = FakeTransport(payload={"id": "container-1", "status_code": "FINISHED", "status": "Finished"})
    status = _client(transport).get_container_status("container-1")
    assert status.container_id == "container-1"
    assert status.status_code == "FINISHED"
    assert status.media_id is None


def test_publish_and_comment_return_ids():
    publish_transport = FakeTransport(payload={"id": "ig-media-1"})
    assert _client(publish_transport).publish_container("container-1") == "ig-media-1"
    assert _form(publish_transport.requests[0][0])["creation_id"] == ["container-1"]

    comment_transport = FakeTransport(payload={"id": "comment-1"})
    assert _client(comment_transport).create_comment("ig-media-1", "First comment") == "comment-1"
    assert _form(comment_transport.requests[0][0])["message"] == ["First comment"]


def test_rate_limit_error_is_retryable_and_auth_error_is_not():
    rate = FakeTransport(status=429, payload={"error": {"message": "rate limited", "code": 4}})
    with pytest.raises(MetaApiError) as exc_info:
        _client(rate).publish_container("container")
    assert exc_info.value.retryable is True
    assert exc_info.value.code == 4

    auth = FakeTransport(status=400, payload={"error": {"message": "invalid token", "code": 190}})
    with pytest.raises(MetaApiError) as exc_info:
        _client(auth).publish_container("container")
    assert exc_info.value.retryable is False
    assert exc_info.value.code == 190


def test_invalid_json_and_missing_ids_fail_closed():
    class BadJson:
        def __call__(self, req, timeout):
            return 502, b"not-json"

    with pytest.raises(MetaApiError, match="invalid JSON") as exc_info:
        _client(BadJson()).publish_container("container")
    assert exc_info.value.retryable is True

    with pytest.raises(MetaApiError, match="did not return an ID"):
        _client(FakeTransport(payload={})).create_comment("media", "comment")
