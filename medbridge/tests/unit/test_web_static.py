"""
Unit tests for the FastAPI-served web front-end (medbridge.web).

Covers:
1. ``mount_web``: page routes, static content types, cache headers, ETag
   revalidation and the css/js/img allow-list (no source/test files served).
2. CSP compatibility of the HTML: no inline scripts or event handlers, every
   script is an external ES module, no third-party/CDN or web-font URLs.
3. JS/CSS hygiene: every relative ES-module import resolves, no eval / new
   Function / document.write, HTML is only ever assigned from the escape-first
   Markdown renderer.
4. The development mock API (dev_mock_server): SSE wire format, the five
   keyword-triggered actions and MessageResponse contract compatibility.
"""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from medbridge.api.schemas.responses import MessageResponse
from medbridge.web import WEB_DIR, mount_web
from medbridge.web.dev_mock_server import create_mock_app

PAGES = {"/": "index.html", "/dashboard": "dashboard.html"}
JS_FILES = sorted((WEB_DIR / "js").rglob("*.js"))
CSS_FILES = sorted((WEB_DIR / "css").glob("*.css"))


class _TagCollector(HTMLParser):
    """Collect start tags (with attributes) and the text inside <script> elements."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.script_bodies: list[str] = []
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        self._in_script = tag == "script"
        if self._in_script:
            self.script_bodies.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self.script_bodies[-1] += data


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    mount_web(app)
    return TestClient(app)


def _parse(html: str) -> _TagCollector:
    parser = _TagCollector()
    parser.feed(html)
    return parser


# ===========================================================================
# 1. Routes, content types, caching, allow-list
# ===========================================================================


class TestMountWeb:
    @pytest.mark.parametrize("path", list(PAGES))
    def test_pages_are_html_without_cache(self, client: TestClient, path: str) -> None:
        resp = client.get(path)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert resp.headers["cache-control"] == "no-cache"
        assert "<!doctype html>" in resp.text.lower()

    @pytest.mark.parametrize(
        ("path", "content_type"),
        [
            ("/static/js/app.js", "text/javascript"),
            ("/static/js/ui/messages.js", "text/javascript"),
            ("/static/css/app.css", "text/css"),
            ("/static/img/logo.svg", "image/svg+xml"),
        ],
    )
    def test_static_content_types_and_cache(self, client: TestClient, path: str, content_type: str) -> None:
        resp = client.get(path)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith(content_type)
        assert "max-age=" in resp.headers["cache-control"]
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers.get("etag")

    def test_etag_revalidation_returns_304(self, client: TestClient) -> None:
        first = client.get("/static/js/markdown.js")
        second = client.get("/static/js/markdown.js", headers={"If-None-Match": first.headers["etag"]})
        assert second.status_code == 304

    @pytest.mark.parametrize(
        "path",
        [
            "/static/__init__.py",
            "/static/dev_mock_server.py",
            "/static/package.json",
            "/static/tests/index.js",
            "/static/index.html",
            "/static/js/../dev_mock_server.py",
            "/static/js/missing.js",
        ],
    )
    def test_only_css_js_img_are_served(self, client: TestClient, path: str) -> None:
        assert client.get(path).status_code == 404

    @pytest.mark.parametrize("page", list(PAGES.values()))
    def test_every_referenced_asset_is_served(self, client: TestClient, page: str) -> None:
        parsed = _parse((WEB_DIR / page).read_text(encoding="utf-8"))
        refs = [
            attrs.get("src") or attrs.get("href")
            for tag, attrs in parsed.tags
            if tag in {"script", "link", "img"}
        ]
        assert refs
        for ref in refs:
            assert ref and ref.startswith("/static/"), ref
            assert client.get(ref).status_code == 200, ref


# ===========================================================================
# 2. CSP compatibility of the HTML pages
# ===========================================================================


class TestHtmlIsCspSafe:
    @pytest.mark.parametrize("path", list(PAGES))
    def test_scripts_are_external_modules(self, client: TestClient, path: str) -> None:
        parsed = _parse(client.get(path).text)
        scripts = [attrs for tag, attrs in parsed.tags if tag == "script"]
        assert scripts, "page loads at least one module"
        for attrs in scripts:
            assert attrs.get("type") == "module"
            assert attrs.get("src", "").startswith("/static/js/")
        assert all(body.strip() == "" for body in parsed.script_bodies), "no inline script bodies"

    @pytest.mark.parametrize("path", list(PAGES))
    def test_no_inline_event_handlers(self, client: TestClient, path: str) -> None:
        parsed = _parse(client.get(path).text)
        for tag, attrs in parsed.tags:
            handlers = [name for name in attrs if name.lower().startswith("on")]
            assert not handlers, f"<{tag}> has inline handler(s) {handlers}"
            for value in attrs.values():
                assert "javascript:" not in (value or "").lower()

    @pytest.mark.parametrize("path", list(PAGES))
    def test_no_third_party_urls(self, client: TestClient, path: str) -> None:
        html = client.get(path).text
        assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", html, re.I)


# ===========================================================================
# 3. JS / CSS hygiene
# ===========================================================================


def _imports(path: Path) -> list[Path]:
    source = path.read_text(encoding="utf-8")
    specs = re.findall(r"""^\s*import\s+(?:[^'"]+\s+from\s+)?['"]([^'"]+)['"]""", source, re.M)
    return [(path.parent / spec).resolve() for spec in specs]


class TestAssetHygiene:
    def test_module_graph_resolves(self) -> None:
        seen: set[Path] = set()
        pending = [WEB_DIR / "js" / "app.js", WEB_DIR / "js" / "dashboard.js"]
        while pending:
            module = pending.pop()
            if module in seen:
                continue
            seen.add(module)
            for target in _imports(module):
                assert target.is_file(), f"{module.name} imports missing {target}"
                assert target.is_relative_to(WEB_DIR / "js"), target
                pending.append(target)
        assert len(seen) >= 15

    @pytest.mark.parametrize("path", JS_FILES, ids=lambda p: p.name)
    def test_no_dynamic_code_or_unsafe_html_sinks(self, path: Path) -> None:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"\beval\s*\(", source)
        assert "new Function" not in source
        assert "document.write" not in source
        assert "insertAdjacentHTML" not in source
        assert "outerHTML" not in source
        assert not re.search(r"setAttribute\(\s*['\"]on", source)
        assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", source), "no remote URLs"

    def test_inner_html_only_receives_renderer_output(self) -> None:
        for path in JS_FILES:
            for line in path.read_text(encoding="utf-8").splitlines():
                if ".innerHTML" not in line:
                    continue
                assert path.name == "messages.js", f"innerHTML used in {path.name}"
                assert re.search(r"\.innerHTML = (renderMarkdown\(|partial;)", line), line

    @pytest.mark.parametrize("path", CSS_FILES, ids=lambda p: p.name)
    def test_css_has_no_remote_resources(self, path: Path) -> None:
        css = path.read_text(encoding="utf-8")
        assert "@import" not in css
        assert not re.search(r"url\(\s*['\"]?(https?:)?//", css)
        assert "fonts.googleapis" not in css


# ===========================================================================
# 4. Development mock API (the UI's contract test double)
# ===========================================================================


def _sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = "\n".join(line[6:] for line in lines if line.startswith("data: "))
        events.append((event, json.loads(data)))
    return events


@pytest.fixture()
def mock_client() -> TestClient:
    return TestClient(create_mock_app(stream_delay=0))


def _session(client: TestClient) -> tuple[str, dict[str, str]]:
    data = client.post("/api/sessions").json()
    return data["session_id"], {"Authorization": f"Bearer {data['session_token']}"}


class TestDevMockServer:
    @pytest.mark.parametrize(
        ("message", "action"),
        [
            ("I have chest pain right now", "ESCALATE"),
            ("Tell me about the French Revolution", "ABSTAIN"),
            ("What dose of amlodipine is right?", "SOFT-ASK"),
            ("Which diet helps?", "GENERALIZE"),
            ("My BP was 145/92, is that high?", "ANSWER"),
        ],
    )
    def test_stream_emits_stages_tokens_and_valid_done(self, mock_client: TestClient, message: str, action: str) -> None:
        sid, headers = _session(mock_client)
        resp = mock_client.post(f"/api/sessions/{sid}/messages/stream", json={"message": message}, headers=headers)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _sse_events(resp.text)
        kinds = [kind for kind, _ in events]
        assert kinds[0] == "stage" and kinds[-1] == "done"
        assert "token" in kinds
        stage = events[0][1]
        assert {"stage", "status", "label"} <= set(stage)
        done = MessageResponse.model_validate(events[-1][1])
        assert done.action.value == action
        assert done.explanation is not None
        assert "".join(d["text"] for k, d in events if k == "token") == done.response_text
        if action == "ESCALATE":
            skipped = {d["stage"] for k, d in events if k == "stage" and d["status"] == "skipped"}
            assert {"extraction", "context_gate", "generation"} <= skipped

    def test_session_routes_require_the_bearer_token(self, mock_client: TestClient) -> None:
        sid, headers = _session(mock_client)
        assert mock_client.get(f"/api/sessions/{sid}/history").status_code == 401
        assert mock_client.get(f"/api/sessions/{sid}/history", headers=headers).status_code == 200

    def test_bp_crisis_flag_and_csp_header(self, mock_client: TestClient) -> None:
        sid, headers = _session(mock_client)
        resp = mock_client.post(f"/api/sessions/{sid}/bp", json={"systolic": 180, "diastolic": 100}, headers=headers)
        assert resp.status_code == 201
        body = resp.json()
        assert body["crisis"] is True
        assert body["reading"]["category"] == "crisis"
        context = mock_client.get(f"/api/sessions/{sid}/context", headers=headers).json()
        assert context["bp_readings"][0]["recorded_at"]
        page = mock_client.get("/")
        assert "script-src 'self'" in page.headers["content-security-policy"]

    def test_admin_endpoints_need_login(self, mock_client: TestClient) -> None:
        assert mock_client.get("/api/metrics").status_code == 401
        token = mock_client.post("/api/admin/login", json={"username": "admin", "password": "admin"}).json()["token"]
        metrics = mock_client.get("/api/metrics", headers={"Authorization": f"Bearer {token}"}).json()
        assert metrics["turns"] > 0
        assert {"final_actions", "gate_1", "gate_2", "latency", "stage_latency", "feedback"} <= set(metrics)
        audit = mock_client.get("/api/admin/audit?limit=5", headers={"Authorization": f"Bearer {token}"}).json()
        assert len(audit["items"]) == 5
