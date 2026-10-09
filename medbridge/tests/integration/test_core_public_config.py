"""Tests for ``GET /api/config/public`` (medbridge/api/routes/config.py)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from medbridge.api.routes import config as config_module
from medbridge.api.routes.config import (
    SUGGESTED_PROMPTS,
    SUGGESTIONS,
    PublicConfig,
    build_disclaimer,
    config_router,
    load_corpus_summary,
)
from medbridge.api.routes.health import APP_VERSION
from medbridge.config import get_settings


def client() -> TestClient:
    app = FastAPI()
    app.include_router(config_router, prefix="/api")
    return TestClient(app)


def test_public_config_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "CORPUS_MANIFEST_PATH", tmp_path / "missing.json")
    response = client().get("/api/config/public")
    assert response.status_code == 200
    body = PublicConfig.model_validate(response.json())
    settings = get_settings()
    assert body.app_name == settings.APP_NAME
    assert body.version == APP_VERSION
    assert body.emergency_number == settings.EMERGENCY_NUMBER
    assert body.ambulance_number == settings.AMBULANCE_NUMBER
    assert body.emergency_region_label == settings.EMERGENCY_REGION_LABEL
    assert body.local_only_mode == settings.LOCAL_ONLY_MODE
    assert body.models == {"gates": settings.gates_model, "generator": settings.generator_model}
    assert len(body.suggested_prompts) == 6 == len(SUGGESTED_PROMPTS)
    assert body.corpus is None
    assert settings.EMERGENCY_NUMBER in body.disclaimer and "911" in body.disclaimer


def test_public_config_has_no_secrets():
    text = client().get("/api/config/public").text.lower()
    for secret in ("password", "secret", "api_key", "database_url", "encryption"):
        assert secret not in text


def test_corpus_summary_from_manifest(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "title": "WHO 2021 Hypertension Guideline",
                        "org": "WHO",
                        "year": 2021,
                        "licence": "CC BY-NC-SA 3.0 IGO",
                        "sha256": "x",
                    },
                    {"title": "ESC 2024 Hypertension", "org": "ESC", "year": "2024"},
                    {"org": "no title"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "CORPUS_MANIFEST_PATH", manifest)
    body = client().get("/api/config/public").json()
    assert body["corpus"] == {
        "documents": 2,
        "items": [
            {"title": "WHO 2021 Hypertension Guideline", "org": "WHO", "year": 2021},
            {"title": "ESC 2024 Hypertension", "org": "ESC", "year": None},
        ],
    }


def test_corpus_summary_accepts_list_and_rejects_bad_json(tmp_path: Path):
    listed = tmp_path / "list.json"
    listed.write_text(json.dumps([{"title": "NICE NG136", "org": "NICE", "year": 2023}]))
    summary = load_corpus_summary(listed)
    assert summary is not None and summary.documents == 1

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert load_corpus_summary(broken) is None
    assert load_corpus_summary(tmp_path / "absent.json") is None


def test_disclaimer_uses_custom_numbers():
    custom = get_settings().model_copy(
        update={
            "EMERGENCY_NUMBER": "999",
            "AMBULANCE_NUMBER": "999",
            "EMERGENCY_REGION_LABEL": "the UK",
        }
    )
    text = build_disclaimer(custom)
    assert "call 999 (or 999 for an ambulance in the UK)" in text
    assert "does not diagnose" in text


def test_public_config_suggestions_have_title_and_prompt():
    body = client().get("/api/config/public").json()
    suggestions = body["suggestions"]
    assert len(suggestions) == 6 == len(SUGGESTIONS)
    assert all(set(item) == {"title", "prompt"} for item in suggestions)
    assert all(item["title"].strip() and item["prompt"].strip() for item in suggestions)
    # the plain list stays in sync for older clients
    assert [item["prompt"] for item in suggestions] == body["suggested_prompts"]
