"""
Capture UI screenshots of the MedBridge SPA and dashboard with Playwright.

Usage (Chrome must be installed; Playwright browsers are NOT downloaded):

    python -m medbridge.web.dev_mock_server &           # or the real backend
    python -m medbridge.web.screenshots --base-url http://127.0.0.1:8100 \
        --out medbridge/web/dev_screenshots --admin-user admin --admin-password admin

Scenes (each at 1440x900 and 390x844, light and dark): welcome, consent,
live pipeline stepper mid-stream, ANSWER with the source drawer open, "Why this answer?" expanded (clinician
mode), SOFT-ASK, ESCALATE, ABSTAIN, context panel with the BP chart, and the
admin dashboard. Console errors (including CSP violations) are collected and
printed; the exit code is 1 if any occurred.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Browser, Page, sync_playwright

ANSWER_PROMPT = "What lifestyle changes help lower blood pressure?"
SOFT_ASK_PROMPT = "What dose of amlodipine should I take?"
ESCALATE_PROMPT = "I have crushing chest pain and my BP is 190/125"
ABSTAIN_PROMPT = "Tell me about the French Revolution"
READINGS = [(128, 82), (136, 86), (142, 91), (151, 94), (139, 88), (133, 84)]
VIEWPORTS = {"desktop": (1440, 900), "mobile": (390, 844)}
ANSWER_TIMEOUT_MS = 180_000


@dataclass
class Capture:
    """Shared state for one capture run."""

    base_url: str
    out: Path
    admin_user: str
    admin_password: str
    errors: list[str] = field(default_factory=list)


def _new_page(browser: Browser, cap: Capture, viewport: tuple[int, int], theme: str, *, consent: bool) -> Page:
    context = browser.new_context(
        viewport={"width": viewport[0], "height": viewport[1]},
        device_scale_factor=2 if viewport[0] < 600 else 1,
        color_scheme="dark" if theme == "dark" else "light",
        reduced_motion="reduce",
    )
    if consent:
        context.add_init_script("try { localStorage.setItem('medbridge.consent', '1'); } catch (e) {}")
    page = context.new_page()
    page.on("console", lambda msg: cap.errors.append(f"{msg.type}: {msg.text}") if msg.type == "error" else None)
    page.on("pageerror", lambda exc: cap.errors.append(f"pageerror: {exc}"))
    return page


def _ask(page: Page, prompt: str, on_streaming: Callable[[], None] | None = None) -> None:
    """Send a prompt and wait until the answer is final (``on_streaming`` runs mid-stream)."""
    count = page.locator(".msg--assistant:not(.is-streaming)").count()
    page.fill("#composer-input", prompt)
    page.keyboard.press("Enter")
    if on_streaming is not None:
        page.wait_for_selector(".steps.is-running .step", timeout=ANSWER_TIMEOUT_MS)
        page.wait_for_timeout(600)
        on_streaming()
    page.wait_for_function(
        "n => document.querySelectorAll('.msg--assistant:not(.is-streaming)').length > n",
        arg=count,
        timeout=ANSWER_TIMEOUT_MS,
    )
    page.wait_for_timeout(300)


def _new_chat(page: Page, mobile: bool) -> None:
    if mobile:
        page.click("#sidebar-toggle")
    page.click("#new-chat")
    page.wait_for_selector("#welcome:not([hidden])")


def _shot(page: Page, cap: Capture, name: str) -> None:
    path = cap.out / f"{name}.png"
    page.screenshot(path=str(path))
    print(f"saved {path}")


def capture_chat(browser: Browser, cap: Capture, label: str, viewport: tuple[int, int], theme: str) -> None:
    """All chat scenes for one viewport/theme."""
    mobile = viewport[0] < 900
    tag = f"{label}-{theme}"

    first = _new_page(browser, cap, viewport, theme, consent=False)
    first.goto(cap.base_url + "/")
    first.wait_for_selector(".modal--consent")
    first.wait_for_timeout(250)
    _shot(first, cap, f"consent-{tag}")
    first.context.close()

    page = _new_page(browser, cap, viewport, theme, consent=True)
    page.goto(cap.base_url + "/")
    page.wait_for_selector(".suggestion")
    _shot(page, cap, f"welcome-{tag}")

    _ask(page, ANSWER_PROMPT, on_streaming=lambda: _shot(page, cap, f"streaming-{tag}"))
    chips = page.locator(".msg--assistant").last.locator(".cite-chip")
    if chips.count():  # Gate 3 may strip every cited sentence; then there is no drawer to show
        chips.first.click()
        page.wait_for_selector("#source-drawer.is-open")
        page.wait_for_timeout(250)
        _shot(page, cap, f"answer-sources-{tag}")
        page.keyboard.press("Escape")
    else:
        print(f"no citation chips in the answer ({tag}); source-drawer shot skipped")

    page.click('[data-mode-option="clinician"]')
    page.locator(".msg--assistant").last.locator("details.why").scroll_into_view_if_needed()
    page.locator(".msg--assistant").last.locator(".why__body").evaluate("el => el.scrollIntoView({block: 'start'})")
    page.wait_for_timeout(300)
    _shot(page, cap, f"why-expanded-{tag}")
    page.click('[data-mode-option="patient"]')

    for prompt, name in ((SOFT_ASK_PROMPT, "softask"), (ESCALATE_PROMPT, "escalate"), (ABSTAIN_PROMPT, "abstain")):
        _new_chat(page, mobile)
        _ask(page, prompt)
        page.locator(".msg--assistant").last.evaluate("el => el.scrollIntoView({block: 'start'})")
        page.wait_for_timeout(200)
        _shot(page, cap, f"{name}-{tag}")

    page.evaluate(
        """async (readings) => {
            const list = JSON.parse(localStorage.getItem('medbridge.conversations'));
            const conv = list.sort((a, b) => b.updatedAt - a.updatedAt)[0];
            const day = 86400000;
            for (let i = 0; i < readings.length; i++) {
                const [systolic, diastolic] = readings[i];
                await fetch(`/api/sessions/${conv.id}/bp`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', Authorization: `Bearer ${conv.token}`},
                    body: JSON.stringify({systolic, diastolic, pulse: 70 + i,
                        measured_at: new Date(Date.now() - (readings.length - i) * day).toISOString()}),
                });
            }
        }""",
        READINGS,
    )
    page.click("#context-toggle")
    page.wait_for_selector(".bp-chart svg")
    page.wait_for_timeout(400)
    _shot(page, cap, f"context-{tag}")
    page.context.close()


def capture_dashboard(browser: Browser, cap: Capture, label: str, viewport: tuple[int, int], theme: str) -> None:
    """Login and dashboard scenes."""
    page = _new_page(browser, cap, viewport, theme, consent=True)
    page.goto(cap.base_url + "/dashboard")
    page.wait_for_selector("#login-form")
    if label == "desktop":
        _shot(page, cap, f"dashboard-login-{label}-{theme}")
    page.fill("#username", cap.admin_user)
    page.fill("#password", cap.admin_password)
    page.click("#login-submit")
    page.wait_for_selector("#metrics-view:not([hidden]) .kpi")
    page.wait_for_timeout(400)
    page.screenshot(path=str(cap.out / f"dashboard-{label}-{theme}.png"), full_page=True)
    print(f"saved {cap.out / f'dashboard-{label}-{theme}.png'}")
    page.context.close()


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "dev_screenshots")
    parser.add_argument("--admin-user", default="admin")
    parser.add_argument("--admin-password", default="admin")
    parser.add_argument("--only", choices=["chat", "dashboard"], default=None)
    parser.add_argument("--viewport", choices=sorted(VIEWPORTS), default=None)
    parser.add_argument("--theme", choices=["light", "dark"], default=None)
    args = parser.parse_args(argv)
    cap = Capture(args.base_url.rstrip("/"), args.out, args.admin_user, args.admin_password)
    cap.out.mkdir(parents=True, exist_ok=True)
    viewports = {args.viewport: VIEWPORTS[args.viewport]} if args.viewport else VIEWPORTS
    themes = [args.theme] if args.theme else ["light", "dark"]
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        try:
            for label, size in viewports.items():
                for theme in themes:
                    if args.only in (None, "chat"):
                        capture_chat(browser, cap, label, size, theme)
                    if args.only in (None, "dashboard"):
                        capture_dashboard(browser, cap, label, size, theme)
        finally:
            browser.close()
    for error in cap.errors:
        print(f"CONSOLE {error}", file=sys.stderr)
    return 1 if cap.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
