"""
Playwright-based downloader for the paywalled / JS-protected papers
that defeat curl-based scraping.

Assumes UW VPN (Husky OnNet) is active so the source IP has institutional
publisher access. Run from the repo root:

    .venv/bin/python scripts/download_paywalled.py            # headed (default)
    .venv/bin/python scripts/download_paywalled.py --headless # quieter re-runs
    .venv/bin/python scripts/download_paywalled.py --only wiley pnas    # filter

Tested 2026-04-25 — what works and what doesn't:

    Works (auto):  pnas, dspace
    Blocked:       wiley, tandfonline, elsevier (all behind Cloudflare bot
                   management that detects Playwright-driven Chromium even
                   with --disable-blink-features=AutomationControlled and
                   institutional VPN). Page hangs at "Just a moment..." for
                   the full timeout. Real browsers pass; Playwright doesn't.
                   Beating this would require playwright-stealth + likely
                   still be flaky.
    Interstitial:  pmc (serves an HTML confirm page before the PDF link),
                   apa (login wall regardless of UW IP).

For Cloudflare-protected papers, open the landing URL in your real browser
and click the publisher's PDF button — once. UW VPN is enough.

Strategy for the ones that work: navigate to landing, then try each
candidate PDF URL with a fetch path that handles three server behaviors:

    (a) inline PDF — Content-Type: application/pdf, no attachment disposition
    (b) attachment download — Content-Disposition: attachment
    (c) inline-PDF that the viewer immediately disposes (resp.body() race)

A download listener stays attached for (b) and (c).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urljoin

from playwright.sync_api import Page, sync_playwright

OUT_DIR = Path(__file__).resolve().parent.parent / "litt"

# (landing_url, output_filename, publisher_key)
PAPERS: list[tuple[str, str, str]] = [
    (
        "https://onlinelibrary.wiley.com/doi/10.1111/lnc3.12061",
        "2014_Breen_implicit_prosody_review.pdf",
        "wiley",
    ),
    (
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC4538954/",
        "2015_Alderson-Day_Fernyhough_inner_speech_review.pdf",
        "pmc",
    ),
    (
        "https://www.pnas.org/doi/10.1073/pnas.2424400122",
        "2025_PNAS_three_components_pragmatic.pdf",
        "pnas",
    ),
    (
        "https://www.tandfonline.com/doi/full/10.1080/23273798.2025.2506641",
        "2025_speech_act_gating_prosody.pdf",
        "tandfonline",
    ),
    (
        "https://www.tandfonline.com/doi/full/10.1080/23273798.2014.963130",
        "2015_Cole_prosody_in_context.pdf",
        "tandfonline",
    ),
    (
        "https://doi.org/10.1016/j.pragma.2005.04.012",
        "2006_Wilson_Wharton_relevance_prosody.pdf",
        "elsevier",
    ),
    (
        "https://doi.org/10.1016/j.specom.2007.11.003",
        "2008_Cheang_Pell_sound_of_sarcasm.pdf",
        "elsevier",
    ),
    (
        "https://psycnet.apa.org/doiLanding?doi=10.1037%2F0022-3514.70.3.614",
        "1996_Banse_Scherer_vocal_emotion_acoustics.pdf",
        "apa",
    ),
    (
        "https://dspace.mit.edu/handle/1721.1/16065",
        "1980_Pierrehumbert_intonation_thesis.pdf",
        "dspace",
    ),
]

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

DOM_SELECTORS: dict[str, list[str]] = {
    "wiley": ['a[href*="/doi/pdfdirect/"]', 'a[href*="/doi/pdf/"]', 'a[title="PDF"]'],
    "pmc": [
        'a[href$=".pdf"]',
        'a:has-text("Download PDF")',
        'a[data-ga-action="click_pdf_download"]',
    ],
    "pnas": ['a[data-test="download-pdf"]', 'a[href*="/doi/pdf/"]'],
    "tandfonline": ['a[href*="/doi/pdf/"]', 'a:has-text("Download PDF")'],
    "elsevier": [
        'a:has-text("View PDF")',
        'a[aria-label*="PDF" i]',
        'a[href*="pdfft"]',
        'a[href*=".pdf"]',
    ],
    "apa": ['a:has-text("Full Text PDF")', 'a[href$=".pdf"]'],
    "dspace": [
        'a[href*="/bitstream/"]',
        'a:has-text("Download")',
        'a[href$=".pdf"]',
    ],
}


def wait_past_cloudflare(page: Page, timeout_ms: int = 30_000) -> bool:
    """Cloudflare's bot-check renders a 'Just a moment...' page that auto-redirects
    to the real content after the JS challenge passes. Block until the title
    no longer matches the interstitial, or until timeout."""
    try:
        page.wait_for_function(
            "() => !/Just a moment|Cloudflare|Checking your browser/i.test(document.title)",
            timeout=timeout_ms,
        )
        # Once title flips, also let the real page settle a bit.
        try:
            page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass
        return True
    except Exception:
        return False


def is_valid_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    return path.read_bytes()[:4] == b"%PDF"


def predicted_pdf_urls(landing_url: str, publisher: str) -> list[str]:
    if publisher == "wiley":
        m = re.search(r"/doi/(.+)", landing_url)
        if m:
            doi = m.group(1)
            return [
                f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}?download=true",
                f"https://onlinelibrary.wiley.com/doi/pdf/{doi}",
            ]
    if publisher == "tandfonline":
        m = re.search(r"/doi/full/(.+)", landing_url)
        if m:
            doi = m.group(1)
            return [f"https://www.tandfonline.com/doi/pdf/{doi}?download=true"]
    if publisher == "pnas":
        m = re.search(r"/doi/(.+)", landing_url)
        if m:
            doi = m.group(1)
            return [f"https://www.pnas.org/doi/pdf/{doi}?download=true"]
    return []


def find_pdf_link_in_dom(page: Page, publisher: str) -> str | None:
    for sel in DOM_SELECTORS.get(publisher, ['a[href$=".pdf"]']):
        link = page.query_selector(sel)
        if not link:
            continue
        href = link.get_attribute("href")
        if not href:
            continue
        return urljoin(page.url, href)
    return None


def fetch_pdf_to(page: Page, pdf_url: str, out: Path) -> bool:
    """Navigate to a PDF URL and save the result.

    Three server behaviors handled:
      (a) inline PDF (Content-Type: application/pdf, no attachment)
          — page.goto returns; we read response.body().
      (b) attachment (Content-Disposition: attachment)
          — page.goto throws 'Download is starting'; download event fires.
      (c) inline PDF that Chromium's viewer immediately disposes
          — page.goto returns but resp.body() raises 'No resource with given
            identifier found'. We rely on the same download listener.

    A download listener is attached for the duration of the call so cases
    (b) and (c) are both covered without timing-dependent ordering.
    """
    download_obj: list = []  # mutable container for closure

    def _on_download(d):
        download_obj.append(d)

    page.on("download", _on_download)
    try:
        resp = None
        try:
            resp = page.goto(pdf_url, wait_until="domcontentloaded", timeout=45_000)
        except Exception as e:
            if "Download is starting" not in str(e):
                print(f"    nav error: {e!s:.140}")

        # Brief settle for the download event to fire.
        page.wait_for_timeout(800)

        if download_obj:
            download_obj[0].save_as(out)
            if is_valid_pdf(out):
                return True
            out.unlink(missing_ok=True)
            print("    captured download did not validate as PDF")
            return False

        if resp is None:
            return False
        if not resp.ok:
            print(f"    HTTP {resp.status}")
            return False
        ct = (resp.headers.get("content-type") or "").lower()
        try:
            body = resp.body()
        except Exception as e:
            print(f"    body read error: {e!s:.140}")
            return False
        if "pdf" in ct or body[:4] == b"%PDF":
            out.write_bytes(body)
            if is_valid_pdf(out):
                return True
            out.unlink(missing_ok=True)
            print("    bytes did not validate as PDF")
            return False
        print(f"    not a PDF (ct={ct!r})")
        return False
    finally:
        page.remove_listener("download", _on_download)


def click_capture(page: Page, selector: str, out: Path) -> bool:
    try:
        with page.expect_download(timeout=20_000) as dl_info:
            page.click(selector, timeout=15_000)
        dl = dl_info.value
        dl.save_as(out)
        if is_valid_pdf(out):
            return True
        out.unlink(missing_ok=True)
    except Exception as e:
        print(f"    click {selector!r}: {e!s:.140}")
    return False


def download_one(page: Page, paper: tuple[str, str, str]) -> bool:
    landing_url, filename, publisher = paper
    out = OUT_DIR / filename
    if is_valid_pdf(out):
        print(f"[skip ] {filename}  (already present)")
        return True

    print(f"[fetch] {filename}")
    print(f"        landing: {landing_url}")

    try:
        page.goto(landing_url, wait_until="domcontentloaded", timeout=60_000)
    except Exception as e:
        print(f"    landing failed: {e!s:.140}")
        return False

    title = page.title()
    if "Just a moment" in title or "Cloudflare" in title:
        print(f"        Cloudflare interstitial — waiting for clear...")
        if not wait_past_cloudflare(page, timeout_ms=30_000):
            print(f"        ⚠ Cloudflare did not clear; trying anyway")
    else:
        try:
            page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass

    print(f"        on:      {page.url}")
    print(f"        title:   {page.title()[:80]!r}")

    for pdf_url in predicted_pdf_urls(landing_url, publisher):
        print(f"    predicted -> {pdf_url}")
        if fetch_pdf_to(page, pdf_url, out):
            print("    ✓ saved (predicted URL)")
            return True
        # bring page back to landing for any subsequent DOM scan
        try:
            page.goto(landing_url, wait_until="domcontentloaded", timeout=30_000)
        except Exception:
            pass

    href = find_pdf_link_in_dom(page, publisher)
    if href:
        print(f"    DOM link  -> {href}")
        if fetch_pdf_to(page, href, out):
            print("    ✓ saved (DOM link)")
            return True
        try:
            page.goto(landing_url, wait_until="domcontentloaded", timeout=30_000)
        except Exception:
            pass

    for sel in DOM_SELECTORS.get(publisher, []):
        if not page.query_selector(sel):
            continue
        print(f"    click+capture -> {sel}")
        if click_capture(page, sel, out):
            print("    ✓ saved (click+download)")
            return True

    print("    ✗ no working PDF source")
    return False


def main() -> None:
    args = sys.argv[1:]
    headless = "--headless" in args
    only_pubs: set[str] = set()
    if "--only" in args:
        idx = args.index("--only")
        only_pubs = {a for a in args[idx + 1 :] if not a.startswith("--")}

    OUT_DIR.mkdir(exist_ok=True)
    todo = [p for p in PAPERS if not only_pubs or p[2] in only_pubs]
    failed: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            accept_downloads=True,
            user_agent=USER_AGENT,
        )
        page = context.new_page()
        for paper in todo:
            ok = download_one(page, paper)
            if not ok:
                failed.append(paper[1])
            print()
        browser.close()

    n_ok = len(todo) - len(failed)
    print(f"Done. {n_ok}/{len(todo)} saved.")
    if failed:
        print("Manual download still needed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
