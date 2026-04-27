#!/usr/bin/env bash
# Fetch publisher PDFs that need institutional access.
#
# !!! KEPT FOR DOCUMENTATION ONLY — DOES NOT WORK !!!
#
# Attempted 2026-04-25 with UW VPN active on the user's machine. All 9
# fetches failed: PNAS, Wiley, Tandfonline, Elsevier, APA, and MIT DSpace
# all sit behind Cloudflare or equivalent JS-challenge bot detection
# (cf-mitigated: challenge in headers). curl cannot pass these challenges
# regardless of headers, cookies, or institutional IP — the challenge
# fires before the publisher sees the request's IP.
#
# Path forward: browser-download the 9 papers manually via litt/README.md.
# Or, if this comes up regularly: switch to a Playwright-based downloader
# that runs a real browser. Overkill for current scope.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LITT_DIR="$REPO_ROOT/litt"
mkdir -p "$LITT_DIR"

UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
ACCEPT='text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.9,*/*;q=0.8'

ok=0
fail=0
fail_list=()

# Visit landing → fetch PDF with same cookie jar.
fetch_chain() {
    local outname="$1" landing="$2" pdf="$3"
    if [ -f "$LITT_DIR/$outname" ]; then
        echo "  [skip] $outname"
        ok=$((ok+1))
        return 0
    fi
    local jar
    jar="$(mktemp -t pilm_jar.XXXXXX)"

    curl -sL -A "$UA" -H "Accept: $ACCEPT" -c "$jar" -b "$jar" \
        --max-time 45 -o /dev/null "$landing" 2>/dev/null || true

    if curl -fsSL -A "$UA" -H "Accept: $ACCEPT" -e "$landing" \
        -c "$jar" -b "$jar" --max-time 120 \
        -o "$LITT_DIR/$outname.tmp" "$pdf" 2>/dev/null; then
        if head -c 4 "$LITT_DIR/$outname.tmp" | grep -q '%PDF'; then
            mv "$LITT_DIR/$outname.tmp" "$LITT_DIR/$outname"
            echo "  [ ok ] $outname"
            ok=$((ok+1))
            rm -f "$jar"
            return 0
        fi
    fi
    rm -f "$LITT_DIR/$outname.tmp" "$jar"
    echo "  [FAIL] $outname"
    fail_list+=("$outname")
    fail=$((fail+1))
    return 1
}

echo "=== Retrying earlier failures ==="

# PNAS — IP-based access via UW should serve the PDF
fetch_chain "2025_PNAS_three_components_pragmatic.pdf" \
    "https://www.pnas.org/doi/10.1073/pnas.2424400122" \
    "https://www.pnas.org/doi/pdf/10.1073/pnas.2424400122?download=true"

# Alderson-Day & Fernyhough 2015 — try APA via UW; PMC link is non-PDF interstitial
fetch_chain "2015_Alderson-Day_Fernyhough_inner_speech_review.pdf" \
    "https://psycnet.apa.org/doiLanding?doi=10.1037%2Fbul0000021" \
    "https://psycnet.apa.org/manuscript/2015-25399-001.pdf"

echo "=== Paywalled / publisher-direct via VPN ==="

# Breen 2014 — Wiley
fetch_chain "2014_Breen_implicit_prosody_review.pdf" \
    "https://onlinelibrary.wiley.com/doi/10.1111/lnc3.12061" \
    "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/lnc3.12061"

# Cole 2015 — Tandfonline
fetch_chain "2015_Cole_prosody_in_context.pdf" \
    "https://www.tandfonline.com/doi/full/10.1080/23273798.2014.963130" \
    "https://www.tandfonline.com/doi/pdf/10.1080/23273798.2014.963130?download=true"

# 2025 gating study — Tandfonline
fetch_chain "2025_speech_act_gating_prosody.pdf" \
    "https://www.tandfonline.com/doi/full/10.1080/23273798.2025.2506641" \
    "https://www.tandfonline.com/doi/pdf/10.1080/23273798.2025.2506641?download=true"

# Wilson & Wharton 2006 — Elsevier ScienceDirect (DOI redirect to PII)
fetch_chain "2006_Wilson_Wharton_relevance_prosody.pdf" \
    "https://doi.org/10.1016/j.pragma.2005.04.012" \
    "https://www.sciencedirect.com/science/article/pii/S0378216606000324/pdfft?download=true"

# Cheang & Pell 2008 — Elsevier ScienceDirect
fetch_chain "2008_Cheang_Pell_sound_of_sarcasm.pdf" \
    "https://doi.org/10.1016/j.specom.2007.11.003" \
    "https://www.sciencedirect.com/science/article/pii/S0167639307001550/pdfft?download=true"

# Banse & Scherer 1996 — APA
fetch_chain "1996_Banse_Scherer_vocal_emotion_acoustics.pdf" \
    "https://psycnet.apa.org/doiLanding?doi=10.1037%2F0022-3514.70.3.614" \
    "https://psycnet.apa.org/manuscript/1996-04263-008.pdf"

# Pierrehumbert 1980 thesis — MIT DSpace
fetch_chain "1980_Pierrehumbert_intonation_thesis.pdf" \
    "https://dspace.mit.edu/handle/1721.1/16065" \
    "https://dspace.mit.edu/bitstream/handle/1721.1/16065/13836172-MIT.pdf"

echo
echo "=== Summary ==="
echo "  succeeded: $ok"
echo "  failed:    $fail"
if [ $fail -gt 0 ]; then
    echo "  failures:"
    for f in "${fail_list[@]}"; do echo "    - $f"; done
fi
echo
ls -1 "$LITT_DIR"/*.pdf | wc -l | xargs echo "PDFs in litt/:"
du -sh "$LITT_DIR"
