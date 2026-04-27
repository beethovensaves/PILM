#!/usr/bin/env bash
# Download free PDFs of the prioritized literature into litt/.
# Paywalled-only papers (Wiley/Elsevier without preprints) are skipped — get them via UW library proxy.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LITT_DIR="$REPO_ROOT/litt"
mkdir -p "$LITT_DIR"

UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

fetch() {
    local url="$1"
    local out="$2"
    if [ -f "$LITT_DIR/$out" ]; then
        echo "  [skip] $out (exists)"
        return 0
    fi
    if curl -fsSL -A "$UA" --max-time 60 -o "$LITT_DIR/$out.tmp" "$url"; then
        # Verify it's a PDF (starts with %PDF)
        if head -c 4 "$LITT_DIR/$out.tmp" | grep -q '%PDF'; then
            mv "$LITT_DIR/$out.tmp" "$LITT_DIR/$out"
            echo "  [ ok ] $out"
        else
            rm -f "$LITT_DIR/$out.tmp"
            echo "  [FAIL non-pdf] $out  ($url)"
        fi
    else
        rm -f "$LITT_DIR/$out.tmp"
        echo "  [FAIL http]    $out  ($url)"
    fi
}

echo "=== Tier 1 (priority) ==="
fetch "https://pmc.ncbi.nlm.nih.gov/articles/PMC4538954/pdf/nihms-712852.pdf" \
      "2015_Alderson-Day_Fernyhough_inner_speech_review.pdf"
fetch "https://arxiv.org/pdf/2507.20091" \
      "2025_Lin_ProsodyLM.pdf"
fetch "https://arxiv.org/pdf/2109.03264" \
      "2022_Kharitonov_pGSLM.pdf"
fetch "https://arxiv.org/pdf/2507.03912" \
      "2025_Roll_phoneme_BERT_prosody.pdf"
fetch "https://janetdeanfodor.wordpress.com/wp-content/uploads/2016/06/fodor-2002-prosodic-disambiguation-in-silent-reading.pdf" \
      "2002_Fodor_prosodic_disambiguation_silent_reading.pdf"
fetch "https://personal.utdallas.edu/~assmann/hcs6367/honorof_whalen05.pdf" \
      "2005_Honorof_Whalen_pitch_F0_range.pdf"
fetch "http://proceedings.mlr.press/v97/kenter19a/kenter19a.pdf" \
      "2019_Kenter_CHiVE.pdf"

echo "=== Tier 2 ==="
fetch "https://www.pnas.org/doi/pdf/10.1073/pnas.2424400122" \
      "2025_PNAS_three_components_pragmatic.pdf"
fetch "https://kinderlab.bcs.rochester.edu/papers/KurumadaClark2016.pdf" \
      "2017_Kurumada_Clark_contrastive_prosody.pdf"
fetch "https://arxiv.org/pdf/2107.04734" \
      "2021_Pasad_layer_wise_SSL.pdf"
fetch "https://arxiv.org/pdf/2302.12057" \
      "2023_deSeyssel_ProsAudit.pdf"
fetch "https://arxiv.org/pdf/2410.00037" \
      "2024_Defossez_Moshi.pdf"

echo "=== Tier 3 ==="
fetch "https://proceedings.mlr.press/v80/skerry-ryan18a/skerry-ryan18a.pdf" \
      "2018_Skerry-Ryan_prosody_transfer_Tacotron.pdf"
fetch "https://arxiv.org/pdf/2106.07447" \
      "2021_Hsu_HuBERT.pdf"
fetch "https://arxiv.org/pdf/2110.13900" \
      "2022_Chen_WavLM.pdf"
fetch "https://arxiv.org/pdf/2006.11477" \
      "2020_Baevski_wav2vec2.pdf"
fetch "https://arxiv.org/pdf/2102.01192" \
      "2021_Lakhotia_GSLM.pdf"
fetch "https://arxiv.org/pdf/2402.05755" \
      "2024_Nguyen_SpiritLM.pdf"
fetch "https://arxiv.org/pdf/2601.19781" \
      "2026_phonological_tokenizer.pdf"
fetch "https://arxiv.org/pdf/2509.13068" \
      "2025_MSR_codec.pdf"
fetch "https://speechlab.cas.msu.edu/PDF/Dilley_Breen_Symp_2018.pdf" \
      "2018_Dilley_Breen_AM_plus.pdf"
fetch "https://aclanthology.org/2025.findings-acl.1041.pdf" \
      "2025_survey_LLM_speech_integration.pdf"
fetch "https://arxiv.org/pdf/2510.00499" \
      "2025_MOSS_Speech.pdf"

echo
echo "=== Summary ==="
echo "Downloaded:"
ls -1 "$LITT_DIR"/*.pdf 2>/dev/null | wc -l | xargs echo "  files:"
du -sh "$LITT_DIR"/*.pdf 2>/dev/null | tail -n +1 | awk '{ total += $1 } END { print "  total: ~"NR" PDFs" }'
echo
echo "Files saved to: $LITT_DIR"
