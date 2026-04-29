#!/bin/zsh
# v9 overnight orchestrator:
#   1. Wait for predictor training (PID-based) to finish.
#   2. Render the test text at multiple emotion + sampling settings.
#   3. Save audio to v9/outputs/overnight/<timestamp>/.
#   4. Print a summary at the end so the morning user can see what got saved.
#
# Designed to run unattended in background. Logs to v9/logs/overnight_*.log.

set -u
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

TS=$(date +%Y%m%d_%H%M%S)
LOG="v9/logs/overnight_${TS}.log"
OUT="v9/outputs/overnight_${TS}"
mkdir -p v9/logs "$OUT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ── Step 1: wait for training process to finish ────────────────────────
TRAIN_PID="$1"
if [[ -z "${TRAIN_PID:-}" ]]; then
    # Auto-detect the running predictor training PID
    TRAIN_PID=$(pgrep -f "train_v9_predictor.py" | head -1)
fi
if [[ -z "$TRAIN_PID" ]]; then
    log "no training PID found — proceeding directly to inference"
else
    log "waiting for predictor training (PID $TRAIN_PID) to finish"
    while ps -p "$TRAIN_PID" > /dev/null 2>&1; do
        sleep 60
    done
    log "training process exited"
fi

# Sanity-check checkpoints exist
TOK="v9/checkpoints/phoneme_rvq/best.pt"
PRED="v9/checkpoints/predictor/best.pt"
for f in "$TOK" "$PRED"; do
    if [[ ! -f "$f" ]]; then
        log "missing required checkpoint: $f"
        exit 1
    fi
done
log "tokenizer: $TOK   predictor: $PRED"

# ── Step 2: inference renders ───────────────────────────────────────────
TEXT='You know, I tried that the other day, and it actually made me feel a lot better.'
TEXT2='What an amazing surprise! I cannot believe you came all this way.'
TEXT3='I dont want to talk about it right now.'

# Default speaker (from DEFAULT_REFERENCE in synthesize_v9.py).
# Optional: render with an RDJ speaker if available.
RDJ_SPK="reference_audio/rdj/rdj_spk_emb.npy"

render() {
    local label="$1" text="$2" emo="$3" intensity="$4" temp="$5" cfg="$6" spk_arg="$7"
    local out="${OUT}/${label}.wav"
    log "render: $label  emo=$emo intensity=$intensity temp=$temp cfg=$cfg"
    python -u v9/inference/synthesize_v9.py "$text" \
        --tokenizer-checkpoint "$TOK" --predictor-checkpoint "$PRED" \
        --emotion "$emo" --intensity "$intensity" \
        --temperature "$temp" --cfg-scale "$cfg" \
        --device cpu --output "$out" $spk_arg 2>&1 | tee -a "$LOG" | tail -3
}

# Test 1 — same text across emotions (default speaker)
render "t1_neutral_05"      "$TEXT" neutral 0.5 1.0 1.0 ""
render "t1_happy_08"        "$TEXT" happy   0.8 1.0 1.0 ""
render "t1_sad_07"          "$TEXT" sad     0.7 1.0 1.0 ""
render "t1_angry_09"        "$TEXT" angry   0.9 1.0 1.0 ""
render "t1_surprise_08"     "$TEXT" surprise 0.8 1.0 1.0 ""

# Test 2 — temperature ablation (neutral, default speaker)
render "t2_neutral_temp00"  "$TEXT" neutral 0.5 0.0 1.0 ""
render "t2_neutral_temp07"  "$TEXT" neutral 0.5 0.7 1.0 ""
render "t2_neutral_temp10"  "$TEXT" neutral 0.5 1.0 1.0 ""

# Test 3 — CFG ablation (happy at high intensity)
render "t3_happy_cfg10"     "$TEXT" happy 0.9 1.0 1.0 ""
render "t3_happy_cfg20"     "$TEXT" happy 0.9 1.0 2.0 ""
render "t3_happy_cfg30"     "$TEXT" happy 0.9 1.0 3.0 ""

# Test 4 — different texts, neutral
render "t4_text2_happy"     "$TEXT2" happy 0.9 1.0 1.5 ""
render "t4_text3_sad"       "$TEXT3" sad   0.8 1.0 1.5 ""

# RDJ if available
if [[ -f "$RDJ_SPK" ]]; then
    render "t5_rdj_neutral"  "$TEXT" neutral  0.5 1.0 1.0 "--speaker-emb $RDJ_SPK"
    render "t5_rdj_happy"    "$TEXT" happy    0.8 1.0 1.5 "--speaker-emb $RDJ_SPK"
    render "t5_rdj_sad"      "$TEXT" sad      0.7 1.0 1.5 "--speaker-emb $RDJ_SPK"
fi

log "done. Outputs in $OUT"
ls -la "$OUT" | tee -a "$LOG"
