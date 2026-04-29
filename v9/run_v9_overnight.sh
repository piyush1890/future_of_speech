#!/bin/zsh
# v9 overnight orchestrator (two-stage rebuild):
#   1. Train Stage 1: V9PerPhonemeStyleEncoder + V9StyleCodebook + V9Renderer (joint).
#   2. Extract per-phoneme style codes from trained encoder+codebook.
#   3. Train Stage 2: V9StylePlanner.
#   4. Render audio at multiple emotion + cfg settings.
#
# All stages run sequentially. Each stage must succeed before next runs.
# Logs to v9/logs/overnight_<TS>/ and outputs to v9/outputs/overnight_<TS>/.

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="v9/logs/overnight_${TS}"
OUT_DIR="v9/outputs/overnight_${TS}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

GLOBAL_LOG="$LOG_DIR/00_overall.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$GLOBAL_LOG"; }

run_step() {
    local name="$1"; shift
    local step_log="$LOG_DIR/${name}.log"
    log "═══ START $name ═══"
    log "command: $*"
    if "$@" 2>&1 | tee -a "$step_log"; then
        log "═══ DONE  $name ═══"
        return 0
    else
        local rc=$?
        log "═══ FAIL  $name (exit $rc) ═══"
        return $rc
    fi
}

# ─── Step 0a: retrain tokenizer with extended (F+2) blocks for boundary continuity ───
run_step "00a_tokenizer_retrain" \
    python -u v9/training/train_phoneme_rvq.py \
        --device mps --epochs 5 --batch-size 8 \
        --warmup-steps 500 --preload --log-every 200 \
        --f-pad 32 --b-pad 256 \
        --checkpoint-dir v9/checkpoints/phoneme_rvq

# ─── Step 0b: re-extract RVQ tokens with new tokenizer (overwrite old codes) ───
rm -rf v9/data/phoneme_tokens
run_step "00b_extract_rvq_tokens" \
    python -u v9/scripts/extract_v9_tokens.py \
        --device mps \
        --tokenizer-checkpoint v9/checkpoints/phoneme_rvq/best.pt \
        --out-dir v9/data/phoneme_tokens

# ─── Step 1: Stage 1 (joint renderer + style encoder + codebook) ───────
# Defaults bumped: codebook=64 (was 512, prevents collapse) + knob_dropout=0.3 (stronger CFG)
run_step "01_stage1_renderer" \
    python -u v9/training/train_v9_renderer.py \
        --device mps --epochs 30 --batch-size 16 \
        --warmup-steps 2000 --preload --log-every 200 \
        --knob-source emotion \
        --checkpoint-dir v9/checkpoints/stage1_renderer

# ─── Step 2: extract style codes from trained encoder+codebook ─────────
rm -rf v9/data/style_codes
run_step "02_extract_style_codes" \
    python -u v9/scripts/extract_v9_style_codes.py \
        --device mps \
        --stage1-checkpoint v9/checkpoints/stage1_renderer/best.pt \
        --out-dir v9/data/style_codes

# ─── Step 3: Stage 2 (style planner) ───────────────────────────────────
run_step "03_stage2_planner" \
    python -u v9/training/train_v9_style_planner.py \
        --device mps --epochs 20 --batch-size 16 \
        --warmup-steps 1000 --preload --log-every 200 \
        --checkpoint-dir v9/checkpoints/stage2_planner

# ─── Step 4: end-to-end inference at multiple settings ─────────────────
TEXT='You know, I tried that the other day, and it actually made me feel a lot better.'
TEXT2='What an amazing surprise! I cannot believe you came all this way.'
TEXT3='I dont want to talk about it right now.'

TOK="v9/checkpoints/phoneme_rvq/best.pt"
S1="v9/checkpoints/stage1_renderer/best.pt"
S2="v9/checkpoints/stage2_planner/best.pt"

render() {
    local label="$1" text="$2" emo="$3" intensity="$4" temp="$5" cfg="$6" pcfg="$7" spk_arg="$8"
    local out="${OUT_DIR}/${label}.wav"
    log "render: $label  emo=$emo intensity=$intensity temp=$temp r-cfg=$cfg p-cfg=$pcfg"
    python -u v9/inference/synthesize_v9_2stage.py "$text" \
        --tokenizer-checkpoint "$TOK" \
        --stage1-checkpoint "$S1" --stage2-checkpoint "$S2" \
        --emotion "$emo" --intensity "$intensity" \
        --temperature "$temp" --cfg-scale "$cfg" --planner-cfg-scale "$pcfg" \
        --device cpu --output "$out" $spk_arg 2>&1 | tee -a "$LOG_DIR/04_render.log" | tail -3
}

# Test 1 — emotion sweep
render "t1_neutral_05"   "$TEXT" neutral  0.5 1.0 1.0 1.0 ""
render "t1_happy_08"     "$TEXT" happy    0.8 1.0 1.0 1.0 ""
render "t1_sad_07"       "$TEXT" sad      0.7 1.0 1.0 1.0 ""
render "t1_angry_09"     "$TEXT" angry    0.9 1.0 1.0 1.0 ""
render "t1_surprise_08"  "$TEXT" surprise 0.8 1.0 1.0 1.0 ""

# Test 2 — CFG ablation (planner CFG; happy)
render "t2_happy_pcfg10" "$TEXT" happy 0.9 1.0 1.0 1.0 ""
render "t2_happy_pcfg20" "$TEXT" happy 0.9 1.0 1.0 2.0 ""
render "t2_happy_pcfg30" "$TEXT" happy 0.9 1.0 1.0 3.0 ""

# Test 3 — different texts
render "t3_text2_happy"  "$TEXT2" happy 0.9 1.0 1.5 1.5 ""
render "t3_text3_sad"    "$TEXT3" sad   0.8 1.0 1.5 1.5 ""

# RDJ if available
RDJ_SPK="reference_audio/rdj/rdj_spk_emb.npy"
if [[ -f "$RDJ_SPK" ]]; then
    render "t4_rdj_neutral" "$TEXT" neutral  0.5 1.0 1.0 1.0 "--speaker-emb $RDJ_SPK"
    render "t4_rdj_happy"   "$TEXT" happy    0.8 1.0 1.5 1.5 "--speaker-emb $RDJ_SPK"
    render "t4_rdj_sad"     "$TEXT" sad      0.7 1.0 1.5 1.5 "--speaker-emb $RDJ_SPK"
fi

log "all done"
ls -la "$OUT_DIR" | tee -a "$GLOBAL_LOG"
