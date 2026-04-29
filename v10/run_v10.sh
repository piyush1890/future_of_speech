#!/bin/zsh
# v10 end-to-end orchestrator.
#   1. Train frame-level RVQ tokenizer.
#   2. Extract per-utt (T, K) frame codes.
#   3. Train Stage 1 (renderer + style encoder jointly).
#   4. Extract per-utt (N+2,) style codes.
#   5. Train Stage 2 (style planner).
#   6. Render audio samples.

set -e
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda activate arttts
cd ~/projects/articulatory-tts

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:${PWD}"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="v10/logs/run_${TS}"
OUT_DIR="v10/outputs/run_${TS}"
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

# ─── Step 1: tokenizer ───────────────────────────────────────────────────
run_step "01_tokenizer" \
    python -u v10/training/train_v10_tokenizer.py \
        --device mps --epochs 5 --batch-size 8 \
        --d-model 256 --enc-layers 4 --dec-layers 4 \
        --codebook-size 1024 --num-quantizers 4 \
        --warmup-steps 1000 --log-every 200 --preload \
        --checkpoint-dir v10/checkpoints/tokenizer

# ─── Step 2: extract frame codes ─────────────────────────────────────────
run_step "02_extract_frame_codes" \
    python -u v10/scripts/extract_v10_frame_codes.py \
        --device mps --batch-size 16 \
        --checkpoint v10/checkpoints/tokenizer/best.pt \
        --out-dir v10/data/frame_codes

# ─── Step 3: stage 1 (renderer + style joint) ────────────────────────────
run_step "03_stage1_renderer" \
    python -u v10/training/train_v10_renderer.py \
        --device mps --epochs 25 --batch-size 16 \
        --warmup-steps 2000 --log-every 200 --preload \
        --knob-source emotion \
        --frame-codes-dir v10/data/frame_codes \
        --checkpoint-dir v10/checkpoints/stage1_renderer

# ─── Step 4: extract style codes ─────────────────────────────────────────
run_step "04_extract_style_codes" \
    python -u v10/scripts/extract_v10_style_codes.py \
        --device mps --batch-size 16 \
        --checkpoint v10/checkpoints/stage1_renderer/best.pt \
        --out-dir v10/data/style_codes

# ─── Step 5: stage 2 (planner) ────────────────────────────────────────────
run_step "05_stage2_planner" \
    python -u v10/training/train_v10_planner.py \
        --device mps --epochs 20 --batch-size 16 \
        --warmup-steps 1000 --log-every 200 --preload \
        --checkpoint-dir v10/checkpoints/stage2_planner

# ─── Step 6: render audio samples ────────────────────────────────────────
TEXT='You know, I tried that the other day, and it actually made me feel a lot better.'
TEXT2='What an amazing surprise! I cannot believe you came all this way.'
TEXT3='I dont want to talk about it right now.'

TOK="v10/checkpoints/tokenizer/best.pt"
S1="v10/checkpoints/stage1_renderer/best.pt"
S2="v10/checkpoints/stage2_planner/best.pt"

render() {
    local label="$1" text="$2" emo="$3" intensity="$4" temp="$5" cfg="$6" pcfg="$7" spk_arg="$8"
    local out="${OUT_DIR}/${label}.wav"
    log "render: $label  emo=$emo intensity=$intensity temp=$temp r-cfg=$cfg p-cfg=$pcfg"
    python -u v10/inference/synthesize_v10.py "$text" \
        --tokenizer-checkpoint "$TOK" \
        --stage1-checkpoint "$S1" --stage2-checkpoint "$S2" \
        --emotion "$emo" --intensity "$intensity" \
        --temperature "$temp" --cfg-scale "$cfg" --planner-cfg-scale "$pcfg" \
        --device mps --output "$out" $spk_arg 2>&1 | tee -a "$LOG_DIR/06_render.log" | tail -3
}

render "t1_neutral_05"  "$TEXT" neutral  0.5 1.0 1.0 1.0 ""
render "t1_happy_08"    "$TEXT" happy    0.8 1.0 1.0 1.0 ""
render "t1_sad_07"      "$TEXT" sad      0.7 1.0 1.0 1.0 ""
render "t1_angry_09"    "$TEXT" angry    0.9 1.0 1.0 1.0 ""
render "t1_surprise_08" "$TEXT" surprise 0.8 1.0 1.0 1.0 ""

render "t2_happy_pcfg10" "$TEXT" happy 0.9 1.0 1.0 1.0 ""
render "t2_happy_pcfg20" "$TEXT" happy 0.9 1.0 1.0 2.0 ""
render "t2_happy_pcfg30" "$TEXT" happy 0.9 1.0 1.0 3.0 ""

render "t3_text2_happy" "$TEXT2" happy 0.9 1.0 1.5 1.5 ""
render "t3_text3_sad"   "$TEXT3" sad   0.8 1.0 1.5 1.5 ""

RDJ_SPK="reference_audio/rdj/rdj_spk_emb.npy"
if [[ -f "$RDJ_SPK" ]]; then
    render "t4_rdj_neutral" "$TEXT" neutral 0.5 1.0 1.0 1.0 "--speaker-emb $RDJ_SPK"
    render "t4_rdj_happy"   "$TEXT" happy   0.8 1.0 1.5 1.5 "--speaker-emb $RDJ_SPK"
    render "t4_rdj_sad"     "$TEXT" sad     0.7 1.0 1.5 1.5 "--speaker-emb $RDJ_SPK"
fi

log "all done"
ls -la "$OUT_DIR" | tee -a "$GLOBAL_LOG"
