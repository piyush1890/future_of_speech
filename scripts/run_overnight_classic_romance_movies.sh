#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/Users/piyush/projects/articulatory-tts"
OUT_BASE="$PROJECT_ROOT/data/licensed_hindi_classic_romance_batch_full_clean_guarded"
LOG_DIR="$PROJECT_ROOT/data/batch_logs/classic_romance_overnight"
PROGRESS_TSV="$OUT_BASE/batch_progress.tsv"

mkdir -p "$OUT_BASE" "$LOG_DIR"

if command -v caffeinate >/dev/null 2>&1 && [[ "${DIALOGUE_BATCH_CAFFEINATED:-0}" != "1" ]]; then
  export DIALOGUE_BATCH_CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

export PYTORCH_ENABLE_MPS_FALLBACK=0
export DIALOGUE_DATASET_BOOTSTRAPPED="${DIALOGUE_DATASET_BOOTSTRAPPED:-0}"

if [[ -x "$PROJECT_ROOT/.venv_dialogue_dataset/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv_dialogue_dataset/bin/python"
else
  PYTHON="$(command -v python3)"
fi

MOVIES=(
  "chashme_buddoor_1981|Chashme Buddoor (1981)|/Users/piyush/Downloads/vidssave.com Chashme Buddoor - चश्मे बद्दूर  - Full Movie - Farooq Sheikh, Deepti Naval, Rakesh Bedi - HD 144P.mp4"
  "gharonda_1977|Gharonda (1977)|/Users/piyush/Downloads/vidssave.com Gharaonda (HD) (1977) Full Hindi Movie _ Amol Palekar, Zarina Wahab, Dr. Shreeram Lagoo 144P.mp4"
  "saath_saath_1982|Saath Saath (1982)|/Users/piyush/Downloads/vidssave.com Saath Saath {HD} Farooque Shaikh _ Deepti Naval _ Satish Shah Hindi Full Movie (With Eng Subtitles) 360P.mp4"
  "ahista_ahista_1981|Ahista Ahista (1981)|/Users/piyush/Downloads/vidssave.com Ahista Ahista (1981) Full Hindi Movie _ Shammi Kapoor, Nanda, Kunal Kapoor. Padmini Kolhapure 144P.mp4"
  "sparsh_1980|Sparsh (1980)|/Users/piyush/Downloads/vidssave.com Sparsh (HD & Eng Subs) Hindi Full Movie - Naseeruddin Shah - Shabana Azmi - Bollywood Classic Movies 144P.mp4"
  "manzil_1979|Manzil (1979)|/Users/piyush/Downloads/vidssave.com Manzil 1979 Full Movie - Amitabh Bachchan, Moushumi Chatterjee _ RD Burman Musical _ Old Hindi Films 144P.mp4"
  "ijaazat_1987|Ijaazat (1987)|/Users/piyush/Downloads/vidssave.com Ijaazat (1987) Full Hindi Movie _ Naseeruddin Shah, Rekha, Anuradha Patel 144P.mp4"
  "ankahi_1985|Ankahi (1985)|/Users/piyush/Downloads/Ankahee- 1985.mp4"
  "gaman_1978|Gaman (1978)|/Users/piyush/Downloads/vidssave.com Gaman - 1978 FULL MOVIE  - Farooq Shaikh, Smita Patil (Super Hit Classic GAMAN Bollywood Movie) 144P.mp4"
  "ek_baar_phir_1980|Ek Baar Phir (1980)|/Users/piyush/Downloads/Ek Baar Phir _ Full Movie _ HD.mp4"
  "mausam_1975|Mausam (1975)|/Users/piyush/Downloads/vidssave.com Mausam Hindi Full HD Movie _ Sanjeev Kumar _ Sharmila Tagore _ 1975 Bollywood Full Movie 144P.mp4"
)

if [[ ! -f "$PROGRESS_TSV" ]]; then
  printf "started_at\tslug\ttitle\tstatus\tpaired_seconds\ttrain_clips\tout_root\tlog\n" > "$PROGRESS_TSV"
fi

run_movie() {
  local slug="$1"
  local title="$2"
  local source="$3"
  local out_root="$OUT_BASE/$slug"
  local log="$LOG_DIR/${slug}.log"
  local started_at
  started_at="$(date -Iseconds)"

  if [[ ! -f "$source" ]]; then
    printf "%s\t%s\t%s\tmissing_source\t0\t0\t%s\t%s\n" "$started_at" "$slug" "$title" "$out_root" "$log" >> "$PROGRESS_TSV"
    return 0
  fi

  if [[ "${DIALOGUE_BATCH_DRY_RUN:-0}" == "1" ]]; then
    printf "%s\t%s\t%s\tdry_run\t0\t0\t%s\t%s\n" "$started_at" "$slug" "$title" "$out_root" "$log" >> "$PROGRESS_TSV"
    return 0
  fi

  if [[ -s "$out_root/batch_summary.json" ]] && "$PYTHON" - "$out_root/batch_summary.json" <<'PY' >/dev/null 2>&1
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if data.get("failures") == 0 and data.get("paired_seconds", 0) > 0 else 1)
PY
  then
    local paired_seconds train_clips
    paired_seconds="$("$PYTHON" - "$out_root/batch_summary.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data.get("paired_seconds", 0))
PY
)"
    train_clips="$("$PYTHON" - "$out_root/batch_summary.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(sum(item.get("train_clips", 0) for item in data.get("summaries", [])))
PY
)"
    printf "%s\t%s\t%s\tskipped_done\t%s\t%s\t%s\t%s\n" "$started_at" "$slug" "$title" "$paired_seconds" "$train_clips" "$out_root" "$log" >> "$PROGRESS_TSV"
    return 0
  fi

  printf "\n[%s] START %s\nsource=%s\nout=%s\nlog=%s\n" "$started_at" "$title" "$source" "$out_root" "$log" >> "$LOG_DIR/batch.log"

  set +e
  "$PYTHON" "$PROJECT_ROOT/run_dialogue_dataset.py" "$source" \
    --out-root "$out_root" \
    --slug "$slug" \
    --title "$title" \
    > "$log" 2>&1
  local status="$?"
  set -e

  local final_status="failed_${status}"
  local paired_seconds="0"
  local train_clips="0"
  if [[ "$status" == "0" && -s "$out_root/batch_summary.json" ]]; then
    final_status="$("$PYTHON" - "$out_root/batch_summary.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print("done" if data.get("failures") == 0 else "done_with_failures")
PY
)"
    paired_seconds="$("$PYTHON" - "$out_root/batch_summary.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data.get("paired_seconds", 0))
PY
)"
    train_clips="$("$PYTHON" - "$out_root/batch_summary.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(sum(item.get("train_clips", 0) for item in data.get("summaries", [])))
PY
)"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$started_at" "$slug" "$title" "$final_status" "$paired_seconds" "$train_clips" "$out_root" "$log" >> "$PROGRESS_TSV"
  printf "[%s] END %s status=%s paired_seconds=%s train_clips=%s\n" "$(date -Iseconds)" "$title" "$final_status" "$paired_seconds" "$train_clips" >> "$LOG_DIR/batch.log"
}

cd "$PROJECT_ROOT"

printf "Batch start: %s\n" "$(date -Iseconds)" >> "$LOG_DIR/batch.log"
printf "Output base: %s\n" "$OUT_BASE" >> "$LOG_DIR/batch.log"
printf "Logs: %s\n" "$LOG_DIR" >> "$LOG_DIR/batch.log"
printf "GPU mode: MPS only, PYTORCH_ENABLE_MPS_FALLBACK=%s\n" "$PYTORCH_ENABLE_MPS_FALLBACK" >> "$LOG_DIR/batch.log"

for movie in "${MOVIES[@]}"; do
  IFS="|" read -r slug title source <<< "$movie"
  run_movie "$slug" "$title" "$source"
done

printf "Batch complete: %s\n" "$(date -Iseconds)" >> "$LOG_DIR/batch.log"
