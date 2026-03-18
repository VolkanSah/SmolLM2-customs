#!/usr/bin/env bash
# smollm2_train.sh – Standalone Training Trigger (Git Actions Fallback or local use)
#
# Usage:
#   export SMOLLM_API_KEY="your_key"
#   ./smollm2_train.sh [mode]          # mode: all | export | validate | finetune | export_validate
#   ./smollm2_train.sh --dry-run all   # Nur Befehle ausgeben, nicht ausführen
#
#  example use
#   SMOLLM_API_KEY=sk-xxx ./smollm2_train.sh all

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
# ── SET YOUR SPACE CONFIG!
API_BASE="https://codey-lab-smollm2-customs.hf.space/v1/train/execute"
TIMEOUT=300
RETRY_MAX=3
RETRY_DELAY=15

# ── Farben (nur wenn Terminal) ────────────────────────────────────────────────
if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
  BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BLUE=''; BOLD=''; RESET=''
fi

# ── Argument Parsing ──────────────────────────────────────────────────────────
DRY_RUN=false
MODE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run|-n) DRY_RUN=true; shift ;;
    all|export|validate|finetune|export_validate) MODE=$1; shift ;;
    --help|-h)
      echo "Usage: $0 [--dry-run] <mode>"
      echo "Modes: all | export | validate | finetune | export_validate"
      exit 0 ;;
    *) echo -e "${RED}Unbekanntes Argument: $1${RESET}"; exit 1 ;;
  esac
done

if [ -z "$MODE" ]; then
  echo -e "${YELLOW}Kein Mode angegeben, verwende 'all'${RESET}"
  MODE="all"
fi

# ── Secret Check ──────────────────────────────────────────────────────────────
if [ -z "${SMOLLM_API_KEY:-}" ]; then
  echo -e "${RED}✗ SMOLLM_API_KEY ist nicht gesetzt!${RESET}"
  echo "  export SMOLLM_API_KEY='dein-key'"
  exit 1
fi

# ── Kern-Funktion ─────────────────────────────────────────────────────────────
call_api() {
  local mode=$1
  local url="${API_BASE}?mode=${mode}"
  local attempt=0
  local delay=$RETRY_DELAY

  echo -e "\n${BLUE}${BOLD}▶ mode=${mode}${RESET}"
  echo -e "  URL: ${url}"

  if [ "$DRY_RUN" = true ]; then
    echo -e "  ${YELLOW}[DRY RUN]${RESET} curl -X POST \"${url}\" \\"
    echo -e "    -H \"Authorization: Bearer ***\" \\"
    echo -e "    -H \"Content-Type: application/json\" \\"
    echo -e "    --max-time ${TIMEOUT}"
    return 0
  fi

  local tmp_body; tmp_body=$(mktemp)
  local tmp_err;  tmp_err=$(mktemp)
  trap "rm -f $tmp_body $tmp_err" RETURN

  while [ $attempt -lt $RETRY_MAX ]; do
    attempt=$((attempt + 1))
    echo -e "  Versuch ${attempt}/${RETRY_MAX}..."

    local http_code
    http_code=$(curl -s \
      -o "$tmp_body" \
      -w "%{http_code}" \
      -X POST "$url" \
      -H "Authorization: Bearer ${SMOLLM_API_KEY}" \
      -H "Content-Type: application/json" \
      --max-time "$TIMEOUT" \
      --connect-timeout 30 \
      2>"$tmp_err") || {
        local curl_exit=$?
        echo -e "  ${YELLOW}⚠ curl Fehler (exit ${curl_exit}): $(cat "$tmp_err")${RESET}"
        if [ $attempt -lt $RETRY_MAX ]; then
          echo -e "  Retry in ${delay}s..."
          sleep "$delay"
          delay=$((delay * 2))
          continue
        fi
        echo -e "${RED}✗ mode=${mode} fehlgeschlagen (curl error)${RESET}"
        return 1
      }

    local body; body=$(cat "$tmp_body" | head -c 500)
    echo -e "  HTTP: ${http_code}"
    [ -n "$body" ] && echo -e "  Body: ${body}"

    if [[ "$http_code" =~ ^2 ]]; then
      echo -e "  ${GREEN}✓ Erfolg (HTTP ${http_code})${RESET}"
      return 0
    elif [[ "$http_code" =~ ^4 ]]; then
      echo -e "${RED}✗ Client-Fehler ${http_code} – kein Retry${RESET}"
      return 1
    else
      echo -e "  ${YELLOW}⚠ Server-Fehler ${http_code}, Retry in ${delay}s...${RESET}"
      sleep "$delay"
      delay=$((delay * 2))
    fi
  done

  echo -e "${RED}✗ mode=${mode} nach ${RETRY_MAX} Versuchen aufgegeben${RESET}"
  return 1
}

# ── Pipeline ──────────────────────────────────────────────────────────────────
echo -e "${BOLD}SmolLM2 Training Pipeline${RESET}"
echo -e "Mode: ${BOLD}${MODE}${RESET} | Dry-run: ${DRY_RUN}"
echo "──────────────────────────────────────────"

FAILED=()

run_step() {
  local step=$1
  call_api "$step" || FAILED+=("$step")
}

case $MODE in
  all)            run_step export; run_step validate; run_step finetune ;;
  export)         run_step export ;;
  validate)       run_step validate ;;
  finetune)       run_step finetune ;;
  export_validate) run_step export; run_step validate ;;
esac

# ── Ergebnis ──────────────────────────────────────────────────────────────────
echo -e "\n──────────────────────────────────────────"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo -e "${GREEN}${BOLD}✓ Pipeline abgeschlossen${RESET}"
else
  echo -e "${RED}${BOLD}✗ Fehlgeschlagene Steps: ${FAILED[*]}${RESET}"
  exit 1
fi
