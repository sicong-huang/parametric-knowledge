#!/usr/bin/env bash
# inspect_corpus.sh -- human-readable viewer for premise2/corpus/*.jsonl
#
# One trace's data is scattered across 6 files, joined on trace_id
# (+ hop_index, condition, sample_id). This stitches them back together.
#
# Usage: premise2/scripts/inspect_corpus.sh <mode> [args] [options]
#
# modes:
#   ls [--dataset D] [--model M] [--flipped]    list trace_ids w/ question + em_correct
#   trace <trace_id>                            manifest row expanded (question/gold/answer/reasoning)
#   target <trace_id>                           candidate row: hop_facts vs gold_fact_text
#   prefix <trace_id> [hop]                     original vs corrupted prefix_text side by side
#   conts <trace_id> [hop]                      sampled continuations grouped by condition
#   follow <trace_id>                           trace + target + prefix + conts, one buffer
#   answers <trace_id> [hop]                    final_answer distribution, original vs corrupted
#
# options:
#   --placebo     read tier2_placebo_* instead of tier2_* (hop defaults to -1)
#   -n N          cap continuations shown per condition
#   -w N          wrap width (default $COLUMNS or 100)
#   --no-pager
set -euo pipefail

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

[ $# -ge 1 ] || usage 1
MODE="$1"; shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CORPUS="$REPO_ROOT/premise2/corpus"
MANIFEST="$CORPUS/traces_manifest.jsonl"
CANDIDATES="$CORPUS/tier2_candidates.jsonl"

TRACE_ID=""
HOP=""
PLACEBO=0
N=""
W="${COLUMNS:-100}"
PAGER=1
LS_DATASET=""
LS_MODEL=""
LS_FLIPPED=0

# positional args differ per mode; parse generically: first non-option arg
# after mode is trace_id (except ls, which takes none), second is hop.
POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --placebo) PLACEBO=1; shift ;;
    -n) N="$2"; shift 2 ;;
    -w) W="$2"; shift 2 ;;
    --no-pager) PAGER=0; shift ;;
    --dataset) LS_DATASET="$2"; shift 2 ;;
    --model) LS_MODEL="$2"; shift 2 ;;
    --flipped) LS_FLIPPED=1; shift ;;
    -h|--help) usage 0 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [ "$MODE" != "ls" ]; then
  [ ${#POSITIONAL[@]} -ge 1 ] || { echo "mode '$MODE' needs a trace_id" >&2; exit 1; }
  TRACE_ID="${POSITIONAL[0]}"
  [ ${#POSITIONAL[@]} -ge 2 ] && HOP="${POSITIONAL[1]}"
fi

if [ "$PLACEBO" = 1 ]; then
  PREFIXES="$CORPUS/tier2_placebo_prefixes.jsonl"
  CONTINUATIONS="$CORPUS/tier2_placebo_continuations.jsonl"
  [ -z "$HOP" ] && HOP=-1
else
  PREFIXES="$CORPUS/tier2_prefixes.jsonl"
  CONTINUATIONS="$CORPUS/tier2_continuations.jsonl"
fi

for f in "$MANIFEST" "$CANDIDATES" "$PREFIXES" "$CONTINUATIONS"; do
  [ -f "$f" ] || { echo "missing corpus file: $f" >&2; exit 1; }
done

if ! [ -t 1 ]; then PAGER=0; fi

# grep-first-then-jq: avoids parsing 14.5k lines to find one trace_id
grep_trace() {
  local file="$1" id="$2"
  grep -F "\"trace_id\": \"$id\"" "$file" || true
}

require_trace_found() {
  local lines="$1" label="$2"
  if [ -z "$lines" ]; then
    echo "trace_id not found in $label: $TRACE_ID" >&2
    exit 1
  fi
}

emit() { printf '%s\n' "$1" | fold -s -w "$W"; }

render_trace() {
  local rec; rec="$(grep_trace "$MANIFEST" "$TRACE_ID")"
  require_trace_found "$rec" "traces_manifest.jsonl"
  printf '%s' "$rec" | jq -r '
    def as_text: if type == "array" then join(" | ") else tostring end;
    "=== \(.trace_id)  [\(.exp_name)/\(.dataset)]  em_correct=\(.em_correct)\n" +
    "Q     " + .question + "\n" +
    "gold  " + (.gold_answer | as_text) + "\n" +
    "final " + (.final_answer | as_text) + "\n" +
    "--- reasoning ---\n" + .reasoning
  ' | fold -s -w "$W"
}

render_target() {
  local rec; rec="$(grep_trace "$CANDIDATES" "$TRACE_ID")"
  require_trace_found "$rec" "tier2_candidates.jsonl"
  printf '%s' "$rec" | jq -r '
    "=== \(.trace_id)  any_hop_matched=\(.any_hop_matched)" as $header |
    [$header] + [.hop_facts[] |
      "-- hop \(.hop_index)  match_score=\(.match_score) --\n" +
      "sentence_span : " + .sentence_span + "\n" +
      "gold_fact     : " + .gold_fact_text
    ] | join("\n")
  ' | fold -s -w "$W"
}

render_prefix() {
  local recs; recs="$(grep_trace "$PREFIXES" "$TRACE_ID")"
  require_trace_found "$recs" "$(basename "$PREFIXES")"
  local hop_filter="."
  [ -n "$HOP" ] && hop_filter="select(.hop_index == ($HOP | tonumber))"

  local filtered
  filtered="$(printf '%s\n' "$recs" | jq -c "$hop_filter" 2>/dev/null || true)"
  [ -n "$filtered" ] || { echo "no prefix rows for hop=$HOP on $TRACE_ID" >&2; return 1; }

  local hops
  hops="$(printf '%s\n' "$filtered" | jq -r '.hop_index' | sort -un)"

  for h in $hops; do
    echo "--- hop $h --------------------------------------------------"
    local orig corr
    orig="$(printf '%s\n' "$filtered" | jq -r --argjson h "$h" 'select(.hop_index==$h and .condition=="original") | .sentence')"
    corr="$(printf '%s\n' "$filtered" | jq -r --argjson h "$h" 'select(.hop_index==$h and .condition=="corrupted") | .sentence')"
    echo "original  : $orig"
    echo "corrupted : $corr"

    if command -v git >/dev/null 2>&1; then
      local orig_full corr_full t1 t2
      orig_full="$(printf '%s\n' "$filtered" | jq -r --argjson h "$h" 'select(.hop_index==$h and .condition=="original") | .prefix_text')"
      corr_full="$(printf '%s\n' "$filtered" | jq -r --argjson h "$h" 'select(.hop_index==$h and .condition=="corrupted") | .prefix_text')"
      if [ -n "$orig_full" ] && [ -n "$corr_full" ]; then
        t1=$(mktemp); t2=$(mktemp)
        trap 'rm -f "$t1" "$t2"' RETURN
        printf '%s\n' "$orig_full" > "$t1"
        printf '%s\n' "$corr_full" > "$t2"
        echo "--- word diff (original -> corrupted) ---"
        git --no-pager diff --no-index --word-diff=color -- "$t1" "$t2" 2>/dev/null | tail -n +6 || true
        rm -f "$t1" "$t2"
        trap - RETURN
      fi
    fi
    echo
  done
}

render_answers() {
  local file="$CONTINUATIONS" recs
  recs="$(grep_trace "$file" "$TRACE_ID")"
  require_trace_found "$recs" "$(basename "$file")"
  local hop_filter="."
  [ -n "$HOP" ] && hop_filter="select(.hop_index == ($HOP | tonumber))"
  echo "=== $TRACE_ID  final_answer distribution ==="
  # truncate to keep the tally scannable -- extraction sometimes fails and
  # final_answer falls back to the full continuation text; `conts` mode
  # shows the untruncated text
  printf '%s\n' "$recs" | jq -r "$hop_filter" 2>/dev/null | \
    jq -r '[.condition, (.final_answer | if length > 80 then .[0:77] + "..." else . end)] | @tsv' | \
    sort | uniq -c | sort -rn
}

render_conts() {
  local file="$CONTINUATIONS" recs
  recs="$(grep_trace "$file" "$TRACE_ID")"
  require_trace_found "$recs" "$(basename "$file")"
  local hop_filter="."
  [ -n "$HOP" ] && hop_filter="select(.hop_index == ($HOP | tonumber))"

  render_answers
  echo

  for cond in original corrupted; do
    echo "=== condition: $cond ==="
    local sub
    sub="$(printf '%s\n' "$recs" | jq -c "$hop_filter" 2>/dev/null | jq -c --arg c "$cond" 'select(.condition==$c)' 2>/dev/null || true)"
    [ -z "$sub" ] && { echo "(none)"; continue; }
    if [ -n "$N" ]; then
      sub="$(printf '%s\n' "$sub" | head -n "$N")"
    fi
    printf '%s\n' "$sub" | jq -r '"[sample \(.sample_id)] -> \(.final_answer)\n" + .continuation_text + "\n"' | fold -s -w "$W"
  done
}

render_ls() {
  local jq_filter="."
  [ -n "$LS_DATASET" ] && jq_filter="$jq_filter | select(.dataset == \"$LS_DATASET\")"
  [ -n "$LS_MODEL" ] && jq_filter="$jq_filter | select(.model == \"$LS_MODEL\")"

  if [ "$LS_FLIPPED" = 1 ]; then
    # compute inline from continuations: flipped = majority final_answer
    # differs between original and corrupted conditions for that trace_id
    echo "computing flips from $(basename "$CONTINUATIONS") (no score_flips.py output found)..." >&2
    jq -r '[.trace_id, .condition, .final_answer] | @tsv' "$CONTINUATIONS" | \
      awk -F'\t' '{key=$1"\t"$2; cnt[key"\t"$3]++; seen[$1]=1}
        END{
          for (k in cnt) {
            split(k, a, "\t"); id=a[1]; cond=a[2]; ans=a[3]; c=cnt[k];
            if (c > best[id"\t"cond]) { best[id"\t"cond]=c; bestans[id"\t"cond]=ans }
          }
          for (id in seen) {
            o=bestans[id"\t""original"]; c=bestans[id"\t""corrupted"];
            if (o != "" && c != "" && o != c) print id
          }
        }' | sort > /tmp/inspect_corpus_flipped_ids.$$
    FLIP_FILTER="/tmp/inspect_corpus_flipped_ids.$$"
    trap 'rm -f "$FLIP_FILTER"' EXIT
  fi

  jq -c "$jq_filter" "$MANIFEST" | while IFS= read -r rec; do
    id=$(printf '%s' "$rec" | jq -r '.trace_id')
    if [ "$LS_FLIPPED" = 1 ] && ! grep -qxF "$id" "$FLIP_FILTER" 2>/dev/null; then
      continue
    fi
    printf '%s' "$rec" | jq -r '"\(.trace_id)  em=\(.em_correct)  " + .question'
  done
}

case "$MODE" in
  ls) OUT="$(render_ls)" ;;
  trace) OUT="$(render_trace)" ;;
  target) OUT="$(render_target)" ;;
  prefix) OUT="$(render_prefix)" ;;
  answers) OUT="$(render_answers)" ;;
  conts) OUT="$(render_conts)" ;;
  follow)
    OUT="$(render_trace; echo; render_target; echo; render_prefix; echo; render_conts)"
    ;;
  *) echo "unknown mode: $MODE" >&2; usage 1 ;;
esac

if [ "$PAGER" = 1 ]; then
  printf '%s\n' "$OUT" | less -R
else
  printf '%s\n' "$OUT"
fi
