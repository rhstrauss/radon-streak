#!/bin/bash
# Survey batch worker. Manifest lines: "NIGHTDIR YYYYMMDD FIELD".
# Per field: download 4 arch.fz -> run_g96_sequence -> keep CSV/JSON, DELETE frames.
# Resumable (skips fields whose *_trails.csv exists). Good-citizen thread caps.
set -u
PY=/astro/store/shire/rstrau/miniforge3/envs/mpchecker/bin/python
SR=/astro/store/shire/rstrau/streak_radon
BASE=https://sbnarchive.psi.edu/pds4/surveys/gbo.ast.catalina.survey/data_calibrated/G96/2024
MANIFEST=$1; NPAR=${2:-20}; OUTROOT=${3:-/astro/store/shire/rstrau/css_pds_pilot/survey/out}
mkdir -p "$OUTROOT"
export PYTHONPATH=$SR
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TMPDIR=/local/tmp/claude-1615156/survey_tmp; mkdir -p "$TMPDIR"

process() {
  NIGHT=$1; YMD=$2; F=$3
  D="$OUTROOT/$NIGHT/$F"
  [ -s "$D/${F}_trails.csv" ] && { echo "[$NIGHT/$F] done"; return 0; }
  mkdir -p "$D"
  for n in 0001 0002 0003 0004; do
    f="G96_${YMD}_2B_${F}_01_${n}.arch.fz"
    [ -s "$D/$f" ] || curl -sS --max-time 600 -o "$D/$f" "${BASE}/${NIGHT}/$f"
  done
  $PY "$SR/bin/run_g96_sequence.py" \
      --frames "$D/G96_${YMD}_2B_${F}_01_000?.arch.fz" \
      --out "$D/${F}_trails.csv" --json "$D/${F}_fits.json" \
      > "$D/${F}.log" 2>&1
  rc=$?
  nd=$(tail -n +2 "$D/${F}_trails.csv" 2>/dev/null | wc -l)
  rm -f "$D"/*.arch.fz          # disk hygiene: keep CSV/JSON/log only
  echo "[$NIGHT/$F] rc=$rc dets=$nd"
}
export -f process; export PY SR BASE OUTROOT

echo "survey batch start $(date); manifest=$MANIFEST parallel=$NPAR out=$OUTROOT"
echo "fields: $(wc -l < "$MANIFEST")"
cat "$MANIFEST" | xargs -P "$NPAR" -L1 bash -c 'process "$@"' _
echo "survey batch DONE $(date)"
