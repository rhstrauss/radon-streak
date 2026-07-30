#!/bin/bash
# process_ztf_unit.sh <night> <field> <ccdid> <qid> <ffd_csv> <filt_csv> <outdir>
#
# Download one ZTF quadrant-visit sequence from IRSA/IBE and run the full
# streak_radon detection chain -> per-unit hldet trail CSV + image log.
# Idempotent: exits immediately if the trail CSV already exists, so a preempted
# and requeued array task re-runs only the units it had not finished.
#
# Unlike the CSS path there is NO bulk pre-stage. ZTF Sep-Dec 2024 is ~91 TB of
# diffs and the /gscratch/astro fileset has ~2.5 TB free, so downloading is
# necessarily per-unit and streaming: fetch, detect, delete. That also makes the
# array itself the download parallelism (N concurrent tasks = N IRSA streams),
# which is what keeps a download-bound campaign from serialising behind one
# prestage job.
#
# ONLY the diff and the mask are fetched. sciimg (37.9 MB, the largest product)
# is NOT needed: SHUTOPEN/SHUTCLSD are present in the diff header, so the
# validated shutter-midpoint epoch costs no extra bytes. 28.3 MB/exposure
# instead of 66.2 MB.
set -uo pipefail
NIGHT=$1; FIELD=$2; CCDID=$3; QID=$4; FFDS=$5; FILTS=$6; OUTDIR=$7

IBE=https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci
PY=/mmfs1/home/rstrau/miniconda3/envs/mpchecker/bin/python
SR=${SR:-/gscratch/astro/rstrau/streak_radon_ztf}
CFG=${CFG:-ztf}

UNIT=$(printf '%s_f%06d_c%02d_q%d' "$NIGHT" "$FIELD" "$CCDID" "$QID")
OUT=$OUTDIR/trails/${UNIT}_trails.csv
IMG=$OUTDIR/imgs/${UNIT}_imgs.txt
FRAMEDIR=$OUTDIR/frames/$UNIT
mkdir -p "$OUTDIR"/{trails,imgs,logs} "$FRAMEDIR"

if [ -s "$OUT" ]; then echo "skip $UNIT (done)"; exit 0; fi

# STREAM=1 (the campaign default): the frames for this unit are deleted on ANY
# exit path, success or failure. Sep-Dec 2024 is ~91 TB of diffs against ~2.5 TB
# free, so nothing may be left behind for a later retune -- a retry re-downloads.
if [ "${STREAM:-1}" = 1 ]; then
  trap 'rm -rf "$FRAMEDIR"' EXIT
fi

IFS=',' read -r -a FFD_ARR <<< "$FFDS"
IFS=',' read -r -a FIL_ARR <<< "$FILTS"
NEXP=${#FFD_ARR[@]}

ngot=0
for i in "${!FFD_ARR[@]}"; do
  ffd=${FFD_ARR[$i]}
  filt=${FIL_ARR[$i]:-${FIL_ARR[0]}}
  # IBE layout: sci/<YYYY>/<MMDD>/<fracday>/ztf_<ffd>_<field>_<filt>_c<cc>_o_q<q>_*
  dir="$IBE/${ffd:0:4}/${ffd:4:4}/${ffd:8}"
  stem=$(printf 'ztf_%s_%06d_%s_c%02d_o_q%d' "$ffd" "$FIELD" "$filt" "$CCDID" "$QID")
  ok=1
  for suf in scimrefdiffimg.fits.fz mskimg.fits; do
    f=$FRAMEDIR/${stem}_${suf}
    if [ ! -s "$f" ]; then
      # --retry-all-errors is unavailable in klone's curl, hence plain --retry.
      curl -sS --max-time 600 --retry 5 --retry-delay 3 \
        -A "streak_radon/1.0 (klone)" \
        -o "$f" "$dir/${stem}_${suf}" || ok=0
    fi
    # A 404 still writes a short HTML/empty body; treat anything tiny as absent.
    [ -s "$f" ] && [ "$(stat -c%s "$f")" -gt 100000 ] || { rm -f "$f"; ok=0; }
  done
  if [ "$ok" = 1 ]; then ngot=$((ngot+1)); else
    echo "WARN $UNIT: missing products for $stem"
    rm -f "$FRAMEDIR/${stem}_"*
  fi
done

if [ "$ngot" -eq 0 ]; then
  echo "ERROR $UNIT: 0/$NEXP exposures downloaded"; exit 1
fi
[ "$ngot" -eq "$NEXP" ] || echo "NOTE $UNIT: proceeding with $ngot/$NEXP exposures"

# FITJSON=1 dumps every surviving fit, including the raw mf_snr that the hldet CSV
# does not carry. Required by the FP calibration: without it the only available SNR
# proxy is 2.5/ln10/sigmag, which does not discriminate (measured p50 134 for FPs).
FITARG=""
if [ "${FITJSON:-0}" = 1 ]; then
  mkdir -p "$OUTDIR/fits"
  FITARG="--json $OUTDIR/fits/${UNIT}.json"
fi

STAMPARG=""
if [ "${STAMPS:-0}" = 1 ]; then
  mkdir -p "$OUTDIR/stamps"
  STAMPARG="--stamps $OUTDIR/stamps/${UNIT}_stamps.npz \
--stamp-size ${STAMP_SIZE:-64} --stamp-plane ${STAMP_PLANE:-diff}"
fi

$PY "$SR/bin/run_ztf_sequence.py" --config "$CFG" \
  --diffs "$FRAMEDIR/*_scimrefdiffimg.fits.fz" \
  --out "$OUT" --imgs "$IMG" $FITARG $STAMPARG > "$OUTDIR/logs/${UNIT}.log" 2>&1
rc=$?
dets=$(($(wc -l < "$OUT" 2>/dev/null || echo 1)-1))
echo "$UNIT nexp=$ngot rc=$rc dets=$dets"
exit $rc
