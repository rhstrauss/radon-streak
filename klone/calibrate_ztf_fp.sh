#!/bin/bash
# calibrate_ztf_fp.sh -- negated-diff false-positive calibration for ZTF.
#
#   TAG=ztf_fpcal NUNITS=40 bash klone/calibrate_ztf_fp.sh
#
# WHY: config/ztf.yaml ships mf_snr_min = 6.0, inherited from the P6 head-to-head
# against the ztf_streak matched filter. That was a RECALL test on two exposures;
# it never measured the false-positive rate. G96's threshold (11.0) came from a
# separate negated-diff calibration, and G96 is the only survey in this repo whose
# threshold has that backing. Running ZTF Sep-Dec 2024 (~3.2M quadrants) on an
# uncalibrated threshold risks a multi-million-row FP flood that would poison
# heliolinc linking -- and would only be discovered after 91 TB of downloads.
#
# METHOD: negate the diff (STREAKRADON_NEGATE=1). Real sources become negative,
# so a positive-streak detector sees a source-free image with completely
# realistic noise, subtraction residuals, and detector artifacts. Every surviving
# detection is therefore a false positive BY CONSTRUCTION. The output histogram
# of mf_snr over FPs gives the threshold: pick mf_snr_min above the bulk, at the
# FP-per-quadrant rate you are willing to hand the linker.
#
# Pair this with the recall side (bin/gate_recall.py / measure_efficiency.py on
# injected trails) before freezing a threshold: this script alone tells you
# purity, not completeness.
set -uo pipefail
SR=${SR:-/gscratch/astro/rstrau/streak_radon_ztf}
TAG=${TAG:-ztf_fpcal}
NUNITS=${NUNITS:-40}
SRC_TAG=${SRC_TAG:-ztf_smoke}          # unit list to draw the sample from
TIME=${TIME:-4:00:00}
PART=${PART:-ckpt-g2}   # PART=ckpt-all when ckpt-g2 is starved (GPU superset)

W=$SR/work/$TAG
SRC=$SR/work/$SRC_TAG/${SRC_TAG}_units.txt
[ -s "$SRC" ] || { echo "ABORT: no $SRC (run klone/scrape_ztf.py --tag $SRC_TAG)"; exit 1; }
mkdir -p "$W"/{trails,imgs,logs,frames}

# Prefer multi-exposure units so the repetition filter is exercised too: an FP
# rate measured without it would be pessimistic relative to the real pipeline.
awk '$5>=2' "$SRC" | head -n "$NUNITS" > "$W/${TAG}_units.txt"
n=$(wc -l < "$W/${TAG}_units.txt")
[ "$n" -gt 0 ] || { echo "ABORT: no multi-exposure units in $SRC"; exit 1; }
echo "FP calibration on $n negated units (from $SRC_TAG)"

JID=$(sbatch --parsable -A astro -p $PART --gres=gpu:1 -c 8 --mem=48G \
  --time=$TIME --requeue -J sr_${TAG} -o "$W/logs/fpcal.out" \
  --wrap="export STREAKRADON_GPU_CLEAN=1 STREAKRADON_CLEAN_ITER=10 \
STREAKRADON_NEGATE=1 OMP_NUM_THREADS=8 SR=$SR STREAM=1; \
while read -r night field ccd qid nexp ffds filts; do \
  bash $SR/klone/process_ztf_unit.sh \$night \$field \$ccd \$qid \$ffds \$filts $W; \
done < $W/${TAG}_units.txt; \
$SR/klone/fp_report.py --trails '$W/trails/*_trails.csv' --nunits $n")
rc=$?
if [ $rc -ne 0 ] || [ -z "$JID" ]; then echo "SUBMIT FAILED (rc=$rc)"; exit 1; fi
echo "SUBMITTED $TAG jobid=$JID  (report lands in $W/logs/fpcal.out)"
echo "$JID" > "$W/array_jobid.txt"
