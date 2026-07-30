#!/bin/bash
# submit_ztf.sh -- submit one ZTF unit-list batch as a packed, streaming GPU array.
#
#   TAG=ztf_sep_a [PACK=auto] [CONC=64] [TIME=5:00:00] [SEC_PER_UNIT=60]
#   [STREAM=1] [STAMPS=0] bash klone/submit_ztf.sh
#
# Three constraints shape this, all learned the hard way on the CSS campaigns:
#
# 1. QOS CAP. MaxSubmitJobsPerUser=2000 counts EVERY pending array task across
#    ALL of your jobs, so the budget is 2000 minus what is already queued, not
#    2000. And `sbatch`'s exit status is CHECKED here: a rejected submit is
#    otherwise indistinguishable from a successful one, which is exactly how the
#    CSS month run silently processed nothing for ~12 h.
#
# 2. WALLTIME. ckpt-g2 clamps walltime to ~5:05 regardless of what you request,
#    so PACK must not exceed what one task can finish in TIME. SEC_PER_UNIT is
#    the measured per-unit cost (download + GPU detect); PACK is capped at
#    TIME/SEC_PER_UNIT. process_ztf_unit.sh is idempotent, so a task that is
#    preempted or runs out of time resumes at the first unfinished unit.
#
# 3. DISK. ZTF Sep-Dec is ~91 TB of diffs; /gscratch/astro has ~2.5 TB free.
#    STREAM=1 (the DEFAULT here, unlike the CSS scripts) deletes each unit's
#    frames as soon as its trail CSV lands. Turning it off will fill the fileset.
set -uo pipefail
SR=${SR:-/gscratch/astro/rstrau/streak_radon_ztf}
: "${TAG:?set TAG (work/<TAG>/<TAG>_units.txt must exist)}"
CONC=${CONC:-64}                 # also the IRSA download concurrency -- be polite
TIME=${TIME:-5:00:00}
STREAM=${STREAM:-1}
STAMPS=${STAMPS:-0}
STAMP_SIZE=${STAMP_SIZE:-64}
STAMP_PLANE=${STAMP_PLANE:-diff}
SEC_PER_UNIT=${SEC_PER_UNIT:-60}
CFG=${CFG:-ztf}

W=$SR/work/$TAG
UNITS=$W/${TAG}_units.txt
[ -s "$UNITS" ] || { echo "ABORT: no $UNITS (run klone/scrape_ztf.py first)"; exit 1; }
mkdir -p "$W"/{trails,imgs,logs,frames}

N=$(wc -l < "$UNITS")

# --- constraint 1: submit headroom -------------------------------------------
CAP=2000
MARGIN=60
INUSE=$(squeue -u "$USER" -h -r -o "%i" 2>/dev/null | wc -l)
BUDGET=$(( CAP - INUSE - MARGIN ))
echo "TAG=$TAG units=$N"
echo "  submit budget: cap=$CAP in_use=$INUSE margin=$MARGIN -> available=$BUDGET"
if [ "$BUDGET" -lt 1 ]; then
  echo "ABORT: no submit headroom (in_use=$INUSE of $CAP). Wait for the queue to drain."
  exit 1
fi

# --- constraint 2: what one task can finish inside TIME ----------------------
hh=${TIME%%:*}; rest=${TIME#*:}; mm=${rest%%:*}
SECS=$(( 10#$hh * 3600 + 10#$mm * 60 ))
PACK_MAX=$(( SECS / SEC_PER_UNIT ))
[ "$PACK_MAX" -lt 1 ] && PACK_MAX=1

PACK_MIN=$(( (N + BUDGET - 1) / BUDGET ))          # smallest pack that fits the cap
if [ "${PACK:-auto}" = auto ]; then
  PACK=$PACK_MIN
  [ "$PACK" -lt 1 ] && PACK=1
fi
NTASKS=$(( (N + PACK - 1) / PACK ))
echo "  pack: min-for-cap=$PACK_MIN max-for-walltime=$PACK_MAX chosen=$PACK -> ntasks=$NTASKS"

if [ "$NTASKS" -gt "$BUDGET" ]; then
  echo "ABORT: $NTASKS tasks exceeds the available budget of $BUDGET."
  echo "       raise PACK to at least $PACK_MIN, or wait for the queue to drain."
  exit 1
fi
if [ "$PACK" -gt "$PACK_MAX" ]; then
  FITS=$(( BUDGET * PACK_MAX ))
  echo "ABORT: PACK=$PACK needs ~$(( PACK * SEC_PER_UNIT / 3600 ))h per task but"
  echo "       TIME=$TIME allows only $PACK_MAX units/task at ${SEC_PER_UNIT}s each."
  echo "       This batch is too big for the current headroom: split the unit list."
  echo "       At most ~$FITS units can be submitted right now"
  echo "       (BUDGET=$BUDGET tasks x PACK_MAX=$PACK_MAX). Scrape fewer nights per batch."
  exit 1
fi

echo "  conc=$CONC stream=$STREAM stamps=$STAMPS cfg=$CFG"
echo "  est download: $(python3 -c "print(f'{$N*1.7*28.3/1e6:.2f} TB')" 2>/dev/null || echo '?')"

JID=$(sbatch --parsable \
  --array=1-${NTASKS}%${CONC} \
  --job-name=sr_${TAG} \
  --account=astro --partition=ckpt-g2 \
  --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=$TIME --requeue \
  --output=$W/logs/array_%a.out \
  --wrap="export STREAKRADON_GPU_CLEAN=1 STREAKRADON_CLEAN_ITER=10 \
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
SR=$SR CFG=$CFG STREAM=$STREAM STAMPS=$STAMPS STAMP_SIZE=$STAMP_SIZE \
STAMP_PLANE=$STAMP_PLANE; \
lo=\$(( (\${SLURM_ARRAY_TASK_ID} - 1) * $PACK + 1 )); \
hi=\$(( lo + $PACK - 1 )); \
sed -n \"\${lo},\${hi}p\" $UNITS | while read -r night field ccd qid nexp ffds filts; do \
  [ -z \"\$filts\" ] && continue; \
  bash $SR/klone/process_ztf_unit.sh \$night \$field \$ccd \$qid \$ffds \$filts $W; \
done")
rc=$?
if [ $rc -ne 0 ] || [ -z "$JID" ]; then
  echo "SUBMIT FAILED (rc=$rc)"; exit 1
fi
echo "SUBMITTED $TAG jobid=$JID ntasks=$NTASKS pack=$PACK"
echo "$JID" > "$W/array_jobid.txt"
