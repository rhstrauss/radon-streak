#!/bin/bash
# launch_ztf.sh -- scrape + submit ONE batch of ZTF nights, sized to fit both the
# QOS submit cap and the ckpt walltime clamp.
#
#   bash klone/launch_ztf.sh <tag> <start YYYY-MM-DD> <end YYYY-MM-DD> [ccdid] [qid]
#
# e.g.  bash klone/launch_ztf.sh ztf_sep_a 2024-09-01 2024-09-04
#       bash klone/launch_ztf.sh ztf_pilot 2024-09-01 2024-09-01 1,2 1
#
# WHY BATCHES AND NOT ONE CAMPAIGN ARRAY: ZTF is ~19,100 work units per night
# (299 units/night/quadrant x 64 readout channels), so Sep 1 - Dec 31 2024 is
# ~2.3M units and ~91 TB of diff+mask downloads. Neither the 2000-task submit cap
# nor the ~5:05 ckpt walltime clamp can express that as a single array, and
# /gscratch/astro has ~2.5 TB free, so the campaign is necessarily a sequence of
# streaming batches of a few nights each. This script is one link in that chain;
# run it repeatedly (or from a /loop) as headroom frees.
#
# ORDER: the requested campaign order is Sep-Dec 2024 first, then BACK through
# Jul and Jun. Note that unlike CSS -- whose telescopes are dark for the Arizona
# monsoon, so Jul/Aug 2024 does not exist at all -- ZTF observed continuously:
# Jun 2024 = 10,933 and Jul 2024 = 11,175 quadrant-exposures per readout channel,
# so the Jun/Jul extension is real data, not a gap.
set -uo pipefail
SR=${SR:-/gscratch/astro/rstrau/streak_radon_ztf}
PY=${PY:-/mmfs1/home/rstrau/miniconda3/envs/mpchecker/bin/python}
TAG=${1:?usage: launch_ztf.sh <tag> <start> <end> [ccdid] [qid]}
START=${2:?start date}
END=${3:?end date}
CCDID=${4:-}
QID=${5:-}

W=$SR/work/$TAG
mkdir -p "$W"

SCRAPE_ARGS=(--tag "$TAG" --start "$START" --end "$END")
[ -n "$CCDID" ] && SCRAPE_ARGS+=(--ccdid "$CCDID")
[ -n "$QID" ] && SCRAPE_ARGS+=(--qid "$QID")

if [ -s "$W/${TAG}_units.txt" ]; then
  echo "unit list already present ($(wc -l < "$W/${TAG}_units.txt") units); skipping scrape"
else
  echo "scraping IBE metadata for $TAG ($START .. $END)..."
  $PY "$SR/klone/scrape_ztf.py" "${SCRAPE_ARGS[@]}" | tee "$W/scrape.log"
  # The scraper exits 2 if any night's metadata query failed; do not launch a
  # partial batch silently -- a missing night looks exactly like an empty night.
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "ABORT: scrape returned rc=$rc (see $W/scrape.log). Fix before submitting."
    exit 1
  fi
fi

N=$(wc -l < "$W/${TAG}_units.txt")
[ "$N" -gt 0 ] || { echo "ABORT: 0 units for $START..$END"; exit 1; }
echo "$TAG: $N units"

TAG=$TAG SEC_PER_UNIT=${SEC_PER_UNIT:-60} CONC=${CONC:-64} \
  TIME=${TIME:-5:00:00} STREAM=${STREAM:-1} STAMPS=${STAMPS:-0} \
  bash "$SR/klone/submit_ztf.sh"
