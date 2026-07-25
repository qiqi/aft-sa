#!/bin/bash
# Move large solution dirs off the /home mount to /local_data/qiqi/sa-ai/
# (mimicking 017-v100-dev's /local_disk/qiqi convention), leaving symlinks
# behind so every repo-relative path keeps working.  Per-directory:
# rsync -> second rsync pass must transfer nothing -> rm -> ln -s.
# Skips dirs that are already symlinks or have open file handles.
set -u
DEST=/local_data/qiqi/sa-ai
SA=/home/qiqi/flexcompute/sa-ai
mkdir -p "$DEST/daedalus" "$DEST/flow360_fr"

migrate() {
    local src=$1 dst=$2
    [ -L "$src" ] && { echo "SKIP(link) $src"; return; }
    [ -d "$src" ] || { echo "SKIP(gone) $src"; return; }
    if lsof +D "$src" >/dev/null 2>&1; then
        echo "SKIP(busy) $src"; return
    fi
    rsync -a "$src/" "$dst/" || { echo "FAIL(rsync) $src"; return; }
    # verify: a second pass must find nothing left to send
    local left
    left=$(rsync -a --dry-run --out-format=%n "$src/" "$dst/" | wc -l)
    if [ "$left" -ne 0 ]; then echo "FAIL(verify $left) $src"; return; fi
    rm -rf "$src" && ln -s "$dst" "$src" && echo "MOVED $src -> $dst"
}

for d in "$SA"/daedalus/case_*; do
    base=$(basename "$d")
    [ "$base" = "case_cavity_L2_saai_a6" ] && { echo "SKIP(running) $d"; continue; }
    migrate "$d" "$DEST/daedalus/$base"
done

for d in "$SA"/flow360_fr/*/; do
    d=${d%/}
    migrate "$d" "$DEST/flow360_fr/$(basename "$d")"
done
echo MIGRATE-DONE
df -h / /local_data | tail -2
