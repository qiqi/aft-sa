#!/bin/bash
# Cross-host status of the extension matrix.  Usage: status.sh
D=/home/qiqi/flexcompute/sa-ai/scripts/ext2026
for h in 014-v100-dev 017-v100-dev 019-v100-dev; do
  if [ "$h" = "$(hostname -s)" ]; then R=""; else R="ssh -o BatchMode=yes -o ConnectTimeout=8 $h"; fi
  echo "=== $h"
  $R bash -c "
    cd $D/logs 2>/dev/null || exit 0
    done_ok=\$(grep -h 'EXT2026-CASE-DONE' *.log 2>/dev/null | grep -c 'rc=0')
    done_bad=\$(grep -h 'EXT2026-CASE-DONE' *.log 2>/dev/null | grep -vc 'rc=0')
    started=\$(grep -hc '^START' *.log 2>/dev/null | paste -sd+ | bc)
    echo \"  started=\${started:-0} done_ok=\$done_ok failed=\$done_bad\"
    for f in *.log; do
      cur=\$(grep '^START' \$f | tail -1 | awk '{print \$2}')
      fin=\$(grep 'EXT2026-CASE-DONE' \$f | tail -1 | awk '{print \$2}')
      b=\$(grep -c '=== batch' \$f)
      [ \"\$cur\" = \"\$fin\" ] && cur='(idle/next)'
      echo \"    \${f%.log}: running=\$cur batches=\$b\"
    done
    grep -h 'rc=' *.log 2>/dev/null | grep -v 'rc=0' | tail -3
  " 2>/dev/null
done
