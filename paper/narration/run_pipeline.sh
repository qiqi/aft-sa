#!/usr/bin/env bash
# Paper -> narrated audio. Run from this directory after rebuilding ../sa-ai.pdf
# (the extraction reads ../sa-ai.tex plus ../sa-ai.aux and ../sa-ai.bbl for numbers).
#
#   1. build_extracted.py  ->  extracted.md           clean Markdown, exact LaTeX math
#   2. build_script.py     ->  narrated_script.txt    verbatim prose + spoken math
#   3. synthesize.py       ->  paper_narrated.wav/.mp3, paper_narrated_chapters.txt
#
# Environment (one time; kept off the root disk):
#   ENVDIR=/local_data_1/qiqi/narration-env
#   uv venv --python 3.10 $ENVDIR/tts
#   uv pip install --python $ENVDIR/tts/bin/python "torch<2.7" \
#       --index-url https://download.pytorch.org/whl/cu121      # V100-compatible build
#   uv pip install --python $ENVDIR/tts/bin/python kokoro soundfile espeakng-loader \
#       imageio-ffmpeg pypandoc_binary "transformers<4.50"
set -euo pipefail
ENVDIR=${ENVDIR:-/local_data_1/qiqi/narration-env}
export VIRTUAL_ENV=$ENVDIR/tts
export HF_HOME=$ENVDIR/hf
export UV_CACHE_DIR=$ENVDIR/uv-cache
export PATH=$VIRTUAL_ENV/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$(dirname "$0")"

python build_extracted.py
python build_script.py
python synthesize.py "$@" 2>&1 | grep -v -i warn
