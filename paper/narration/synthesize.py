#!/usr/bin/env python3
"""Step 4: narrated_script.txt -> paper_narrated.wav (+ .mp3).

Kokoro (KPipeline lang 'a', voice af_heart, speed 1.0), one paragraph at a
time, waveforms joined with short silences: a longer one around headings and
display equations. Also writes paper_narrated_chapters.txt, the start time of
every section and subsection.

Usage: synthesize.py [--voice af_heart] [--device cuda] [--limit N]
"""
import argparse
import os
import re
import subprocess
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np            # noqa: E402
import soundfile as sf        # noqa: E402
import torch                  # noqa: E402
from kokoro import KPipeline  # noqa: E402

from pronounce import pronounce  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 24000

ap = argparse.ArgumentParser()
ap.add_argument('--voice', default='af_heart')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--limit', type=int, default=0, help='only the first N paragraphs')
ap.add_argument('--out', default='paper_narrated')
args = ap.parse_args()

paras = [p.strip() for p in
         open(os.path.join(HERE, 'narrated_script.txt'), encoding='utf-8').read().split('\n\n')
         if p.strip()]
if args.limit:
    paras = paras[:args.limit]

pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device=args.device)


def silence(seconds):
    return np.zeros(int(SR * seconds), dtype=np.float32)


def kind(p):
    if re.match(r'^(Section \d|Subsection [A-J])\. ', p) or p in (
            'Abstract.', 'Nomenclature.', 'Acknowledgments.') or p.startswith('Appendix:'):
        return 'heading'
    if re.match(r'^Equation \d+[.,]', p):
        return 'equation'
    if re.match(r'^(Figure|Table) \d+', p):
        return 'caption'
    return 'text'


def letters(s):
    return len(re.sub(r'[^A-Za-z0-9]', '', s))


chunks, chapters, short = [], [], []
t0 = time.time()
pos = 0
for n, p in enumerate(paras):
    k = kind(p)
    gap = {'heading': 1.0, 'equation': 0.7, 'caption': 0.6, 'text': 0.45}[k]
    if n:
        chunks.append(silence(gap))
        pos += int(SR * gap)
    if k == 'heading' or n == 0:
        chapters.append((pos / SR, p.rstrip('.')))
    spoken = pronounce(p)
    got = 0
    for graphemes, phonemes, audio in pipe(spoken, voice=args.voice, speed=1.0):
        if audio is None or len(audio) == 0:
            continue
        a = audio.detach().cpu().numpy() if hasattr(audio, 'detach') else np.asarray(audio)
        chunks.append(a.astype(np.float32))
        pos += len(a)
        got += letters(graphemes)
    want = letters(re.sub(r'\[a\]\(/[^)]*/\)', 'a', spoken))
    if got < 0.97 * want:
        short.append((n, want, got, p[:70]))
    if k in ('heading', 'equation'):
        chunks.append(silence(0.35))
        pos += int(SR * 0.35)
    if n % 25 == 0:
        print('paragraph %d/%d  audio %.1f min  wall %.0f s'
              % (n + 1, len(paras), pos / SR / 60, time.time() - t0), flush=True)

wave = np.concatenate(chunks)
wav_path = os.path.join(HERE, args.out + '.wav')
sf.write(wav_path, wave, SR, subtype='PCM_16')
dur = len(wave) / SR



def chapter_lines(chapters):
    """Start time of every section and subsection, numbered as in the paper
    (II, II.A, ...), for jumping to a point in the audio."""
    roman = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']
    sec = ''
    for t, title in chapters:
        m = re.match(r'Section (\d)\. (.*)', title)
        if m:
            sec = roman[int(m.group(1))]
            num, name = sec, m.group(2)
        else:
            m = re.match(r'Subsection ([A-Z])\. (.*)', title)
            if m:
                num, name = '%s.%s' % (sec, m.group(1)), m.group(2)
            else:
                num, name = '', title
        yield '%d:%02d:%02d  %-6s %s' % (t // 3600, t % 3600 // 60, t % 60, num, name)


with open(os.path.join(HERE, args.out + '_chapters.txt'), 'w') as f:
    f.write('\n'.join(chapter_lines(chapters)) + '\n')

mp3_path = None
try:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    mp3_path = os.path.join(HERE, args.out + '.mp3')
    subprocess.run([ff, '-y', '-loglevel', 'error', '-i', wav_path, '-codec:a', 'libmp3lame',
                    '-b:a', '64k', '-ac', '1', mp3_path], check=True)
except Exception as e:  # noqa: BLE001
    print('mp3 conversion skipped:', e)
    mp3_path = None

print('device: %s (%s)' % (args.device, torch.cuda.get_device_name(0) if args.device == 'cuda' else 'cpu'))
print('paragraphs spoken: %d' % len(paras))
if short:
    print('WARNING: %d paragraphs came back shorter than their text:' % len(short))
    for row in short:
        print('   para %d: %d letters wanted, %d spoken: %s' % row)
else:
    print('every paragraph was spoken in full (letter counts match)')
print('duration: %d:%02d:%02d (%.1f min), wall time %.0f s'
      % (dur // 3600, dur % 3600 // 60, dur % 60, dur / 60, time.time() - t0))
print('chapters: %s' % os.path.join(HERE, args.out + '_chapters.txt'))
print('wav: %s (%.0f MB)' % (wav_path, os.path.getsize(wav_path) / 1e6))
if mp3_path:
    print('mp3: %s (%.0f MB)' % (mp3_path, os.path.getsize(mp3_path) / 1e6))
