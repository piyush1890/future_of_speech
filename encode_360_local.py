"""
Encode train-clean-360 from the BACK of the file list on local Mac.
Colab encodes from the front. They meet in the middle.
Files saved to data/features_360_raw/ locally (merge with Drive download later).
"""
import sys
sys.path.insert(0, '.')

import glob
import time
from pathlib import Path
import numpy as np


def main():
    import torch
    from sparc import load_model

    # Paths
    LIBRI_DIR = 'data/LibriSpeech/train-clean-360'
    OUTPUT_DIR = Path('data/features_360_raw')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all FLAC files (sorted to match Colab's order)
    files = sorted(glob.glob(f'{LIBRI_DIR}/**/*.flac', recursive=True))
    total = len(files)
    print(f'Found {total} files')

    if total == 0:
        print(f'ERROR: No files found in {LIBRI_DIR}')
        print(f'Download first: cd data && curl -L -o tc360.tar.gz http://www.openslr.org/resources/12/train-clean-360.tar.gz && tar xzf tc360.tar.gz && rm tc360.tar.gz')
        return

    # Encode from the BACK (reversed) so we don't collide with Colab
    files = files[::-1]

    # Check what's already done locally
    done = {p.stem for p in OUTPUT_DIR.glob('*.npz') if p.stem != 'norm_stats'}
    print(f'Already done locally: {len(done)}')

    # Load SPARC on MPS
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Loading SPARC on {device}...')
    coder = load_model('en', device=device)
    print('SPARC loaded.')

    success = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for i, f in enumerate(files):
        utt_id = Path(f).stem

        if utt_id in done:
            skipped += 1
            continue

        out_path = OUTPUT_DIR / f'{utt_id}.npz'
        if out_path.exists():
            skipped += 1
            continue

        try:
            code = coder.encode(f)
            ema = np.asarray(code['ema'], dtype=np.float32)
            pitch = np.asarray(code['pitch'], dtype=np.float32).squeeze(-1)
            loudness = np.asarray(code['loudness'], dtype=np.float32).squeeze(-1)
            spk_emb = np.asarray(code['spk_emb'], dtype=np.float32)

            min_len = min(ema.shape[0], pitch.shape[0], loudness.shape[0])
            ema, pitch, loudness = ema[:min_len], pitch[:min_len], loudness[:min_len]

            np.savez_compressed(str(out_path), ema=ema, pitch=pitch, loudness=loudness, spk_emb=spk_emb)
            success += 1
        except Exception as e:
            if failed < 5:
                print(f'FAILED {utt_id}: {e}')
            failed += 1

        # Progress every 100 files
        if (i + 1) % 100 == 0 and success > 0:
            elapsed = time.time() - t_start
            rate = success / max(elapsed, 1)
            remaining = (total - i - 1) / max(rate, 0.01)
            print(f'  [{i+1}/{total}] {success} new, {skipped} skipped, ~{remaining/60:.0f}min left')

    elapsed = time.time() - t_start
    total_saved = len(list(OUTPUT_DIR.glob('*.npz')))
    print(f'\nDone in {elapsed/3600:.1f}h: {success} new, {skipped} skipped, {failed} failed')
    print(f'Total files in {OUTPUT_DIR}: {total_saved}')


if __name__ == '__main__':
    main()
