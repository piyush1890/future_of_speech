"""
Test Hindi TTS by manually constructing phoneme sequences.
The model only knows ARPABET from training, so we write Hindi words
using ARPABET phonemes that approximate the Hindi pronunciation.
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import soundfile as sf
import torch

from inference.synthesize_rvq import ArticulatoryTTSRVQ
from models.phoneme_vocab import PhonemeVocab


def main():
    tts = ArticulatoryTTSRVQ(
        rvq_checkpoint='checkpoints_rvq/rvq_best.pt',
        transformer_checkpoint='checkpoints_rvq/transformer_best.pt',
        vocab_path='data/processed_all/vocab_mfa.json',
        norm_stats_path='data/features_all/norm_stats.npz',
        device='cpu',
    )

    with open('data/features_all/speaker_embeddings.json') as f:
        spk_embs = json.load(f)
    spk_emb = np.array(list(spk_embs.values())[0], dtype=np.float32)

    # Hindi words written in ARPABET (approximate pronunciation)
    # Vowels: AH=अ, AA=आ, IH=इ, IY=ई, UH=उ, UW=ऊ, EH=ए, OW=ओ, AW=औ, AY=ऐ
    # Consonants: similar to English: N=न, M=म, T=त, D=द, K=क, G=ग, etc.
    # Note: Hindi has retroflex consonants that don't exist in English
    #       We'll use closest approximations
    hindi_tests = {
        "namaste": ["N", "AH1", "M", "AH0", "S", "T", "EY1"],
        "dhanyavaad": ["D", "AH0", "N", "Y", "AH0", "V", "AA1", "D"],
        "mera_naam": ["M", "EH1", "R", "AH0", "<sil>", "N", "AA1", "M"],
        "aap_kaise_hain": ["AA1", "P", "<sil>", "K", "EY1", "S", "EY0", "<sil>", "HH", "EY1", "N"],
        "kya_haal_hai": ["K", "Y", "AH1", "<sil>", "HH", "AA1", "L", "<sil>", "HH", "EH1"],
    }

    vocab = tts.vocab

    for name, phonemes in hindi_tests.items():
        # Build token indices
        indices = vocab.encode(phonemes, add_bos_eos=True)
        phoneme_ids = torch.tensor([indices], dtype=torch.long)
        spk = torch.from_numpy(spk_emb).unsqueeze(0)

        with torch.no_grad():
            token_ids, _ = tts.transformer.generate(phoneme_ids, spk)
            features_norm = tts.rvq.decode_indices(token_ids)
            features_norm = features_norm.squeeze(0).numpy()
        features = features_norm * tts.feat_std + tts.feat_mean

        ema = features[:, :12]
        pitch = features[:, 12]
        loudness = features[:, 13]

        wav = tts.sparc.decode(ema, pitch, loudness, spk_emb)
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()

        out_path = f"outputs/hindi_{name}.wav"
        sf.write(out_path, wav, tts.sparc.sr)
        print(f"Saved {out_path}: phonemes={phonemes}")

    print("\nListen to outputs/hindi_*.wav")
    print("Note: Hindi pronunciation is approximated with English ARPABET")


if __name__ == "__main__":
    main()
