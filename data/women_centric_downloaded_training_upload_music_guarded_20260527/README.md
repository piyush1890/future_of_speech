# Women-Centric Downloaded Hindi Dialogue Training Upload

Use `train_all_movies.jsonl` for the full compact dataset. Each row has relative `audio` and `voice_prompt` paths rooted at this folder.

Per-movie copies live under `movies/<movie_id>/` with target clips, voice prompts, and small manifests only. Source videos, rejected clips, Demucs/RoFormer outputs, and other bulky intermediate folders are intentionally excluded.
