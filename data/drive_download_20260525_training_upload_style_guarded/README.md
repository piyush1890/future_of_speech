# Drive Download Hindi Dialogue Training Upload

Use `train_all_movies.jsonl` for the combined dataset. Each row has relative `audio` and `voice_prompt` paths rooted at this folder.

Per-movie copies are under `movies/<movie_id>/` with their own clips, prompts, manifests, and summary.

This package uses the repaired style-aware voice prompt pairing plus guarded global join outputs.
