# Mixed + Vicky Hindi Dialogue Training Upload

Use `train_all_movies.jsonl` for the full post-music-guard training set. Each row has relative `audio` and `voice_prompt` paths rooted at this folder.

Per-movie data lives under `movies/<movie_id>/` with only target clips, voice prompts, and small manifests. Source videos, rejected review folders, and intermediate processing outputs are intentionally excluded.
