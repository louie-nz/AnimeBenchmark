# Anime Character Benchmark — MVP
This is a minimal reproducible benchmark for identity consistency,
style fidelity, and short sequence coherence in anime/manga image generation.
## Components
- Prompt bank (identity/style/narrative)
- Preprocessing pipeline (resize, pHash, CLIP embedding)
- Automated metrics (cosine similarity)
- MMMU-style evaluation structure (parsed_output.json, result.json)
## How to run
python preprocessing/preprocess.py
python eval/main_eval.py
