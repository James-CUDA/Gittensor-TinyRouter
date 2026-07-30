# Discord announcement — TinyRouter milestones

Paste this:

```
🐡 **TinyRouter — 3 milestones**

**1️⃣ Prompt triage** *(no API)*
Train a tiny head to predict **domain** + **difficulty**.
Submit head + weights only (<1M params, 1/day).
Host scores king vs challenger; beat the king by ≥0.02 → merge.
Dataset: https://huggingface.co/datasets/James-Cuda/tinyrouter-m1
Guide: https://github.com/James-CUDA/Gittensor-TinyRouter/blob/main/docs/MILESTONE1.md

**2️⃣ Model routing** *(still offline)*
Learn which model fits a prompt using cached router corpora.
Train/eval without live OpenRouter.

**3️⃣ Live TinyRouter** *(full competition → TAO)*
Route each query to 3 models × 3 roles (Thinker / Worker / Verifier).
Pool: Qwen · Gemini Flash Lite · DeepSeek Flash
Beat the king on the leaderboard → earn **TAO** (SN74).
Submit: https://github.com/James-CUDA/Gittensor-TinyRouter/blob/main/SUBMITTING.md

🐙 Repo: https://github.com/James-CUDA/Gittensor-TinyRouter

Start at 1 → level up to 2 → compete on 3.
```
