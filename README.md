# Blurred Vision: Comparing Multimodal AI Image Classification Across Blur Types and Severity Levels

**Status: Status: Complete

## Overview

This project studies how modern multimodal AI systems (ChatGPT, Claude, Gemini, accessed via OpenRouter) handle blurred images during image classification. It uses the **CIFAR-10-C** benchmark, focusing on motion, glass, and defocus blur across five severity levels, and treats each model as a black-box classifier under controlled visual degradation.

Full research framing, terminology, and design rationale live in [`Research context.md`](./Research%20context.md).

## Setup

```bash
git clone <repo-url>
cd ResearchProject
pip install -r requirements.txt
cp .env.example .env  # then fill in your OpenRouter API key
```

Download CIFAR-10-C into `data/` — see `data/README.md`.

## Usage

```bash
python -m src.pipeline
```

Currently runs the ported proof-of-concept (one image, one free test model, defocus blur, severity 1) using the persistence-safe append pattern. Not yet the full experiment — see TODOs in `src/pipeline.py`.

## Repository Structure

```
src/            classification pipeline (config, data loading, OpenRouter client, persistence)
data/           CIFAR-10-C .npy files (not tracked in git — see data/README.md)
results/        experiment output CSVs (not tracked in git)
notebooks/      archived original Colab proof-of-concept
```

## Citation

TODO

## License

MIT — see [LICENSE](./LICENSE).
