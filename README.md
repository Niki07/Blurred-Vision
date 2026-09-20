# Blurred Vision

**Comparing Multimodal AI Image Classification Across Blur Types and Severity Levels**

This repository contains the code, experimental pipeline, results, and analysis for an independent research study evaluating the robustness of multimodal AI models under blurred visual inputs.

**Paper status:** Manuscript in preparation for submission.

## Overview

Multimodal AI systems are increasingly used for tasks involving visual inputs, but their performance may decline when images are degraded.

This study evaluates how nine multimodal AI models perform under three types of blur corruption from CIFAR-10-C:

- Motion blur
- Defocus blur
- Glass blur

Each corruption was evaluated across five severity levels.

### Experiment Scale

- **9 multimodal AI models**
- **3 blur types**
- **5 severity levels**
- **1,400 class-balanced base images**
- **189,000 total classification trials**

The models were evaluated using:

- Classification accuracy
- Response latency
- Inference cost
- Robustness across increasing corruption severity

## Research Question

How do motion blur, defocus blur, and glass blur across increasing severity levels affect image-classification performance across multimodal AI models?

## Models Evaluated

The experiment compared models across several providers:

- Claude Opus 5
- Claude Sonnet 5
- Claude Haiku 4.5
- GPT-5.6 Sol
- GPT-5.6 Terra
- GPT-5.6 Luna
- Gemini 3.1 Flash Lite
- Grok 4.3
- Kimi K3

All models were accessed through OpenRouter and evaluated using the same image set and classification prompt.

## Dataset

The experiment uses **CIFAR-10-C**, a corruption benchmark derived from CIFAR-10.

A class-balanced subset of **1,400 base images** was selected, consisting of 140 images from each of the ten CIFAR-10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The same image indices were reused across every model, blur type, and severity level to maintain a controlled comparison.

## Experimental Pipeline

The evaluation pipeline was written in **Python** and designed to support large-scale API-based experimentation.

The pipeline:

- Loads corrupted CIFAR-10-C images
- Sends images to multimodal models through OpenRouter
- Uses a fixed classification prompt and temperature
- Parses model responses into CIFAR-10 labels
- Records exact model IDs
- Tracks classification accuracy
- Measures response latency
- Records per-request inference cost
- Saves raw model responses
- Distinguishes invalid responses from incorrect classifications
- Writes results to disk after every completed request
- Automatically skips previously completed trials when restarting
- Supports concurrent model evaluation
- Monitors API balance during long-running experiments

This design allowed experiments to resume safely after interruptions without repeating completed API calls.

## Key Findings

Across all blur types and severity levels:

- **Gemini 3.1 Flash Lite achieved the highest overall accuracy at 73.29%.**
- Claude Opus 5 achieved the second-highest overall accuracy at 61.92%.
- Model performance differed substantially across blur types and severity levels.
- **Defocus blur produced the highest average accuracy** across models.
- **Glass blur produced the lowest average accuracy.**
- Accuracy generally declined as blur severity increased, although the relationship was not always linear.
- Higher inference cost and latency did not consistently correspond to stronger classification performance.

These results suggest that visual robustness depends on the specific model, corruption type, and corruption severity rather than overall model capability alone.

## Results

The `results/` directory contains experiment outputs and visualizations for:

- Motion blur
- Glass blur
- Defocus blur
- Combined cross-blur analysis

Example analyses include:

- Accuracy by model and severity
- Accuracy across blur types
- Model robustness drop-off
- Class-level accuracy
- Cost vs. accuracy
- Latency vs. accuracy
- Cross-model performance comparisons

## Repository Structure

```text
Blurred-Vision/
│
├── src/
│   ├── pipeline.py       # Main experiment pipeline
│   ├── classify.py       # Image classification requests
│   ├── client.py         # OpenRouter client
│   ├── config.py         # Models and experiment configuration
│   ├── data.py           # CIFAR-10-C loading and sampling
│   └── persist.py        # Result persistence and resume logic
│
├── scripts/
│   ├── generate_figures.py
│   ├── generate_overall_figures.py
│   ├── pilot_comparison.py
│   └── smoke_test.py
│
├── results/
│   ├── motion/
│   ├── glass/
│   ├── defocus/
│   └── overall/
│
├── notebooks/
│   └── proof_of_concept.py
│
├── main.py
├── requirements.txt
└── README.md
