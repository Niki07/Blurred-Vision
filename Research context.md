# Research Context

## Project Overview

This project studies how modern multimodal AI systems handle blurred images during image classification.

**Core research question:**  
How do the effects of motion, glass, and defocus blur across five severity levels on image-classification performance compare among ChatGPT, Claude, and Gemini?

The study uses the public **CIFAR-10-C** benchmark. CIFAR-10-C contains corrupted versions of CIFAR-10 images. The project focuses only on:

- motion blur
- glass blur
- defocus blur

Each corruption has **five severity levels**, from least severe to most severe.

This project is **not** about training new image classifiers. It treats existing multimodal AI models as black-box classifiers under a controlled visual-degradation task.

---

## Main Research Goal

The project compares performance across four main dimensions:

1. **AI model**
   - ChatGPT / OpenAI
   - Claude / Anthropic
   - Gemini / Google

2. **Blur type**
   - motion
   - glass
   - defocus

3. **Blur severity**
   - levels 1 through 5

4. **Outcome**
   - top-1 classification correctness / accuracy

Questions of interest include:

- Does accuracy decrease as blur severity increases?
- Are some blur types more damaging than others?
- Is one model more robust than another?
- Does the strongest model change as severity increases?
- Do different models degrade differently across blur types?

---

## Dataset Context

The project uses **CIFAR-10-C**, which is based on CIFAR-10.

CIFAR-10 contains 10 classes:

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

Relevant files include:

- `motion_blur.npy`
- `glass_blur.npy`
- `defocus_blur.npy`
- `labels.npy`

Each corruption `.npy` file contains **50,000 images total**:

- images 0–9,999 = severity 1
- images 10,000–19,999 = severity 2
- images 20,000–29,999 = severity 3
- images 30,000–39,999 = severity 4
- images 40,000–49,999 = severity 5

The same underlying CIFAR image can therefore be viewed across all five severity levels by using the same base image index with 10,000-image offsets.

`labels.npy` contains the correct CIFAR-10 label for each of the underlying 10,000 test images.

CIFAR-10 label mapping:

- 0 = airplane
- 1 = automobile
- 2 = bird
- 3 = cat
- 4 = deer
- 5 = dog
- 6 = frog
- 7 = horse
- 8 = ship
- 9 = truck

Important terminology detail: `cat_number = 0` can mean “the first cat found after filtering,” but the actual CIFAR-10 class label for cat is **3**.

---

## Experimental Framing

The experiment should remain controlled and intentionally simple.

The same selected underlying image sample should be used consistently across:

- all three AI models
- all three blur types
- all five severity levels

This avoids confounding model differences with different image difficulty.

Every model should receive the same classification task and the same set of allowed labels.

A typical prompt is conceptually:

> Classify this CIFAR-10 image. Choose exactly one label from: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Respond with only the label.

The purpose of the fixed label set is to make scoring clean and reproducible.

The exact API model identifiers / versions used in the real experiment should be recorded because model versions can change over time.

---

## API / Infrastructure Context

The project is being developed in **Google Colab** using Python.

Colab is used to:

- load `.npy` files
- select images
- convert NumPy arrays into normal image formats when needed
- send image requests to models
- collect predictions
- compare predictions with ground-truth labels
- persist results
- analyze results with pandas

The project was advised to use **OpenRouter** rather than three completely separate API integrations.

Conceptually:

**Google Colab / Python → OpenRouter → chosen model → prediction**

OpenRouter provides a common API layer for models from multiple providers, making it easier to keep the request structure similar while changing only the model ID.

A free OpenRouter route/model can be used for early pipeline testing, but the real experiment should use fixed, explicitly named GPT, Claude, and Gemini models. A dynamic free router should not be used for final data because it may choose different underlying models.

---

## Data Collection Structure

Every individual classification should produce one structured result row.

Core result fields:

- `model`
- `blur_type`
- `severity`
- `image_id`
- `true_label`
- `prediction`
- `correct`

Example:

| model | blur_type | severity | image_id | true_label | prediction | correct |
|---|---|---:|---:|---|---|---:|
| GPT | defocus | 3 | 47 | cat | cat | 1 |
| Claude | defocus | 3 | 47 | cat | dog | 0 |

`correct` is stored numerically:

- `1` = correct
- `0` = incorrect

This raw table becomes the experimental dataset used for later analysis.

---

## Critical Persistence Requirement

A major project constraint is that completed results must **not** remain only in memory until the end of a long run.

The user specifically wants completed results persisted continuously during testing.

Reason:

- API credits could run out mid-run
- Colab could disconnect or crash
- a long batch could fail halfway through

If saving happened only at the end, completed work could be lost.

Therefore, the research pipeline must conceptually preserve results as they are finalized:

**after each completed test, or after the smallest practical batch, the latest finalized rows should already exist in persistent storage.**

Preferred persistent target:

- Google Drive or another persistent location mounted into Colab

A local runtime CSV is useful but should not be the only copy for finalized runs.

Every persisted row should include at minimum:

- model
- blur type
- severity
- image ID
- true label
- prediction
- correct / incorrect

The user may also want downloadable CSV copies, but the core requirement is that the experiment never depends on saving everything only after all API calls finish.

---

## Planned Analysis

After collection, the CSV is the main dataset for analysis.

The raw data can be loaded into pandas and grouped by:

- model
- blur type
- severity

The main summary metric is **classification accuracy**.

Because `correct` is stored as 0/1:

**accuracy = mean(correct)**

Planned outputs include:

- accuracy by model
- accuracy by blur type
- accuracy by severity
- model comparisons within each blur type
- severity trends within each model

Likely visualizations:

- line graphs
- x-axis = severity 1–5
- y-axis = classification accuracy
- separate lines for ChatGPT, Claude, and Gemini
- one graph per blur type

Class-level analysis may be added later if useful, but the main project should remain focused.

---

## Research Scope

The project is intentionally smaller than a full benchmark reproduction.

The user plans to use a **limited fixed sample** of CIFAR-10-C images instead of all available images.

This is deliberate so that:

- API cost stays manageable
- runtime stays manageable
- debugging stays manageable
- the analysis remains understandable

The technical challenge should remain accessible to someone without extensive Python experience.

The project does **not** require:

- training a model
- implementing a neural network
- modifying weights
- advanced computer-vision architecture work

The technical work is mainly:

- loading files
- selecting images
- making API calls
- looping through conditions
- saving structured results
- summarizing with pandas
- making simple plots

---

## Relationship to RobustBench

RobustBench is important background, but this project is not simply reproducing RobustBench.

RobustBench is a standardized robustness benchmarking framework for image-classification models under controlled evaluation settings.

This project is narrower and focuses on:

- modern general-purpose multimodal AI systems
- ChatGPT, Claude, and Gemini
- three specific blur corruptions
- five severity levels
- a smaller controlled research sample
- direct comparison of multimodal model behavior on the same CIFAR-10-C classification task

The strongest distinction is **not** that RobustBench is merely “old.” The better framing is that this project asks a different question about modern multimodal systems under specific blur conditions.

Relevant background resources:

- RobustBench paper: https://arxiv.org/abs/2010.09670
- RobustBench project: https://robustbench.github.io/
- CIFAR-10-C dataset: https://zenodo.org/records/2535967
- CIFAR corruption robustness repository: https://github.com/hendrycks/robustness

---

## Scientific Reasoning Context

The project is being framed using a “data detective” mindset.

Important reasoning principles:

- Do not assume one observed model difference has only one explanation.
- A model performing worse under a blur may reflect true robustness differences, but sample choice or other design factors could also matter.
- Most claims are **statistical tendencies**, not strict predictions.
- One correctly classified highly blurred image does not refute a general degradation trend.
- Conclusions should come from patterns across many images, not isolated examples.
- Claims should stay tied to the tested sample and methodology.

Example of preferred wording:

Instead of:

> Claude is bad at motion blur.

Prefer:

> In the selected CIFAR-10-C sample, Claude showed a larger decline in classification accuracy under motion blur than the comparison models.

---

## Paper Context

The eventual research paper is expected to include sections such as:

- Title
- Author Background
- Abstract
- Introduction
- Background / Related Work
- Methodology
- Results
- Discussion
- Limitations
- Conclusion
- References

Current working title:

**Blurred Vision: Comparing Multimodal AI Image Classification Across Blur Types and Severity Levels**

The paper is intended to be mostly technical while still having a slightly engaging title.

The user has been reviewing previous Pioneer journal papers as style references.

Likely discussion topics include:

- why robustness to visual degradation matters
- CIFAR-10 and CIFAR-10-C
- multimodal AI systems
- RobustBench and related robustness research
- API-based experimental design
- accuracy trends
- limitations of sample size
- limitations caused by changing model/API versions
- implications of differences in blur robustness

---

## Current Project Status

The project has already reached a basic proof-of-concept stage.

Completed so far:

- opened CIFAR-10-C `.npy` files in Colab
- displayed individual images
- displayed the same image across severity levels
- filtered images to specific classes using `labels.npy`
- sent at least one image through OpenRouter
- received a classification response
- created a simple `results.csv`
- confirmed that the basic end-to-end pipeline works

The next phase is moving from a toy test to a controlled experiment using fixed model IDs, a fixed image sample, and persistent result logging.

---

## Important Terminology

Use **multimodal**, not “multimodel,” for systems that can accept different input types such as text and images.

The systems being tested are better described as:

- general-purpose multimodal AI systems
- multimodal AI models

They are not traditional dedicated image classifiers, but this study evaluates their **image-classification performance** under controlled CIFAR-10-C conditions.

---

## Core Mental Model

```text
CIFAR-10-C blurred image
        ↓
Google Colab / Python
        ↓
OpenRouter API
        ↓
Chosen GPT / Claude / Gemini model
        ↓
One CIFAR-10 label
        ↓
Compare with labels.npy
        ↓
correct = 1 or 0
        ↓
persist result row
        ↓
aggregate with pandas
        ↓
accuracy tables + graphs
        ↓
research conclusions
```

This file is intended as a persistent context pack for Claude Code or another coding assistant. It explains the research goal, dataset structure, experimental design, terminology, constraints, and current project state without serving as a step-by-step build guide.
