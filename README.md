<p align="center">
<h1 align="center">CollisionPhysEval-t2v</h1>
<h4 align="center">Collision Dynamics Evaluation for Physically Plausible Text-to-Video Generation</h4>
</p>
<p align="center">
  <p align="center">
    <a href="#">Lulu Liu</a><sup>1</sup><br>
    Advisor: <a href="https://jiataogu.me/">Jiatao Gu</a><sup>1</sup><br>
    <sup>1</sup>University of Pennsylvania<br>
    <em>Master's Thesis</em>
  </p>
  <h3 align="center">Built on top of <a href="https://arxiv.org/abs/2506.08009">Self-Forcing</a> | <a href="https://self-forcing.github.io">Self-Forcing Website</a> | <a href="https://huggingface.co/gdhe17/Self-Forcing/tree/main">Base Models (HuggingFace)</a></h3>
</p>

---

This repository extends [Self-Forcing](https://arxiv.org/abs/2506.08009) with an **inference-time scaling pipeline** that generates Best-of-N video candidates and automatically selects the most physically plausible one using a multi-stage physics filtering system.

**CollisionPhysEval-t2v** adds on top:
- Best-of-N candidate generation via iterative latent noise sampling
- Multi-stage physics filtering pipeline using video segmentation + kinematic analysis
- Physics plausibility scoring (CoR, energy dissipation, arc monotonicity, artifact detection)
- A mini benchmark for quantifying physical realism in generated videos

---

## Pipeline Overview

```
Prompt
  │
  ▼
[Self-Forcing Inference] ──► N video candidates
  │
  ▼
[Stage 0] LLM Segmentation (ACTIVE) + First Frame Grounding  (optional, off by default)
  - Use LLM for text segmentation in json format: subject, object, background etc for downstream physics reasoning
  - Qwen-VL validates first frame via bounding boxes
  │
  ▼
[Stage 1a] Hallucination Filter  (optional, off by default)
  - Verification on first 20% of frames 
  - Eliminate videos where the subject, object and background mismatch
  │
  ▼
[Stage 1b] SAM3 Segmentation + Object Preservation  (ACTIVE)
  - SAM3 tracks subject and object centre-of-mass 
  - Filters videos where the subject disappears, flickers, or fails identity consistency checks (colour histogram artifact detection)
  │
  ▼
[Stage 1c] Physics Scoring & Selection (ACTIVE)
  Kinematic analysis on surviving candidates:
  - Arc detection (pb → contact → pa)
  - CoR estimation per bounce
  - Energy dissipation check across bounces
  - Velocity monotonicity (NON_INCREASING vy)
  - Final settlement check
  LLM queries material pair → expected CoR range (Qwen3-8B)
  Score-based selection — highest score wins
  │
  ▼
Selected video (most physically plausible)
```

> **Default behaviour:** Stage 0 and Stage 1a are implemented but disabled by default. The active pipeline runs Stage 1b (SAM3) → Stage 1c (physics scoring). Enable Stage 0 Grounding with `run_stage0=True` and Stage 1a with `qwen_precheck=True`.

---

## Scoring System

Starting score: **100**. Deductions applied per violation:

| Category | Max Deduction | Method |
|---|---|---|
| CoR > 2.0 (severe) | −40 | `n_severe / n_bounces × 40` |
| CoR > 1.0 (impossible) | −25 | `n_impossible / n_bounces × 25` |
| CoR out of expected range | −15 | `n_outrange / n_bounces × 15` |
| Energy gain across bounces | −25 | `violation_pairs / total_pairs × 25` |
| Motion monotonicity (vy) | −20 | `violations / total_steps × 20` |
| Rotation artifact (hist > 2.0) | −5/frame | raw count |
| Not settled at end | −15 | flat |

Expected CoR range is queried once from Qwen3-8B based on the subject and surface material described in the prompt.

---

## Requirements

- Tested with Nvidia GPU 90GB memory
- Linux operating system

## Installation

### Self-Forcing environment
```bash
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

### SAM3 environment (for segmentation)
SAM3 runs as a subprocess in a separate conda environment.
```bash
conda create -n sam3 python=3.10 -y
conda activate sam3
# install SAM3 dependencies per sam3/README
pip install av  # pyav for video decoding
```

Set the SAM3 python path before running the pipeline:
```bash
export SAM3_PYTHON=/path/to/miniconda3/envs/sam3/bin/python
```

---

## Quick Start

### 1. Download checkpoints
```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir wan_models/Wan2.1-I2V-14B-720P
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### 2. Generate Best-of-N candidates
```bash
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/candidates \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/bounce_prompts.txt \
    --num_candidates 100 \
    --use_ema
```

### 3. Run the physics filtering pipeline (Tested with slurm scripts)
```bash
bash run_sam3_stage1.sh
```

Key arguments in `run_sam3_stage1.sh`:
```bash
--video_dir       /path/to/candidate/videos
--output_dir      /path/to/output
--spec_json       '{"subject": ..., "action_phases": ...}'  # or omit to use Stage 0 
--num_candidates  100       # N in Best-of-N
--sam3_fps        3       # sampling fps for SAM3 segmentation
--top_k           5       # keyframes to analyse per video
--use_stage1_filter       # enable VLM artifact detection
```

### 4. Output
The pipeline outputs:
- Selected video path (highest physics score)
- Per-video score breakdown with deductions
- Bounce plots (arc visualisations per window)
- SAM3 segmentation masks
- Full kinematic profile (CoR per bounce, energy dissipation, motion violations)

---

## Physics Filtering Details

### Stage 0 — LLM Grounding (`validate_and_ground_first_frame`) — *optional*
Qwen-VL analyses the first frame to:
- Validate subject identity and initial pose match the prompt
- Extract bounding boxes for subject, surface, background
- Parse action phases (pre-contact, bounce, post-contact)
- Record subject and surface material for CoR estimation

Enable with `run_stage0=True`. When disabled, a JSON spec must be provided via `--spec_json`.

### Stage 1a — Hallucination Filter (`verify_subject_in_video`) — *optional*
Samples frames from the first 20% of the video. Rejects videos where:
- Subject is absent or misidentified
- Initial action phase does not match the prompt

Enable with `qwen_precheck=True`.

### Stage 1b — SAM3 Segmentation + Object Preservation (`detect_contact_with_sam3`) — *active*
Runs SAM3 on each candidate to track the subject centre-of-mass at `sam3_fps`. Filters on:
- Contact distance (subject must reach the surface)
- Colour histogram consistency (artifact/flickering detection, `hist > 2.0`)
- Identity preservation across the full video

### Stage 1c — Physics Scoring (`run_tournament_selection`) — *active*
For each surviving candidate:
1. One LLM call (Qwen3-8B) determines expected CoR range for the material pair
2. Arc detection per sliding window: finds `pb` (pre-bounce peak) → contact → `pa` (post-bounce peak)
3. Central-difference velocity computed; arcs split into descending/ascending phases
4. Monotonicity violations, CoR, energy dissipation, and settlement checked
5. Score computed; highest score selected

Entry point: `MultiObjectiveScorer.run_stage1_filter()` → `run_stage1_until_convergence()` → `multi_gpu_stage1_temporal_validation()`

---

## Acknowledgements

The base autoregressive video diffusion model and training framework are from [Self-Forcing](https://arxiv.org/abs/2506.08009) (Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman — Adobe Research & UT Austin), which is in turn built on [CausVid](https://github.com/tianweiy/CausVid) by Tianwei Yin and [Wan2.1](https://github.com/Wan-Video/Wan2.1). SAM3 is used for video object segmentation.

## Citation

If you use the physics filtering pipeline or evaluation benchmark, please cite:

```bibtex
@mastersthesis{liu2026physeval,
  title={Does Scale Guarantee Physics? A Multi-Model Selection Pipeline for Physically Plausible Video Generation},
  author={Liu, Lulu},
  school={University of Pennsylvania},
  year={2026},
  note={Advisor: Jiatao Gu. Computer and Information Science}
}
```

If you use the base Self-Forcing model, please also cite:

```bibtex
@article{huang2025selfforcing,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={arXiv preprint arXiv:2506.08009},
  year={2025}
}
```
