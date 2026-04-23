# NuScenes-CoC: Replicating the Alpamayo-R1 Chain-of-Context Dataset

A pipeline that reproduces the **Chain-of-Context (CoC)** annotation methodology from NVIDIA's [Alpamayo-R1](https://arxiv.org/abs/2505.03047) paper using the public [nuScenes](https://www.nuscenes.org/) dataset.

---

## What This Project Does

The Alpamayo-R1 paper introduces a driving dataset where each clip is annotated with:
- A **driving decision** (longitudinal × lateral action pair)
- A **chain-of-thought (CoT)** sentence explaining *why* the ego vehicle takes that action
- **Critical components** that causally triggered the decision

This project replicates that pipeline end-to-end on nuScenes:

```
nuScenes raw data
      ↓  [segment extraction + rule-based decision labeling]
Candidate CoC samples  (JSON)
      ↓  [VLM teacher labeling via qwen-vl-max]
CoC dataset with English CoT  (JSON)
      ↓  [format export]
ood_reasoning.parquet  (official Alpamayo-R1 format)
```

### Completed Features

| Component | Status | Description |
|-----------|--------|-------------|
| Segment extraction | ✅ | Sliding-window candidate segment detection from nuScenes samples |
| Rule-based decision labeling | ✅ | 6 longitudinal × 4 lateral decision classes derived from ego kinematics, map topology, and surrounding agents |
| Low-signal filtering | ✅ | Drops trivially boring segments (set_speed_tracking with no causal trigger within 18 m) |
| VLM teacher labeling | ✅ | Calls qwen-vl-max (or any OpenAI-compatible VLM) with CAM_FRONT image + structured scene context |
| English CoT generation | ✅ | Produces action-focused English sentences aligned with official Alpamayo-R1 style |
| Official format export | ✅ | Exports `ood_reasoning.parquet` with `clip_id / feature / event_cluster / events / split` schema |
| Visual evaluation report | ✅ | Self-contained HTML report for human annotation review with rating buttons |

---

## Repository Structure

```
Coc/
├── nuscenes_coc/               # Core library
│   ├── cli.py                  # Main pipeline orchestration (build_dataset)
│   ├── segment_filter.py       # Candidate segment detection & filtering
│   ├── decision_rules.py       # Rule-based longitudinal/lateral decision logic
│   ├── component_extractor.py  # Extract critical scene components
│   ├── teacher_prompt.py       # VLM system/user prompt & JSON schema
│   ├── geometry.py             # Ego-relative coordinate transforms
│   ├── motion.py               # Velocity / acceleration helpers
│   ├── nusc_access.py          # nuScenes API wrappers
│   ├── meta_actions.py         # Action taxonomy definitions
│   ├── exporter.py             # JSON serialization helpers
│   ├── quality.py              # Post-hoc quality checks
│   └── text_templates.py       # Fallback text templates
│
├── scripts/
│   ├── generate_nuscenes_coc.py        # Step 1: extract segments → CoC JSON
│   ├── build_teacher_labeling_assets.py # Step 2: build VLM request JSONL
│   ├── run_teacher_llm_labeling.py     # Step 3: call VLM API, write responses
│   ├── export_to_official_format.py    # Step 4: export to parquet
│   └── generate_eval_report.py         # Optional: HTML evaluation report
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n coc_nuscenes python=3.10 -y
conda activate coc_nuscenes
pip install -r requirements.txt
pip install pandas pyarrow openai
```

> **Windows note:** nuScenes map loading requires the expansion maps to be accessible at `maps/expansion/`. If you encounter `FileNotFoundError`, create a directory junction:
> ```powershell
> # Run as Administrator
> cmd /c mklink /J "path\to\nuscenes\maps\expansion" "path\to\nuscenes\maps\nuScenes-map-expansion-v1.3\expansion"
> ```

### 2. Download nuScenes Data

Download [nuScenes mini](https://www.nuscenes.org/nuscenes#download) (or the full trainval split) and extract to a local directory, e.g.:
```
nuscenes-mini/
  v1.0-mini/
  samples/
  sweeps/
  maps/
```

### 3. Run the Full Pipeline

```bash
DATA_ROOT="path/to/nuscenes-mini"
cd Coc

# Step 1 – Extract CoC segments
python scripts/generate_nuscenes_coc.py \
    --data-root "$DATA_ROOT" \
    --version v1.0-mini \
    --output out/nuscenes_coc.json \
    --stats-output out/stats.json

# Step 2 – Build VLM labeling requests
python scripts/build_teacher_labeling_assets.py \
    --input out/nuscenes_coc.json \
    --output-dir out

# Step 3 – Call VLM (requires an OpenAI-compatible API key)
python scripts/run_teacher_llm_labeling.py \
    --requests-input out/teacher_requests.jsonl \
    --responses-output out/teacher_responses.jsonl \
    --model qwen-vl-max \
    --base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --api-key "YOUR_API_KEY" \
    --with-images \
    --cameras CAM_FRONT \
    --image-frames 0

# Step 4 – Merge responses into final dataset
python scripts/build_teacher_labeling_assets.py \
    --input out/nuscenes_coc.json \
    --output-dir out \
    --responses out/teacher_responses.jsonl \
    --final-output out/nuscenes_coc_final.json

# Step 5 – Export to official parquet format
python scripts/export_to_official_format.py \
    --input out/nuscenes_coc_final.json \
    --output-dir out/official \
    --auto-split
```

The final output `out/official/reasoning/ood_reasoning.parquet` matches the Alpamayo-R1 `ood_reasoning.parquet` schema.

---

## Output Format

The exported parquet file follows the official Alpamayo-R1 schema:

| Column | Type | Description |
|--------|------|-------------|
| `clip_id` | string (index) | Unique clip identifier |
| `feature` | string | Camera modality (`camera_front_wide_120fov`) |
| `event_cluster` | string | High-level event category (e.g. `PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY`) |
| `events` | dict | `{event_start_frame, event_start_timestamp, cot}` — the CoT annotation |
| `split` | string | `train` or `val` |

### Event Cluster Mapping

| Decision | Event Cluster |
|----------|---------------|
| `lead_obstacle_following` | `SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR` |
| `speed_adaptation_road` | `PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY` or `WORK_ZONES_TEMP_TRAFFIC_CONTROL` |
| `stop_static_constraint` | `WORK_ZONES_TEMP_TRAFFIC_CONTROL` |
| `yield_agent_right_of_way` | `PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY` |
| `maintain_speed` + turn lane | `COMPLEX_INTERSECTION_INTERACTION` |
| others | `SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR` |

---

## Human Evaluation

Generate a self-contained HTML report for manual annotation review:

```bash
python scripts/generate_eval_report.py \
    --input out/nuscenes_coc_final.json \
    --output out/eval_report.html
```

Open `eval_report.html` in any browser. Each sample card shows:
- CAM_FRONT images (history frames → keyframe ★)
- Decision badges, VLM CoT, rule-based preliminary annotation
- Critical components table
- **✓ Correct / ~ Partial / ✗ Wrong** rating buttons (auto-saved in browser localStorage)
- Export ratings as JSON

---

## VLM Compatibility

The teacher labeling step (`run_teacher_llm_labeling.py`) works with any **OpenAI-compatible** API:

| Provider | Model | `--base-url` |
|----------|-------|-------------|
| Alibaba DashScope | `qwen-vl-max` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| OpenAI | `gpt-4o` | *(default)* |
| Local (Ollama) | `llava`, etc. | `http://localhost:11434/v1` |

---

## Driving Decision Taxonomy

### Longitudinal (6 classes)

| Class | Meaning |
|-------|---------|
| `lead_obstacle_following` | Follow lead vehicle at safe distance |
| `speed_adaptation_road` | Adapt speed to road/environment conditions |
| `stop_static_constraint` | Stop for stop line / traffic control |
| `yield_agent_right_of_way` | Yield to pedestrian or crossing agent |
| `maintain_speed` | Cruise at current speed |
| `set_speed_tracking` | Track set speed (no causal trigger — filtered if trivial) |

### Lateral (4 classes)

| Class | Meaning |
|-------|---------|
| `lane_keeping_centering` | Stay centered in lane |
| `in_lane_nudge_left` | Small leftward adjustment within lane |
| `in_lane_nudge_right` | Small rightward adjustment within lane |
| `lane_change_left/right` | Full lane change |

---

## Limitations vs Official Alpamayo-R1

| Aspect | Official | This Repo |
|--------|----------|-----------|
| Dataset scale | ~1,740 clips | 14 (mini) / ~1,000 (trainval, not tested) |
| CoT quality | Human-verified | VLM-generated (qwen-vl-max) |
| `event_start_timestamp` | Relative ms from clip start | Absolute nuScenes microsecond timestamp |
| Data source | Proprietary internal data | Public nuScenes |

---

## References

- [Alpamayo-R1 paper (arXiv)](https://arxiv.org/abs/2505.03047)
- [nuScenes dataset](https://www.nuscenes.org/)
- [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)
