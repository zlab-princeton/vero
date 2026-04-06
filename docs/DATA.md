# Vero Training Data

## Overview

Vero is trained on **600K curated RL samples** from **59 datasets** spanning six core visual task categories. The data is formatted in the veRL JSONL format for use with the GSPO training pipeline.

**Download**: [zlab-princeton/Vero-600k](https://huggingface.co/datasets/zlab-princeton/Vero-600k)

## Task Categories

### 1. Chart & OCR (9 datasets)
Extracting and reasoning over structured information in documents, charts, tables, and infographics.

- chartqa_difficulty
- CoSyn_400k_chart
- CoSyn_400k_table
- CoSyn_400k_diagram
- arxivqa_formatted
- infographic_vqa
- evochart
- ecd_vqa
- reachqa

### 2. Grounding, Counting & Visual Search (11 datasets)
Spatially localizing objects via bounding boxes, counting entities, and searching among distractors.

- tallyqa
- pixmo
- objects365
- refcocog
- aerialvg
- groundui
- visual_probe
- pixelreasoner
- oodvqa
- multihop_qa
- osatlas

### 3. STEM (13 datasets)
Mathematical diagram reasoning, scientific figure interpretation, and medical image understanding. Answers are typically numeric or symbolic.

- visualwebinstruct
- CoSyn_400k_math
- geomverse
- mavis_math_metagen
- mavis_math_rule_geo
- raven
- ai2d_merged
- CoSyn_400k_chemical

### 4. Spatial & Action (8 datasets)
Embodied reasoning, UI navigation, and 3D spatial understanding. Requires reasoning about spatial transformations and action sequences.

- game_QA
- visual_jigsaw_3d
- visual_jigsaw_2d
- magma_mind2web
- magma_aitw
- spatial_ssrl
- stvqa
- robo2vlm

### 5. Knowledge & Recognition (12 datasets)
Visual QA combining object, scene, and entity recognition with external or commonsense knowledge.

- iconqa
- indoor_qa
- visual7w
- vizwiz
- vqav2
- aokvqa
- gqa
- viquae
- vcrqa
- kvqa

### 6. Captioning & Instruction Following (6 datasets)
Open-ended image description and prompt instruction following. These samples align response style, instruction adherence, and open-ended generation quality. Question filtering is not applied to this category.

- PixMo-AskAnything
- PixMo-CapQA
- PixMo-Cap
- MM-RLVR-IFEval
- MMIF-23K
- Flickr30K

## Data Curation Pipeline

Starting from 250+ candidate datasets, we apply three stages of curation. The filtering scripts are released in [`scripts/filtering/`](../scripts/filtering/).

### Step 1: Dataset Curation
- **Heuristic filtering**: Discard datasets with <1K examples, <200K pixels average resolution, or exclusively binary questions
- **Manual filtering**: Inspect ~50 examples per dataset against three criteria:
  - Correctness (<5% annotation error rate)
  - Unambiguity (single verifiable answer)
  - Verifiability (compatible with reward functions)
- **Result**: 59 datasets retained from ~100 that passed heuristic screening

### Step 2: Question Filtering

Score each sample using Qwen3-VL-235B on relevance, ambiguity, language quality, verifiability, and numeric precision. Domain-specific prompts are provided for each category:

| Domain | Prompt |
|--------|--------|
| Chart & OCR | [`question_filter_chart_ocr.txt`](../scripts/filtering/prompts/question_filter_chart_ocr.txt) |
| STEM | [`question_filter_stem.txt`](../scripts/filtering/prompts/question_filter_stem.txt) |
| Spatial & Action | [`question_filter_spatial_action.txt`](../scripts/filtering/prompts/question_filter_spatial_action.txt) |
| Knowledge & Recognition | [`question_filter_knowledge_recognition.txt`](../scripts/filtering/prompts/question_filter_knowledge_recognition.txt) |
| Grounding, Counting & Search | [`question_filter_counting_grounding_search.txt`](../scripts/filtering/prompts/question_filter_counting_grounding_search.txt) |

To run question filtering:
```bash
bash scripts/filtering/run_question_filtering.sh \
  --domain chart_ocr \
  --model <model_name_or_path> \
  --data-file <input_jsonl> \
  --output-dir <output_dir>
```

See [`scripts/filtering/generate_question_filter_rollouts.py`](../scripts/filtering/generate_question_filter_rollouts.py) for the generation script.

### Step 3: Answer Canonicalization

Normalize ground-truth answers using a text-only LLM for stable reward computation:
- Numeric: strip units/currency, convert to decimal, evaluate expressions
- Multiple-choice: normalize to single canonical letter
- Exclude multi-value or ambiguous descriptions (except captioning/instruction-following)

Domain-specific prompts:

| Domain | Prompt |
|--------|--------|
| Default | [`answer_filter_default.txt`](../scripts/filtering/prompts/answer_filter_default.txt) |
| Knowledge & Recognition | [`answer_filter_knowledge_recognition.txt`](../scripts/filtering/prompts/answer_filter_knowledge_recognition.txt) |

To run answer filtering:
```bash
bash scripts/filtering/run_answer_filtering.sh \
  --domain stem \
  --model <model_name_or_path> \
  --data-file <input_jsonl> \
  --output-dir <output_dir>
```

See [`scripts/filtering/generate_answer_filter_rollouts.py`](../scripts/filtering/generate_answer_filter_rollouts.py) for the generation script.

### Step 4: Data Mixtures
We tested four weighting schemes (uniform, difficulty-weighted, dataset-size-weighted, reasoning-length-weighted). **Uniform sampling** across the 6 categories achieved the best benchmark average (+5.8 pts), outperforming alternatives.

## Data Format

Each training sample is a JSON line with the following fields:

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": [
      {"type": "image", "image": "path/to/image.jpg"},
      {"type": "text", "text": "What is shown in this chart?"}
    ]}
  ],
  "images": ["path/to/image.jpg"],
  "reward_type": "exact_match",
  "data_source": "chartqa",
  "domain": "chart_ocr",
  "answer": "42"
}
```

### Fields

| Field | Description |
|-------|-------------|
| `prompt` | Chat-formatted prompt with system + user messages |
| `images` | List of image paths referenced in the prompt |
| `reward_type` | Reward function to use — routes to the corresponding verifier in [`vero_reward/`](../vero-rl/vero_reward/) |
| `data_source` | Name of the source dataset |
| `domain` | Task category (one of the 6 categories) |
| `answer` | Canonicalized ground-truth answer |

### Reward Types

Each `reward_type` is routed to a verifier in the [`vero_reward/`](../vero-rl/vero_reward/) package:

| Reward Type | Verifier | Description |
|-------------|----------|-------------|
| `exact_match` | [`string_match.py`](../vero-rl/vero_reward/string_match.py) | Exact string match |
| `multiple_choice` | [`math_verify_reward_type_boxed.py`](../vero-rl/vero_reward/math_verify_reward_type_boxed.py) | Normalized letter matching |
| `numeric` | [`math_verify_reward_type_boxed.py`](../vero-rl/vero_reward/math_verify_reward_type_boxed.py) | Numeric comparison with tolerance |
| `list_match` | [`math_verify_reward_type_boxed.py`](../vero-rl/vero_reward/math_verify_reward_type_boxed.py) | Unordered list matching |
| `ordering` | [`math_verify_reward_type_boxed.py`](../vero-rl/vero_reward/math_verify_reward_type_boxed.py) | Ordered sequence matching |
| `web_action` | [`math_verify_reward_type_boxed.py`](../vero-rl/vero_reward/math_verify_reward_type_boxed.py) | Web navigation action matching |
| `grounding` | [`grounding_reward.py`](../vero-rl/vero_reward/grounding_reward.py) | Bounding box IoU matching |
| `clicking` | [`click_reward.py`](../vero-rl/vero_reward/click_reward.py) | Click coordinate distance |
| `instruction_following` | [`instructions.py`](../vero-rl/vero_reward/instructions.py) | Instruction adherence check |
| `llm_judge` | [`vero_vllm_judge.py`](../vero-rl/verl/workers/reward_manager/vero_vllm_judge.py) | LLM-based evaluation scoring |
