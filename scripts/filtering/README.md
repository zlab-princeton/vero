# Filtering

Minimal scripts for generating release-facing question-filter and answer-filter rollouts.

Supported domains:
- `chart_ocr`
- `stem`
- `spatial_action`
- `knowledge_recognition`
- `counting_grounding_search`

Question filtering:

```bash
bash scripts/filtering/run_question_filtering.sh \
  --domain chart_ocr \
  --model <model_name_or_path> \
  --data-file <input_jsonl> \
  --output-dir <output_dir>
```

Answer filtering:

```bash
bash scripts/filtering/run_answer_filtering.sh \
  --domain stem \
  --model <model_name_or_path> \
  --data-file <input_jsonl> \
  --output-dir <output_dir>
```

Users must provide their own model, input JSONL, and output directory. The scripts only generate rollout JSONL files.
