# Vero Model Cards

## Model Zoo

| Model | Base Model | Parameters | HuggingFace |
|-------|-----------|------------|-------------|
| **Vero-Qwen25-7B** | Qwen2.5-VL-7B-Instruct | 7B | [zlab-princeton/Vero-Qwen25-7B](https://huggingface.co/zlab-princeton/Vero-Qwen25-7B) |
| **Vero-Qwen3I-8B** | Qwen3-VL-8B-Instruct | 8B | [zlab-princeton/Vero-Qwen3I-8B](https://huggingface.co/zlab-princeton/Vero-Qwen3I-8B) |
| **Vero-Qwen3T-8B** | Qwen3-VL-8B-Thinking | 8B | [zlab-princeton/Vero-Qwen3T-8B](https://huggingface.co/zlab-princeton/Vero-Qwen3T-8B) |
| **Vero-MiMo-7B** | MiMo-VL-7B-SFT | 7B | [zlab-princeton/Vero-MiMo-7B](https://huggingface.co/zlab-princeton/Vero-MiMo-7B) |

## Training Details

All models are trained with the same recipe:
- **Algorithm**: GSPO (Group Relative Policy Optimization)
- **Training data**: 600K samples, uniform sampling across 6 task categories
- **Training steps**: 4,000
- **Hardware**: 8x A100/H100 GPUs
- **Learning rate**: 1e-6 with 40-step warmup

## Inference Example

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "zlab-princeton/Vero-Qwen3I-8B"
model = AutoModelForVision2Seq.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=16384,
    temperature=0.6,
    top_p=1.0,
    do_sample=True,
)
output_text = processor.batch_decode(
    generated_ids[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True,
)[0]
print(output_text)
```

## Inference with vLLM

For faster inference with vLLM:

```bash
vllm serve zlab-princeton/Vero-Qwen3I-8B \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9
```

Then query via the OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="zlab-princeton/Vero-Qwen3I-8B",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        }
    ],
    max_tokens=16384,
    temperature=0.6,
    top_p=1.0,
)
print(response.choices[0].message.content)
```

## Output Format

Vero models produce chain-of-thought reasoning in `<think>` tags followed by a final answer in `<answer>` tags:

```
<think>
The image shows a bar chart comparing quarterly revenue...
Looking at Q3, the value appears to be approximately $4.2M...
</think>
<answer>
$4.2 million
</answer>
```

## License

All Vero models are released under the [MIT License](../LICENSE).
