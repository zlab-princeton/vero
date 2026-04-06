"""
Vero model inference example.

Usage:
    python examples/inference.py \
        --model-path zlab-princeton/Vero-Qwen3I-8B \
        --image path/to/image.jpg \
        --prompt "What is shown in this image?"

    # With a local checkpoint
    python examples/inference.py \
        --model-path /path/to/checkpoint \
        --image path/to/chart.png \
        --prompt "What is the value for Q3 revenue?"

    # Greedy decoding
    python examples/inference.py \
        --model-path zlab-princeton/Vero-Qwen3I-8B \
        --image path/to/image.jpg \
        --prompt "Describe this chart." \
        --temperature 0
"""

import argparse

from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info


def main():
    parser = argparse.ArgumentParser(description="Run inference with a Vero model")
    parser.add_argument(
        "--model-path",
        default="zlab-princeton/Vero-Qwen3I-8B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Question or instruction")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generate_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )

    generated_ids = model.generate(**inputs, **generate_kwargs)
    output = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )[0]

    print(output)


if __name__ == "__main__":
    main()
