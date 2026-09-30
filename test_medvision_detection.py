from pathlib import Path

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "YongchengYAO/MedVision-V0-7B"

IMAGE_PATH = Path(
    r"D:\roco-medsam-test\roco_pilot\Angiography\ROCOv2_2023_train_000024.png"
)


# ============================================================
# MedVision official system prompt
# ============================================================

SYSTEM_PROMPT = (
    "A conversation between a User and an Assistant. "
    "The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks through the reasoning process internally, "
    "then provides the User with the answer. "
    "The reasoning process and the final answer must be enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively. "
    "For example: <think> reasoning process here </think> "
    "<answer> answer here </answer>. "
    "Within the <think> </think> tags, report the reasoning process "
    "for each step inside <step-k-reasoning> </step-k-reasoning> tags, "
    "followed by the intermediate results in <step-k-answer> "
    "</step-k-answer> tags. "
    "For example: <think> "
    "<step-1-reasoning> reasoning for step 1 </step-1-reasoning> "
    "<step-1-answer> intermediate result from step 1 "
    "</step-1-answer> </think>."
)


# ============================================================
# Official MedVision detection prompt
# ============================================================

QUESTION = (
    "Task:\n"
    "Given the input medical image, return the coordinates of the "
    "lower-left and upper-right corners of the bounding box for "
    "the most clinically relevant abnormal region.\n"

    "Format requirement:\n"
    "The reasoning process and the final answer must be enclosed "
    "within <think> </think> and <answer> </answer> tags, respectively. "
    "For example: <think> reasoning process here </think> "
    "<answer> answer here </answer>. "
    "The answer should be four decimal numbers separated by commas "
    "without any units or additional text. "
    "The first two numbers are the coordinates of the lower-left "
    "corner and the last two numbers are the coordinates of the "
    "upper-right corner of the bounding box. "
    "Use relative coordinates in the image space, where the origin "
    "is at the lower-left corner of the image. "
    "Relative coordinates should be values between 0 and 1, "
    "representing the relative positions in the image.\n"

    "Reasoning steps:\n"
    "Step 1: Identify the relative coordinates of the bounding box. "
    "The relative coordinates must be written as (x, y), where x "
    "is the relative position in width and y is the relative "
    "position in height. "
    "Report the reasoning process and final answer within "
    "<think> </think> and <answer> </answer> tags, respectively. "
    "Inside <think> </think>, include reasoning and step results "
    "using <step-k-reasoning> </step-k-reasoning> and "
    "<step-k-answer> </step-k-answer> tags.\n"

    "Follow the reasoning steps to get the final answer in the "
    "required format."
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MedVision-V0 one-image detection diagnostic")
    print("=" * 70)

    print("\nCUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    print("Loading model...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    print("Model loaded.")

    print("\nLoading image:")

    print(IMAGE_PATH)

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print(
        "Image size:",
        image.size,
    )

    # --------------------------------------------------------
    # Construct exact MedVision-style conversation
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": QUESTION,
                },
            ],
        },
    ]

    # --------------------------------------------------------
    # Process inputs
    # --------------------------------------------------------

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # The model is using device_map="auto".
    # Put input tensors on the model's main device.
    inputs = inputs.to(model.device)

    print("\nGenerating...")

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.inference_mode():

        generated = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    # Remove input tokens
    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated,
        )
    ]

    # Decode WITHOUT parsing anything
    output = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
    )[0]

    # --------------------------------------------------------
    # Print complete raw output
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FULL RAW MODEL OUTPUT")
    print("=" * 70)

    print(output)

    print("=" * 70)
    print("END OUTPUT")
    print("=" * 70)


if __name__ == "__main__":
    main()