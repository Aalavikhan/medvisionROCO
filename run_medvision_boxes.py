# from pathlib import Path
# import re
# import json
# import torch

# from PIL import Image
# from tqdm import tqdm

# from transformers import (
#     AutoProcessor,
#     Qwen2_5_VLForConditionalGeneration,
# )

# from qwen_vl_utils import process_vision_info


# MODEL_ID = "YongchengYAO/MedVision-V0-7B"

# IMAGE_ROOT = Path(
#     r"D:\roco-medsam-test\roco_pilot"
# )

# OUTPUT_ROOT = Path(
#     r"D:\medvision-roco\box_results"
# )

# OUTPUT_ROOT.mkdir(
#     parents=True,
#     exist_ok=True
# )


# SYSTEM_PROMPT = (
#     "A conversation between a User and an Assistant. "
#     "The User asks a question, and the Assistant solves it. "
#     "The Assistant first thinks through the reasoning process internally, "
#     "then provides the User with the answer. "
#     "The reasoning process and the final answer must be enclosed within "
#     "<think> </think> and <answer> </answer> tags, respectively. "
#     "Within the <think> tags, report reasoning using "
#     "<step-k-reasoning> and <step-k-answer> tags."
# )


# QUERIES = {
#     "A": "the main abnormality",
#     "B": "the most clinically relevant abnormal region",
#     "C": "the pathological region in this image",
#     "D": "any visible lesion or abnormality",
# }


# def build_prompt(label: str) -> str:

#     return (
#         "Task:\n"
#         "Given the input medical image, return the coordinates "
#         "of the lower-left and upper-right corners of the bounding "
#         f"box for {label}.\n\n"

#         "Format requirement:\n"
#         "The reasoning process and the final answer must be enclosed "
#         "within <think> </think> and <answer> </answer> tags, respectively. "
#         "The answer should be four decimal numbers separated by commas "
#         "without any units or additional text. "
#         "The first two numbers are the coordinates of the lower-left "
#         "corner and the last two numbers are the coordinates of the "
#         "upper-right corner of the bounding box. "
#         "Use relative coordinates in image space, where the origin "
#         "is at the lower-left corner of the image. "
#         "Relative coordinates must be between 0 and 1.\n\n"

#         "Reasoning steps:\n"
#         "Step 1: Identify the relative coordinates of the bounding box.\n"
#         "Report the reasoning process and final answer using the "
#         "required tags and format."
#     )


# def parse_box(output: str):

#     match = re.search(
#         r"<answer>(.*?)</answer>",
#         output,
#         re.DOTALL,
#     )

#     if not match:
#         return None

#     numbers = re.findall(
#         r"-?\d+(?:\.\d+)?",
#         match.group(1),
#     )

#     if len(numbers) < 4:
#         return None

#     values = [
#         float(x)
#         for x in numbers[-4:]
#     ]

#     x0, y0, x1, y1 = values

#     # Basic validity check
#     if not all(
#         0.0 <= x <= 1.0
#         for x in values
#     ):
#         return None

#     if x1 <= x0 or y1 <= y0:
#         return None

#     return [
#         x0,
#         y0,
#         x1,
#         y1,
#     ]


# def run_query(
#     model,
#     processor,
#     image,
#     label,
# ):

#     question = build_prompt(label)

#     messages = [
#         {
#             "role": "system",
#             "content": SYSTEM_PROMPT,
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": image,
#                 },
#                 {
#                     "type": "text",
#                     "text": question,
#                 },
#             ],
#         },
#     ]

#     text = processor.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )

#     image_inputs, video_inputs = (
#         process_vision_info(messages)
#     )

#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     ).to(model.device)

#     with torch.inference_mode():

#         generated = model.generate(
#             **inputs,
#             max_new_tokens=512,
#             do_sample=False,
#         )

#     trimmed = [
#         output_ids[len(input_ids):]
#         for input_ids, output_ids
#         in zip(
#             inputs.input_ids,
#             generated,
#         )
#     ]

#     output = processor.batch_decode(
#         trimmed,
#         skip_special_tokens=True,
#     )[0]

#     box = parse_box(output)

#     return output, box


# def main():

#     processor = AutoProcessor.from_pretrained(
#         MODEL_ID
#     )

#     model = (
#         Qwen2_5_VLForConditionalGeneration
#         .from_pretrained(
#             MODEL_ID,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#         )
#     )

#     image_paths = []

#     for path in IMAGE_ROOT.rglob("*"):

#         if (
#             path.is_file()
#             and path.suffix.lower()
#             in {
#                 ".png",
#                 ".jpg",
#                 ".jpeg",
#             }
#         ):
#             image_paths.append(path)

#     print(
#         f"Found {len(image_paths)} images."
#     )

#     for image_path in tqdm(
#         image_paths,
#         desc="MedVision",
#     ):

#         image = Image.open(
#             image_path
#         ).convert("RGB")

#         record = {
#             "image": str(image_path),
#             "queries": {},
#         }

#         for query_name, label in QUERIES.items():

#             output, box = run_query(
#                 model,
#                 processor,
#                 image,
#                 label,
#             )

#             record["queries"][query_name] = {
#                 "prompt": label,
#                 "raw_output": output,
#                 "box": box,
#             }

#             print(
#                 f"\n{image_path.name}"
#                 f" [{query_name}]"
#             )

#             print(
#                 "Box:",
#                 box,
#             )

#         relative = image_path.relative_to(
#             IMAGE_ROOT
#         )

#         output_path = (
#             OUTPUT_ROOT
#             / relative.parent
#             / f"{relative.stem}.json"
#         )

#         output_path.parent.mkdir(
#             parents=True,
#             exist_ok=True,
#         )

#         with open(
#             output_path,
#             "w",
#             encoding="utf-8",
#         ) as f:

#             json.dump(
#                 record,
#                 f,
#                 indent=2,
#             )


# if __name__ == "__main__":
#     main()






















from pathlib import Path
import re
import json

import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "YongchengYAO/MedVision-V0-7B"

IMAGE_ROOT = Path(
    r"D:\roco-medsam-test\roco_pilot"
)

OUTPUT_ROOT = Path(
    r"D:\medvision-roco\box_results"
)

# Start with 5 images.
# Change to None later to process all images.
MAX_IMAGES = 5

# One generic prompt only.
QUERY_LABEL = (
    "the most clinically relevant abnormal region"
)


# ============================================================
# Setup
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


SYSTEM_PROMPT = (
    "A conversation between a User and an Assistant. "
    "The Assistant answers the User's question. "
    "Return the final answer inside "
    "<answer> </answer> tags."
)


# ============================================================
# Prompt
# ============================================================

def build_prompt() -> str:
    return (
        "Task:\n"
        "Given the input medical image, identify and localize "
        "the most clinically relevant abnormal region.\n\n"

        "Return one bounding box around that region.\n\n"

        "Format requirement:\n"
        "Return the final answer inside <answer> </answer> tags. "
        "The answer must contain exactly four decimal numbers "
        "separated by commas.\n\n"

        "The four numbers represent:\n"
        "x0, y0, x1, y1\n\n"

        "Use relative coordinates between 0 and 1.\n"
        "The coordinate origin is the lower-left corner "
        "of the image.\n\n"

        "x0, y0 = lower-left corner\n"
        "x1, y1 = upper-right corner\n\n"

        "Example format:\n"
        "<answer>0.20,0.30,0.70,0.80</answer>"
    )


# ============================================================
# Parse bounding box
# ============================================================

def parse_box(output: str):
    match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        output,
        re.DOTALL | re.IGNORECASE,
    )

    if not match:
        return None

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        match.group(1),
    )

    if len(numbers) < 4:
        return None

    values = [
        float(x)
        for x in numbers[-4:]
    ]

    x0, y0, x1, y1 = values

    # Coordinates must be within [0, 1]
    if not all(
        0.0 <= value <= 1.0
        for value in values
    ):
        return None

    # Box must have positive width and height
    if x1 <= x0 or y1 <= y0:
        return None

    return [
        x0,
        y0,
        x1,
        y1,
    ]


# ============================================================
# Run MedVision on one image
# ============================================================

def run_query(
    model,
    processor,
    image,
):
    question = build_prompt()

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
                    "text": question,
                },
            ],
        },
    ]

    # Convert conversation to model text
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Extract image inputs
    image_inputs, video_inputs = process_vision_info(
        messages
    )

    # Prepare tensors
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move tensors to the same device used by the model
    if torch.cuda.is_available():
        inputs = {
            key: value.to("cuda")
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

    # Generate
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    # Remove prompt tokens from generated sequence
    input_ids = inputs["input_ids"]

    trimmed = [
        output_ids[len(input_ids_single):]
        for input_ids_single, output_ids
        in zip(
            input_ids,
            generated,
        )
    ]

    # Decode
    output = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
    )[0]

    # Parse bounding box
    box = parse_box(output)

    return output, box


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("MedVision-V0 ROCO bounding-box test")
    print("=" * 60)

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

    print(
        "CUDA:",
        torch.cuda.is_available(),
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_paths = []

    for path in IMAGE_ROOT.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
            }
        ):
            image_paths.append(path)

    image_paths.sort()

    print(
        f"\nFound {len(image_paths)} images."
    )

    # Limit for pilot test
    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    print(
        f"Processing {len(image_paths)} images."
    )

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for image_path in tqdm(
        image_paths,
        desc="MedVision",
    ):

        try:
            print(
                f"\nProcessing: {image_path.name}"
            )

            # Load image
            image = Image.open(
                image_path
            ).convert("RGB")

            # Run one generation
            output, box = run_query(
                model,
                processor,
                image,
            )

            print(
                "Box:",
                box,
            )

            if box is None:
                print(
                    "WARNING: Could not parse a valid box."
                )

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            relative = image_path.relative_to(
                IMAGE_ROOT
            )

            output_path = (
                OUTPUT_ROOT
                / relative.parent
                / f"{relative.stem}.json"
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            record = {
                "image": str(image_path),
                "prompt": QUERY_LABEL,
                "raw_output": output,
                "box": box,
            }

            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    record,
                    f,
                    indent=2,
                )

            print(
                "Saved:",
                output_path,
            )

        except Exception as e:

            print(
                f"ERROR processing {image_path.name}:"
            )

            print(
                repr(e)
            )

    print("\nDone.")


if __name__ == "__main__":
    main()






















