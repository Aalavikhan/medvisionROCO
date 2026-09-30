# # from pathlib import Path
# import re
# import json
# import sys

# import numpy as np
# import torch
# from PIL import Image, ImageDraw
# from tqdm import tqdm
# from pathlib import Path
# from transformers import (
#     AutoProcessor,
#     Qwen2_5_VLForConditionalGeneration,
# )

# from qwen_vl_utils import process_vision_info


# # ============================================================
# # Paths
# # ============================================================

# MODEL_ID = "YongchengYAO/MedVision-V0-7B"

# IMAGE_ROOT = Path(
#     r"D:\roco-medsam-test\roco_pilot"
# )

# # Separate output directory for the visible-cue experiment
# OUTPUT_ROOT = Path(
#     r"D:\medvision-roco\medsam_results_visible_cues"
# )

# MEDSAM_ROOT = Path(
#     r"D:\roco-medsam-test\MedSAM"
# )

# MEDSAM_CHECKPOINT = Path(
#     r"D:\roco-medsam-test\checkpoints\medsam_vit_b.pth"
# )


# # ============================================================
# # Process ALL images across ALL modality folders
# # ============================================================

# MAX_IMAGES = None


# # ============================================================
# # MedSAM source
# # ============================================================

# sys.path.insert(
#     0,
#     str(MEDSAM_ROOT)
# )

# from segment_anything import (
#     sam_model_registry,
#     SamPredictor,
# )


# # ============================================================
# # Output
# # ============================================================

# OUTPUT_ROOT.mkdir(
#     parents=True,
#     exist_ok=True,
# )


# # ============================================================
# # MedVision system prompt
# # ============================================================

# SYSTEM_PROMPT = (
#     "A conversation between a User and an Assistant. "
#     "The User asks a question, and the Assistant solves it. "
#     "The Assistant first thinks through the reasoning process internally, "
#     "then provides the User with the answer. "
#     "The reasoning process and the final answer must be enclosed within "
#     "<think> </think> and <answer> </answer> tags, respectively. "
#     "For example: <think> reasoning process here </think> "
#     "<answer> answer here </answer>. "
#     "Within the <think> </think> tags, report the reasoning process "
#     "for each step inside <step-k-reasoning> </step-k-reasoning> tags, "
#     "followed by the intermediate results in <step-k-answer> "
#     "</step-k-answer> tags. "
#     "For example: <think> "
#     "<step-1-reasoning> reasoning for step 1 </step-1-reasoning> "
#     "<step-1-answer> intermediate result from step 1 "
#     "</step-1-answer> </think>."
# )


# # ============================================================
# # Visible-cue localization prompt
# # ============================================================

# QUESTION = (
#     "Task:\n"
#     "Given the input medical image, identify and localize the "
#     "most clinically relevant abnormal region.\n\n"

#     "Important visual cue:\n"
#     "The image may contain visual annotations that are genuinely "
#     "part of the medical image, such as arrows, circles, pointers, "
#     "markers, labels, or other indicators that draw attention to "
#     "an abnormal finding. Use these visible cues as evidence when "
#     "determining the location of the abnormality.\n\n"

#     "If an arrow, circle, pointer, marker, or similar annotation "
#     "points toward an abnormality, localize the underlying "
#     "abnormal tissue, lesion, vessel, structure, or finding that "
#     "the annotation indicates.\n\n"

#     "Do NOT return a bounding box around the arrow, circle, label, "
#     "or annotation itself. Return the bounding box of the actual "
#     "underlying abnormal region being indicated.\n\n"

#     "If no useful visual annotation is present, rely on the "
#     "medical image itself to identify the most clinically relevant "
#     "abnormal region.\n\n"

#     "Format requirement:\n"
#     "The reasoning process and the final answer must be enclosed "
#     "within <think> </think> and <answer> </answer> tags, respectively. "
#     "For example: <think> reasoning process here </think> "
#     "<answer> answer here </answer>. "
#     "The answer should be four decimal numbers separated by commas "
#     "without any units or additional text. "
#     "The first two numbers are the coordinates of the lower-left "
#     "corner and the last two numbers are the coordinates of the "
#     "upper-right corner of the bounding box. "
#     "Use relative coordinates in the image space, where the origin "
#     "is at the lower-left corner of the image. "
#     "Relative coordinates should be values between 0 and 1, "
#     "representing the relative positions in the image.\n\n"

#     "Reasoning steps:\n"
#     "Step 1: Identify the clinically relevant abnormality and use "
#     "any visible arrows, circles, pointers, markers, labels, or "
#     "other embedded visual cues to determine its location.\n"
#     "Step 2: Identify the relative coordinates of the bounding box. "
#     "The relative coordinates must be written as (x, y), where x "
#     "is the relative position in width and y is the relative "
#     "position in height.\n"
#     "Report the reasoning process and final answer within "
#     "<think> </think> and <answer> </answer> tags, respectively.\n\n"

#     "Follow the reasoning steps to get the final answer in the "
#     "required format."
# )


# # ============================================================
# # Parse MedVision output
# # ============================================================

# def parse_box(output: str):

#     match = re.search(
#         r"<answer>\s*(.*?)\s*</answer>",
#         output,
#         re.DOTALL | re.IGNORECASE,
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


# # ============================================================
# # MedVision coordinates -> PIL/OpenCV pixel coordinates
# #
# # MedVision:
# #   origin = lower-left
# #
# # PIL/OpenCV:
# #   origin = upper-left
# # ============================================================

# def normalized_box_to_pixels(
#     box,
#     image_width,
#     image_height,
# ):

#     x0, y0, x1, y1 = box

#     pixel_x0 = round(
#         x0 * image_width
#     )

#     pixel_x1 = round(
#         x1 * image_width
#     )

#     pixel_y0 = round(
#         (1.0 - y1) * image_height
#     )

#     pixel_y1 = round(
#         (1.0 - y0) * image_height
#     )

#     pixel_x0 = max(
#         0,
#         min(
#             pixel_x0,
#             image_width - 1,
#         ),
#     )

#     pixel_x1 = max(
#         0,
#         min(
#             pixel_x1,
#             image_width - 1,
#         ),
#     )

#     pixel_y0 = max(
#         0,
#         min(
#             pixel_y0,
#             image_height - 1,
#         ),
#     )

#     pixel_y1 = max(
#         0,
#         min(
#             pixel_y1,
#             image_height - 1,
#         ),
#     )

#     if pixel_x1 <= pixel_x0:
#         return None

#     if pixel_y1 <= pixel_y0:
#         return None

#     return [
#         pixel_x0,
#         pixel_y0,
#         pixel_x1,
#         pixel_y1,
#     ]


# # ============================================================
# # MedVision inference
# # ============================================================

# def run_medvision(
#     model,
#     processor,
#     image,
# ):

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
#                     "text": QUESTION,
#                 },
#             ],
#         },
#     ]

#     text = processor.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )

#     image_inputs, video_inputs = process_vision_info(
#         messages
#     )

#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )

#     inputs = inputs.to(
#         model.device
#     )

#     with torch.inference_mode():

#         generated = model.generate(
#             **inputs,
#             max_new_tokens=1024,
#             do_sample=False,
#         )

#     trimmed = [
#         output_ids[len(input_ids):]
#         for input_ids, output_ids in zip(
#             inputs.input_ids,
#             generated,
#         )
#     ]

#     output = processor.batch_decode(
#         trimmed,
#         skip_special_tokens=True,
#     )[0]

#     box = parse_box(
#         output
#     )

#     return output, box


# # ============================================================
# # MedSAM inference
# # ============================================================

# def run_medsam(
#     predictor,
#     image,
#     pixel_box,
# ):

#     image_np = np.array(
#         image.convert("RGB")
#     )

#     predictor.set_image(
#         image_np
#     )

#     box_np = np.array(
#         pixel_box,
#         dtype=np.float32,
#     )

#     masks, scores, logits = predictor.predict(
#         point_coords=None,
#         point_labels=None,
#         box=box_np,
#         multimask_output=False,
#     )

#     mask = masks[0]

#     score = float(
#         scores[0]
#     )

#     return mask, score


# # ============================================================
# # Save visualization
# # ============================================================

# def save_overlay(
#     image,
#     pixel_box,
#     mask,
#     output_path,
# ):

#     base = image.convert(
#         "RGBA"
#     )

#     overlay_array = np.zeros(
#         (
#             image.height,
#             image.width,
#             4,
#         ),
#         dtype=np.uint8,
#     )

#     overlay_array[mask, 0] = 255
#     overlay_array[mask, 1] = 0
#     overlay_array[mask, 2] = 0
#     overlay_array[mask, 3] = 90

#     overlay = Image.fromarray(
#         overlay_array,
#         mode="RGBA",
#     )

#     result = Image.alpha_composite(
#         base,
#         overlay,
#     )

#     draw = ImageDraw.Draw(
#         result
#     )

#     x0, y0, x1, y1 = pixel_box

#     draw.rectangle(
#         [x0, y0, x1, y1],
#         outline=(255, 255, 0, 255),
#         width=3,
#     )

#     result.save(
#         output_path
#     )


# # ============================================================
# # Main
# # ============================================================

# def main():

#     print("=" * 70)
#     print(
#         "MedVision -> MedSAM "
#         "VISIBLE-CUE EXPERIMENT"
#     )
#     print("=" * 70)

#     # --------------------------------------------------------
#     # CUDA
#     # --------------------------------------------------------

#     if not torch.cuda.is_available():

#         raise RuntimeError(
#             "CUDA is not available."
#         )

#     print(
#         "\nGPU:",
#         torch.cuda.get_device_name(0),
#     )

#     print(
#         "CUDA:",
#         torch.version.cuda,
#     )

#     # --------------------------------------------------------
#     # Load MedVision
#     # --------------------------------------------------------

#     print(
#         "\nLoading MedVision processor..."
#     )

#     processor = AutoProcessor.from_pretrained(
#         MODEL_ID
#     )

#     print(
#         "Loading MedVision model..."
#     )

#     model = (
#         Qwen2_5_VLForConditionalGeneration
#         .from_pretrained(
#             MODEL_ID,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#         )
#     )

#     model.eval()

#     print(
#         "MedVision loaded."
#     )

#     # --------------------------------------------------------
#     # Load MedSAM
#     # --------------------------------------------------------

#     print(
#         "\nLoading MedSAM..."
#     )

#     if not MEDSAM_CHECKPOINT.exists():

#         raise FileNotFoundError(
#             f"MedSAM checkpoint not found:\n"
#             f"{MEDSAM_CHECKPOINT}"
#         )

#     medsam_model = sam_model_registry[
#         "vit_b"
#     ](
#         checkpoint=str(
#             MEDSAM_CHECKPOINT
#         )
#     )

#     medsam_model.to(
#         device="cuda"
#     )

#     medsam_model.eval()

#     predictor = SamPredictor(
#         medsam_model
#     )

#     print(
#         "MedSAM loaded."
#     )

#     # --------------------------------------------------------
#     # Find ALL images in ALL modality folders
#     # --------------------------------------------------------

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

#             image_paths.append(
#                 path
#             )

#     image_paths.sort()

#     if MAX_IMAGES is not None:
#         image_paths = image_paths[
#             :MAX_IMAGES
#         ]

#     print(
#         f"\nFound {len(image_paths)} images "
#         "across all modality folders."
#     )

#     # Show distribution
#     modality_counts = {}

#     for path in image_paths:

#         relative = path.relative_to(
#             IMAGE_ROOT
#         )

#         modality = (
#             relative.parts[0]
#             if len(relative.parts) > 1
#             else "Unknown"
#         )

#         modality_counts[modality] = (
#             modality_counts.get(
#                 modality,
#                 0,
#             )
#             + 1
#         )

#     print(
#         "\nImage distribution:"
#     )

#     for modality, count in sorted(
#         modality_counts.items()
#     ):

#         print(
#             f"  {modality}: {count}"
#         )

#     print(
#         "\nStarting processing..."
#     )

#     # --------------------------------------------------------
#     # Process all images
#     # --------------------------------------------------------

#     for image_path in tqdm(
#         image_paths,
#         desc="Visible cues -> MedVision -> MedSAM",
#     ):

#         relative = image_path.relative_to(
#             IMAGE_ROOT
#         )

#         modality = (
#             relative.parts[0]
#             if len(relative.parts) > 1
#             else "Unknown"
#         )

#         print(
#             "\n"
#             + "=" * 70
#         )

#         print(
#             "Modality:",
#             modality,
#         )

#         print(
#             "Image:",
#             image_path.name,
#         )

#         try:

#             # ------------------------------------------------
#             # Load image
#             # ------------------------------------------------

#             image = Image.open(
#                 image_path
#             ).convert("RGB")

#             width, height = image.size

#             # ------------------------------------------------
#             # MedVision
#             # ------------------------------------------------

#             print(
#                 "Running MedVision..."
#             )

#             raw_output, normalized_box = run_medvision(
#                 model,
#                 processor,
#                 image,
#             )

#             print(
#                 "Normalized box:",
#                 normalized_box,
#             )

#             if normalized_box is None:

#                 print(
#                     "WARNING: No valid MedVision box."
#                 )

#                 continue

#             # ------------------------------------------------
#             # Convert coordinates
#             # ------------------------------------------------

#             pixel_box = normalized_box_to_pixels(
#                 normalized_box,
#                 width,
#                 height,
#             )

#             print(
#                 "Pixel box:",
#                 pixel_box,
#             )

#             if pixel_box is None:

#                 print(
#                     "WARNING: Invalid pixel box."
#                 )

#                 continue

#             # ------------------------------------------------
#             # MedSAM
#             # ------------------------------------------------

#             print(
#                 "Running MedSAM..."
#             )

#             mask, medsam_score = run_medsam(
#                 predictor,
#                 image,
#                 pixel_box,
#             )

#             print(
#                 "MedSAM score:",
#                 medsam_score,
#             )

#             # ------------------------------------------------
#             # Output paths
#             # ------------------------------------------------

#             output_directory = (
#                 OUTPUT_ROOT
#                 / relative.parent
#             )

#             output_directory.mkdir(
#                 parents=True,
#                 exist_ok=True,
#             )

#             stem = relative.stem

#             json_path = (
#                 output_directory
#                 / f"{stem}.json"
#             )

#             mask_path = (
#                 output_directory
#                 / f"{stem}_mask.png"
#             )

#             overlay_path = (
#                 output_directory
#                 / f"{stem}_overlay.png"
#             )

#             # ------------------------------------------------
#             # Save mask
#             # ------------------------------------------------

#             mask_image = Image.fromarray(
#                 (
#                     mask.astype(np.uint8)
#                     * 255
#                 ),
#                 mode="L",
#             )

#             mask_image.save(
#                 mask_path
#             )

#             # ------------------------------------------------
#             # Save overlay
#             # ------------------------------------------------

#             save_overlay(
#                 image,
#                 pixel_box,
#                 mask,
#                 overlay_path,
#             )

#             # ------------------------------------------------
#             # Save metadata
#             # ------------------------------------------------

#             record = {
#                 "image": str(image_path),
#                 "modality": modality,

#                 "experiment": (
#                     "visible_image_cues"
#                 ),

#                 "medvision_prompt": (
#                     "Use visible arrows, circles, "
#                     "pointers, markers, labels, or "
#                     "other embedded visual cues to "
#                     "localize the underlying clinically "
#                     "relevant abnormal region."
#                 ),

#                 "medvision_raw_output": (
#                     raw_output
#                 ),

#                 "medvision_normalized_box_lower_left": (
#                     normalized_box
#                 ),

#                 "pixel_box_upper_left": (
#                     pixel_box
#                 ),

#                 "image_width": width,

#                 "image_height": height,

#                 "medsam_score": (
#                     medsam_score
#                 ),

#                 "mask_path": str(
#                     mask_path
#                 ),

#                 "overlay_path": str(
#                     overlay_path
#                 ),
#             }

#             with open(
#                 json_path,
#                 "w",
#                 encoding="utf-8",
#             ) as f:

#                 json.dump(
#                     record,
#                     f,
#                     indent=2,
#                 )

#             print(
#                 "Saved:"
#             )

#             print(
#                 "  Mask:",
#                 mask_path,
#             )

#             print(
#                 "  Overlay:",
#                 overlay_path,
#             )

#             print(
#                 "  Metadata:",
#                 json_path,
#             )

#         except Exception as error:

#             print(
#                 "\nERROR processing:"
#             )

#             print(
#                 image_path
#             )

#             print(
#                 repr(error)
#             )

#     print(
#         "\n"
#         + "=" * 70
#     )

#     print(
#         "VISIBLE-CUE EXPERIMENT COMPLETE"
#     )

#     print(
#         "Results:",
#         OUTPUT_ROOT,
#     )


# if __name__ == "__main__":
#     main()






from pathlib import Path
import re
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# Paths
# ============================================================

MODEL_ID = "YongchengYAO/MedVision-V0-7B"

IMAGE_ROOT = Path(
    r"D:\roco-medsam-test\roco_pilot"
)

# Keep this experiment separate from the previous results.
OUTPUT_ROOT = Path(
    r"D:\medvision-roco\medsam_results_visible_cues_30pct"
)

MEDSAM_ROOT = Path(
    r"D:\roco-medsam-test\MedSAM"
)

MEDSAM_CHECKPOINT = Path(
    r"D:\roco-medsam-test\checkpoints\medsam_vit_b.pth"
)


# ============================================================
# Process all images across all modalities
# ============================================================

MAX_IMAGES = None


# ============================================================
# Automatic bounding-box expansion
# ============================================================

BOX_EXPANSION = 0.30


# ============================================================
# Add MedSAM source
# ============================================================

sys.path.insert(
    0,
    str(MEDSAM_ROOT)
)

from segment_anything import (
    sam_model_registry,
    SamPredictor,
)


# ============================================================
# Output directory
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MedVision system prompt
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
    "</step-k-answer> tags."
)


# ============================================================
# Marking-aware localization prompt
# ============================================================

QUESTION = (
    "Task:\n"
    "Given the input medical image, identify and localize the "
    "most clinically relevant abnormal region.\n\n"

    "IMPORTANT LOCALIZATION CUE:\n"
    "Carefully inspect the image for visual markings that are "
    "genuinely part of the image. These may include arrows, "
    "circles, pointers, markers, labels, crosshairs, outlines, "
    "or other annotation-like visual indicators.\n\n"

    "Use these visible markings as localization cues. If a "
    "marking points to, circles, outlines, labels, or otherwise "
    "identifies an abnormal finding, use that information to "
    "determine where the underlying abnormality is located.\n\n"

    "The object of interest is the UNDERLYING MEDICAL ABNORMALITY, "
    "not the visual marking itself.\n\n"

    "Do NOT make the bounding box only around an arrow, circle, "
    "pointer, label, marker, or text. Instead, determine the "
    "actual pathological or abnormal anatomical region that the "
    "marking is indicating and place the bounding box around that "
    "region.\n\n"

    "The predicted region may intentionally include surrounding "
    "normal anatomy because the purpose is to obtain a robust "
    "region of interest for downstream segmentation.\n\n"

    "If there is no useful visible marking, rely on the medical "
    "image itself to identify the most clinically relevant "
    "abnormal region.\n\n"

    "Format requirement:\n"
    "The reasoning process and the final answer must be enclosed "
    "within <think> </think> and <answer> </answer> tags, respectively. "
    "The answer should be four decimal numbers separated by commas "
    "without any units or additional text.\n\n"

    "The first two numbers are the coordinates of the lower-left "
    "corner and the last two numbers are the coordinates of the "
    "upper-right corner of the bounding box.\n\n"

    "Use relative coordinates in image space, where the origin "
    "is at the lower-left corner of the image. Relative coordinates "
    "must be between 0 and 1.\n\n"

    "Reasoning steps:\n"
    "Step 1: Inspect the image for arrows, circles, pointers, "
    "labels, markers, or other visible markings that indicate "
    "an abnormal finding.\n"
    "Step 2: Determine the underlying abnormality indicated by "
    "the marking rather than the marking itself.\n"
    "Step 3: Determine the relative coordinates of a bounding "
    "box around the underlying abnormality.\n\n"

    "Follow the reasoning steps to get the final answer in the "
    "required format."
)


# ============================================================
# Parse MedVision bounding box
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
        float(value)
        for value in numbers[-4:]
    ]

    x0, y0, x1, y1 = values

    if not all(
        0.0 <= value <= 1.0
        for value in values
    ):
        return None

    if x1 <= x0 or y1 <= y0:
        return None

    return [
        x0,
        y0,
        x1,
        y1,
    ]


# ============================================================
# Convert MedVision normalized coordinates
# to upper-left-origin pixel coordinates
# ============================================================

def normalized_box_to_pixels(
    box,
    image_width,
    image_height,
):

    x0, y0, x1, y1 = box

    pixel_x0 = round(
        x0 * image_width
    )

    pixel_x1 = round(
        x1 * image_width
    )

    pixel_y0 = round(
        (1.0 - y1) * image_height
    )

    pixel_y1 = round(
        (1.0 - y0) * image_height
    )

    pixel_x0 = max(
        0,
        min(
            pixel_x0,
            image_width - 1,
        ),
    )

    pixel_x1 = max(
        0,
        min(
            pixel_x1,
            image_width - 1,
        ),
    )

    pixel_y0 = max(
        0,
        min(
            pixel_y0,
            image_height - 1,
        ),
    )

    pixel_y1 = max(
        0,
        min(
            pixel_y1,
            image_height - 1,
        ),
    )

    if pixel_x1 <= pixel_x0:
        return None

    if pixel_y1 <= pixel_y0:
        return None

    return [
        pixel_x0,
        pixel_y0,
        pixel_x1,
        pixel_y1,
    ]


# ============================================================
# Expand box by 30%
#
# Expansion is relative to the ORIGINAL box width/height.
# 30% on each side.
# ============================================================

def expand_box(
    pixel_box,
    image_width,
    image_height,
    expansion=0.30,
):

    x0, y0, x1, y1 = pixel_box

    box_width = x1 - x0
    box_height = y1 - y0

    expand_x = box_width * expansion
    expand_y = box_height * expansion

    expanded_x0 = round(
        x0 - expand_x
    )

    expanded_y0 = round(
        y0 - expand_y
    )

    expanded_x1 = round(
        x1 + expand_x
    )

    expanded_y1 = round(
        y1 + expand_y
    )

    # Clamp to image boundaries.
    expanded_x0 = max(
        0,
        expanded_x0,
    )

    expanded_y0 = max(
        0,
        expanded_y0,
    )

    expanded_x1 = min(
        image_width - 1,
        expanded_x1,
    )

    expanded_y1 = min(
        image_height - 1,
        expanded_y1,
    )

    if expanded_x1 <= expanded_x0:
        return None

    if expanded_y1 <= expanded_y0:
        return None

    return [
        expanded_x0,
        expanded_y0,
        expanded_x1,
        expanded_y1,
    ]


# ============================================================
# MedVision inference
# ============================================================

def run_medvision(
    model,
    processor,
    image,
):

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

    inputs = inputs.to(
        model.device
    )

    with torch.inference_mode():

        generated = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated,
        )
    ]

    output = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
    )[0]

    box = parse_box(
        output
    )

    return output, box


# ============================================================
# MedSAM inference
# ============================================================

def run_medsam(
    predictor,
    image,
    pixel_box,
):

    image_np = np.array(
        image.convert("RGB")
    )

    predictor.set_image(
        image_np
    )

    box_np = np.array(
        pixel_box,
        dtype=np.float32,
    )

    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_np,
        multimask_output=False,
    )

    mask = masks[0]

    score = float(
        scores[0]
    )

    return mask, score


# ============================================================
# Save overlay
# ============================================================

def save_overlay(
    image,
    original_box,
    expanded_box,
    mask,
    output_path,
):

    base = image.convert(
        "RGBA"
    )

    overlay_array = np.zeros(
        (
            image.height,
            image.width,
            4,
        ),
        dtype=np.uint8,
    )

    # Mask is deliberately the ONLY thing given to the
    # segmentation overlay. The arrows/text are not treated
    # as segmentation targets by this code.
    overlay_array[mask, 0] = 255
    overlay_array[mask, 1] = 0
    overlay_array[mask, 2] = 0
    overlay_array[mask, 3] = 90

    overlay = Image.fromarray(
        overlay_array,
        mode="RGBA",
    )

    result = Image.alpha_composite(
        base,
        overlay,
    )

    draw = ImageDraw.Draw(
        result
    )

    # Original MedVision box: yellow
    ox0, oy0, ox1, oy1 = original_box

    draw.rectangle(
        [ox0, oy0, ox1, oy1],
        outline=(255, 255, 0, 255),
        width=3,
    )

    # Expanded MedSAM box: cyan
    ex0, ey0, ex1, ey1 = expanded_box

    draw.rectangle(
        [ex0, ey0, ex1, ey1],
        outline=(0, 255, 255, 255),
        width=3,
    )

    result.save(
        output_path
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "MedVision -> MedSAM "
        "VISIBLE-CUE + 30% BOX EXPANSION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "CUDA:",
        torch.version.cuda,
    )

    print(
        "Box expansion:",
        f"{BOX_EXPANSION * 100:.0f}% per side",
    )

    # --------------------------------------------------------
    # Load MedVision
    # --------------------------------------------------------

    print(
        "\nLoading MedVision processor..."
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    print(
        "Loading MedVision model..."
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    )

    model.eval()

    print(
        "MedVision loaded."
    )

    # --------------------------------------------------------
    # Load MedSAM
    # --------------------------------------------------------

    print(
        "\nLoading MedSAM..."
    )

    if not MEDSAM_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"MedSAM checkpoint not found:\n"
            f"{MEDSAM_CHECKPOINT}"
        )

    medsam_model = sam_model_registry[
        "vit_b"
    ](
        checkpoint=str(
            MEDSAM_CHECKPOINT
        )
    )

    medsam_model.to(
        device="cuda"
    )

    medsam_model.eval()

    predictor = SamPredictor(
        medsam_model
    )

    print(
        "MedSAM loaded."
    )

    # --------------------------------------------------------
    # Find all images
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
            image_paths.append(
                path
            )

    image_paths.sort()

    if MAX_IMAGES is not None:
        image_paths = image_paths[
            :MAX_IMAGES
        ]

    print(
        f"\nFound {len(image_paths)} images."
    )

    # --------------------------------------------------------
    # Modality distribution
    # --------------------------------------------------------

    modality_counts = {}

    for path in image_paths:

        relative = path.relative_to(
            IMAGE_ROOT
        )

        modality = (
            relative.parts[0]
            if len(relative.parts) > 1
            else "Unknown"
        )

        modality_counts[modality] = (
            modality_counts.get(
                modality,
                0,
            )
            + 1
        )

    print(
        "\nImage distribution:"
    )

    for modality, count in sorted(
        modality_counts.items()
    ):

        print(
            f"  {modality}: {count}"
        )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for image_path in tqdm(
        image_paths,
        desc="Visible cues -> MedVision -> MedSAM",
    ):

        relative = image_path.relative_to(
            IMAGE_ROOT
        )

        modality = (
            relative.parts[0]
            if len(relative.parts) > 1
            else "Unknown"
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Modality:",
            modality,
        )

        print(
            "Image:",
            image_path.name,
        )

        try:

            # ------------------------------------------------
            # Image
            # ------------------------------------------------

            image = Image.open(
                image_path
            ).convert("RGB")

            width, height = image.size

            # ------------------------------------------------
            # MedVision
            # ------------------------------------------------

            print(
                "Running MedVision..."
            )

            raw_output, normalized_box = run_medvision(
                model,
                processor,
                image,
            )

            print(
                "MedVision normalized box:",
                normalized_box,
            )

            if normalized_box is None:

                print(
                    "WARNING: No valid MedVision box."
                )

                continue

            # ------------------------------------------------
            # Convert to pixel coordinates
            # ------------------------------------------------

            original_pixel_box = (
                normalized_box_to_pixels(
                    normalized_box,
                    width,
                    height,
                )
            )

            print(
                "Original pixel box:",
                original_pixel_box,
            )

            if original_pixel_box is None:

                print(
                    "WARNING: Invalid original box."
                )

                continue

            # ------------------------------------------------
            # Expand by 30%
            # ------------------------------------------------

            expanded_pixel_box = expand_box(
                original_pixel_box,
                width,
                height,
                BOX_EXPANSION,
            )

            print(
                "Expanded pixel box:",
                expanded_pixel_box,
            )

            if expanded_pixel_box is None:

                print(
                    "WARNING: Invalid expanded box."
                )

                continue

            # ------------------------------------------------
            # MedSAM
            # ------------------------------------------------

            print(
                "Running MedSAM..."
            )

            mask, medsam_score = run_medsam(
                predictor,
                image,
                expanded_pixel_box,
            )

            print(
                "MedSAM score:",
                medsam_score,
            )

            # ------------------------------------------------
            # Output directory
            # ------------------------------------------------

            output_directory = (
                OUTPUT_ROOT
                / relative.parent
            )

            output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            stem = relative.stem

            json_path = (
                output_directory
                / f"{stem}.json"
            )

            mask_path = (
                output_directory
                / f"{stem}_mask.png"
            )

            overlay_path = (
                output_directory
                / f"{stem}_overlay.png"
            )

            # ------------------------------------------------
            # Save mask
            # ------------------------------------------------

            mask_image = Image.fromarray(
                (
                    mask.astype(np.uint8)
                    * 255
                ),
                mode="L",
            )

            mask_image.save(
                mask_path
            )

            # ------------------------------------------------
            # Save overlay
            # ------------------------------------------------

            save_overlay(
                image,
                original_pixel_box,
                expanded_pixel_box,
                mask,
                overlay_path,
            )

            # ------------------------------------------------
            # Save metadata
            # ------------------------------------------------

            record = {
                "image": str(image_path),
                "modality": modality,

                "experiment": (
                    "visible_cues_with_30_percent_box_expansion"
                ),

                "box_expansion_per_side": (
                    BOX_EXPANSION
                ),

                "medvision_instruction": (
                    "Use arrows, circles, pointers, "
                    "labels, markers, and other visible "
                    "embedded markings as localization "
                    "cues, but localize the underlying "
                    "abnormality rather than the marking."
                ),

                "segmentation_instruction": (
                    "The visible markings are localization "
                    "cues only. The MedSAM target is the "
                    "underlying medical abnormality."
                ),

                "medvision_raw_output": (
                    raw_output
                ),

                "medvision_normalized_box_lower_left": (
                    normalized_box
                ),

                "original_pixel_box_upper_left": (
                    original_pixel_box
                ),

                "expanded_pixel_box_upper_left": (
                    expanded_pixel_box
                ),

                "image_width": width,

                "image_height": height,

                "medsam_score": (
                    medsam_score
                ),

                "mask_path": str(
                    mask_path
                ),

                "overlay_path": str(
                    overlay_path
                ),
            }

            with open(
                json_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    record,
                    f,
                    indent=2,
                )

            print(
                "Saved mask:",
                mask_path,
            )

            print(
                "Saved overlay:",
                overlay_path,
            )

            print(
                "Saved metadata:",
                json_path,
            )

        except Exception as error:

            print(
                "\nERROR processing:"
            )

            print(
                image_path
            )

            print(
                repr(error)
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VISIBLE-CUE + 30% EXPANSION EXPERIMENT COMPLETE"
    )

    print(
        "Results:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()