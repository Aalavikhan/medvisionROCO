import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

MODEL_ID = "YongchengYAO/MedVision-V0-7B"

print("Loading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

print("Loading model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Model loaded.")
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))