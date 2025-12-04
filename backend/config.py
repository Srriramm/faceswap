"""
Configuration settings for the Face Swap API
"""

# Model settings
MODEL_NAME = "Qwen/Qwen-Image-Edit-2509"
LORA_REPO = "Alissonerdx/BFS-Best-Face-Swap"
LORA_WEIGHT_NAME = "bfs_head_v3_qwen_image_edit_2509.safetensors"
LORA_ADAPTER_NAME = "bfs"

# Inference settings
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 5.0
FACE_CROP_MULTIPLIER = 3  # How much context to include around face

# Server settings
HOST = "0.0.0.0"
PORT = 8000
CORS_ORIGINS = ["*"]

# Face swap prompt template
FACE_SWAP_PROMPT = (
    "head_swap: start with Picture 1 as the base image, keeping its lighting, "
    "environment, and background. remove the head from Picture 1 completely and "
    "replace it with the head from Picture 2. ensure the head and body have correct "
    "anatomical proportions, and blend the skin tones, shadows, and lighting naturally "
    "so the final result appears as one coherent, realistic person."
)
