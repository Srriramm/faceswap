"""
Face Swap API Server
Uses Qwen Image Edit Pipeline with BFS Head V3 LoRA for face swapping
"""

# Standard library imports
import io
import json
from typing import Dict

# Third-party imports
import numpy as np
import torch
from PIL import Image
import face_recognition
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from model import model_manager
from utils import image_to_base64, base64_to_image, calculate_crop_region
from config import CORS_ORIGINS, FACE_SWAP_PROMPT, NUM_INFERENCE_STEPS, GUIDANCE_SCALE, FACE_CROP_MULTIPLIER


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI startup and shutdown events.
    Model loading is not done here to prevent blocking container startup.
    Models are loaded lazily on first inference request.
    """
    print("🚀 Face Swap API Server started successfully")
    print("ℹ️  Model will be loaded on first inference request (lazy loading)")
    yield
    print("👋 Shutting down Face Swap API Server")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Face Swap API",
    description="API for face detection and swapping using Qwen Image Edit + BFS LoRA",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "running",
        "message": "Face Swap API is ready!",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Returns healthy even when model is not loaded (lazy loading approach).
    """
    gpu_available = torch.cuda.is_available()
    model_loaded = model_manager.is_loaded()
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "model_loaded": model_loaded,
        "ready_for_inference": model_loaded,
        "message": "Model ready" if model_loaded else "Model will load on first inference request"
    }


@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """
    Detect faces in an uploaded image
    
    Returns:
        JSON with detected face locations and preview images
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # Detect faces using face_recognition library
    face_locations = face_recognition.face_locations(image_np)

    results = []
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Create a crop for preview
        face_image = image.crop((left, top, right, bottom))

        results.append({
            "id": i,
            "box": [top, right, bottom, left],
            "preview": image_to_base64(face_image)
        })

    print(f"✅ Detected {len(results)} face(s) in uploaded image")
    return JSONResponse(content={"faces": results})


@app.post("/swap")
async def swap_faces(
    source_image: UploadFile = File(...),
    target_faces: str = Form("{}")
):
    """
    Perform face swapping on source image
    
    Args:
        source_image: Image containing faces to be replaced
        target_faces: JSON mapping of face IDs to target face images (base64)
    
    Returns:
        PNG image with swapped faces
    """
    print("🔄 Starting face swap operation...")
    print(f"   Received file: {source_image.filename}")
    print(f"   Target faces data length: {len(target_faces)}")
    
    # Parse target faces JSON
    try:
        targets: Dict[str, str] = json.loads(target_faces)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in target_faces: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid JSON in target_faces: {str(e)}"}
        )

    # Load source image
    source_content = await source_image.read()
    source_pil = Image.open(io.BytesIO(source_content)).convert("RGB")
    source_np = np.array(source_pil)

    # Detect faces in source
    face_locations = face_recognition.face_locations(source_np)
    print(f"   Found {len(face_locations)} face(s) in source image")

    # Start with copy of source as canvas
    current_canvas = source_pil.copy()

    # Lazy load model if not already loaded
    if not model_manager.is_loaded():
        print("🔄 Model not loaded yet, loading now (this may take a few minutes)...")
        try:
            model_manager.load()
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Model loading failed: {str(e)}"}
            )
    
    # Process each target face
    for face_id_str, target_b64 in targets.items():
        face_id = int(face_id_str)
        
        if face_id >= len(face_locations):
            print(f"   ⚠️ Skipping face {face_id} (out of range)")
            continue

        print(f"   Processing face {face_id}...")
        
        # Get face location
        face_box = face_locations[face_id]
        target_pil = base64_to_image(target_b64).convert("RGB")

        # Calculate crop region with context
        crop_top, crop_bottom, crop_left, crop_right = calculate_crop_region(
            face_box,
            source_np.shape,
            FACE_CROP_MULTIPLIER
        )

        # Crop source image with context
        source_crop = current_canvas.crop((crop_left, crop_top, crop_right, crop_bottom))

        # Run model inference
        try:
            output = model_manager.generate(
                source_image=source_crop,
                target_image=target_pil,
                prompt=FACE_SWAP_PROMPT,
                num_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE
            )

            # Resize output to match crop size and paste back
            output = output.resize(source_crop.size)
            current_canvas.paste(output, (crop_left, crop_top))
            print(f"   ✅ Face {face_id} swapped successfully")

        except Exception as e:
            print(f"   ❌ Inference failed for face {face_id}: {e}")

    # Return final result as PNG
    buffered = io.BytesIO()
    current_canvas.save(buffered, format="PNG")
    print("✅ Face swap completed!")
    
    return Response(content=buffered.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT
    
    print("🚀 Starting Face Swap API Server...")
    uvicorn.run(app, host=HOST, port=PORT)
