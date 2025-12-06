import gradio as gr
import requests
import base64
import io
from PIL import Image
import json
from typing import List, Dict

# Backend API URL - use Docker service name in container, localhost otherwise
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://13.201.15.166:8000")

# Global state to store detected faces
detected_faces = []
source_image_path = None


def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            status = "✅ Backend is healthy"
            if data.get("gpu_available"):
                gpu_name = data.get("gpu_name", "Unknown GPU")
                status += f"\n🎮 GPU Available: {gpu_name}"
            else:
                status += f"\n⚠️ Running on CPU (slower)"
            if data.get("model_loaded"):
                status += f"\n🤖 Model loaded and ready"
            else:
                status += f"\n⏳ Model will load on first request"
            return status
        return "❌ Backend not responding"
    except Exception as e:
        return f"❌ Cannot connect to backend: {str(e)}\n\nMake sure to run:\ncd backend && uvicorn server:app --reload"


def detect_faces_from_image(image):
    """Detect faces in the uploaded image"""
    global detected_faces, source_image_path
    
    if image is None:
        return None, "Please upload an image first", []
    
    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Send to backend
        files = {"file": ("image.png", img_byte_arr, "image/png")}
        response = requests.post(f"{BACKEND_URL}/detect", files=files)
        
        if response.status_code == 200:
            data = response.json()
            detected_faces = data.get("faces", [])
            source_image_path = image
            
            if len(detected_faces) == 0:
                return image, "❌ No faces detected in the image", []
            
            # Create gallery of detected faces
            face_gallery = []
            for face_data in detected_faces:
                face_img = base64_to_pil(face_data["preview"])
                face_gallery.append((face_img, f"Face {face_data['id']}"))
            
            message = f"✅ Detected {len(detected_faces)} face(s)!\n\nClick on each face below to upload a replacement face."
            return image, message, face_gallery
        else:
            return image, f"❌ Error: {response.text}", []
            
    except Exception as e:
        return image, f"❌ Error detecting faces: {str(e)}", []


def create_face_uploader(face_id):
    """Create upload widget for a specific face"""
    return gr.Image(
        type="pil",
        label=f"Upload replacement for Face {face_id}",
        interactive=True
    )


def swap_faces(source_img, *target_images):
    """Perform face swap with uploaded target faces"""
    global detected_faces, source_image_path
    
    if source_img is None:
        return None, "❌ Please upload a source image first"
    
    if len(detected_faces) == 0:
        return source_img, "❌ No faces detected. Click 'Detect Faces' first"
    
    # Prepare target faces mapping
    target_faces_dict = {}
    
    for i, target_img in enumerate(target_images):
        if target_img is not None and i < len(detected_faces):
            # Convert PIL to base64
            target_b64 = pil_to_base64(target_img)
            target_faces_dict[str(i)] = target_b64
    
    if len(target_faces_dict) == 0:
        return source_img, "❌ Please upload at least one target face"
    
    try:
        # Convert source image to bytes
        img_byte_arr = io.BytesIO()
        source_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Send to backend
        files = {"source_image": ("source.png", img_byte_arr, "image/png")}
        data = {"target_faces": json.dumps(target_faces_dict)}
        
        response = requests.post(
            f"{BACKEND_URL}/swap",
            files=files,
            data=data,
            timeout=120  # 2 minutes timeout
        )
        
        if response.status_code == 200:
            # Convert response bytes to PIL Image
            result_image = Image.open(io.BytesIO(response.content))
            return result_image, f"✅ Face swap completed successfully!\n\nSwapped {len(target_faces_dict)} face(s)"
        else:
            return source_img, f"❌ Error during face swap: {response.text}"
            
    except Exception as e:
        return source_img, f"❌ Error: {str(e)}"


def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_pil(b64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


# Create Gradio Interface
with gr.Blocks(title="Face Swap Application") as demo:
    gr.Markdown("""
    # 🎭 Face Swap Application
    ### Powered by Qwen Image Edit + BFS Head V3
    
    **Instructions:**
    1. Check backend status
    2. Upload your source image
    3. Click "Detect Faces" to find all faces
    4. Upload replacement faces for each detected face
    5. Click "Swap Faces" to generate the result
    """)
    
    with gr.Row():
        with gr.Column():
            health_btn = gr.Button("🔍 Check Backend Status", variant="secondary")
            health_output = gr.Textbox(label="Backend Status", interactive=False, lines=3)
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Upload Source Image")
            source_image = gr.Image(type="pil", label="Source Image", interactive=True)
            detect_btn = gr.Button("🔎 Detect Faces", variant="primary", size="lg")
            detect_status = gr.Textbox(label="Detection Status", interactive=False, lines=2)
        
        with gr.Column(scale=1):
            gr.Markdown("### Step 2: Detected Faces")
            face_gallery = gr.Gallery(
                label="Detected Faces",
                show_label=True,
                columns=3,
                rows=2,
                object_fit="contain",
                height="auto"
            )
    
    gr.Markdown("---")
    gr.Markdown("### Step 3: Upload Replacement Faces")
    
    # Create upload slots for up to 6 faces
    with gr.Row():
        target_face_0 = gr.Image(type="pil", label="Replacement for Face 0", interactive=True)
        target_face_1 = gr.Image(type="pil", label="Replacement for Face 1", interactive=True)
        target_face_2 = gr.Image(type="pil", label="Replacement for Face 2", interactive=True)
    
    with gr.Row():
        target_face_3 = gr.Image(type="pil", label="Replacement for Face 3", interactive=True)
        target_face_4 = gr.Image(type="pil", label="Replacement for Face 4", interactive=True)
        target_face_5 = gr.Image(type="pil", label="Replacement for Face 5", interactive=True)
    
    gr.Markdown("---")
    gr.Markdown("### Step 4: Generate Result")
    
    swap_btn = gr.Button("🎨 Swap Faces", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            result_image = gr.Image(type="pil", label="Result")
            result_status = gr.Textbox(label="Swap Status", interactive=False, lines=2)
    
    # Event handlers
    health_btn.click(
        fn=check_backend_health,
        inputs=[],
        outputs=[health_output]
    )
    
    detect_btn.click(
        fn=detect_faces_from_image,
        inputs=[source_image],
        outputs=[source_image, detect_status, face_gallery]
    )
    
    swap_btn.click(
        fn=swap_faces,
        inputs=[
            source_image,
            target_face_0,
            target_face_1,
            target_face_2,
            target_face_3,
            target_face_4,
            target_face_5
        ],
        outputs=[result_image, result_status]
    )
    
    gr.Markdown("""
    ---
    ### 💡 Tips:
    - Use high-quality images for better results
    - Ensure faces are clearly visible
    - The model works best with frontal faces
    - Processing may take 1-2 minutes depending on image size
    
    ### ⚙️ Technical Details:
    - **Model**: Qwen Image Edit 2509 + BFS Head V3 LoRA
    - **Backend**: FastAPI (Port 8000)
    - **GPU**: Recommended for faster processing
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
