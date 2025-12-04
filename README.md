# Face Swap Application

A powerful face swapping application using **Qwen Image Edit Pipeline** with **BFS Head V3 LoRA** for high-quality face swaps.

## 🌟 Features

- **Multi-Face Detection**: Automatically detects all faces in an image
- **Selective Face Swapping**: Choose which faces to swap
- **High Quality**: Uses state-of-the-art AI models (Qwen + BFS)
- **GPU Accelerated**: Optimized for CUDA-enabled GPUs
- **User-Friendly Interface**: Clean Gradio web interface
- **🐳 Docker Support**: Easy deployment with Docker Compose

## 🚀 Quick Start with Docker (Recommended)

The easiest way to run the application is with Docker Compose:

### Prerequisites
- [Docker Desktop](https://www.docker.com/get-started/) installed
- NVIDIA GPU with [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (optional, for GPU support)

### Run with GPU
```bash
docker-compose up --build
```

### Run on CPU (slower, no GPU required)
```bash
docker-compose -f docker-compose.cpu.yml up --build
```

### Access the Application
- **Frontend**: http://localhost:7860
- **Backend API**: http://localhost:8000/docs

That's it! 🎉

📖 **For detailed Docker instructions**, see [DOCKER.md](DOCKER.md)
```

### 4. Start the Application

#### Option A: Using Batch Scripts (Windows)

Open **two separate terminals**:

**Terminal 1 - Backend:**
```bash
start_backend.bat
```

**Terminal 2 - Frontend:**
```bash
start_frontend.bat
```

#### Option B: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
python app.py
```

### 5. Access the Application

- **Frontend UI**: http://localhost:7860
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📖 How to Use

1. **Check Backend Status**: Click "Check Backend Status" to ensure the server is running
2. **Upload Source Image**: Select an image containing faces you want to modify
3. **Detect Faces**: Click "Detect Faces" to find all faces in the image
4. **Upload Replacement Faces**: For each detected face, upload a target face image
5. **Swap Faces**: Click "Swap Faces" to generate the final result
6. **Download Result**: Right-click the result image to save

## 🎯 Use Cases

- **Portrait Photography**: Replace faces in group photos
- **Content Creation**: Create entertaining face-swapped content
- **Historical Recreation**: Swap modern faces with historical figures
- **Entertainment**: Fun experiments with celebrity faces

## ⚙️ Configuration

### Backend Configuration

Edit `backend/server.py` to modify:

- **Model Selection**: Change the model weights or LoRA adapter
- **Inference Steps**: Adjust `num_inference_steps` (default: 30)
- **Guidance Scale**: Modify `guidance_scale` (default: 5.0)
- **Crop Size**: Change the face crop multiplier (default: 3x)

### Frontend Configuration

Edit `frontend/app.py` to modify:

- **Port**: Change `server_port` in `demo.launch()`
- **Backend URL**: Update `BACKEND_URL` if running on different host
- **Max Faces**: Extend the target face upload slots

## 🔧 Troubleshooting

### Backend Won't Start

- **CUDA Error**: Ensure you have CUDA installed and GPU drivers updated
- **Module Not Found**: Run `pip install -r requirements.txt` again
- **Port Already in Use**: Change port in uvicorn command

### No Faces Detected

- Ensure faces are clearly visible and well-lit
- Try images with frontal faces
- Check image resolution (not too small)

### Slow Processing

- First run downloads models (~5GB) - this is normal
- Use GPU for faster inference
- Reduce `num_inference_steps` for quicker results

### Connection Error in Frontend

- Ensure backend is running first
- Check `BACKEND_URL` matches your backend address
- Verify firewall isn't blocking ports

## 📦 Model Information

- **Base Model**: `Qwen/Qwen-Image-Edit-2509`
- **LoRA Adapter**: `Alissonerdx/BFS-Best-Face-Swap`
- **Weight**: `bfs_head_v3_qwen_image_edit_2509.safetensors`
- **Precision**: FP16 (half precision)

## 🎨 Technical Stack

### Backend
- **FastAPI**: Modern web framework
- **PyTorch**: Deep learning framework
- **Diffusers**: Hugging Face diffusion models
- **face_recognition**: Face detection library
- **OpenCV & Pillow**: Image processing

### Frontend
- **Gradio**: ML web interface framework
- **Requests**: HTTP client

## 📝 API Endpoints

### `GET /`
Health check endpoint

### `GET /health`
Returns backend status, GPU availability, and model loading status

### `POST /detect`
Detects faces in uploaded image

**Request:**
- `file`: Image file (multipart/form-data)

**Response:**
```json
{
  "faces": [
    {
      "id": 0,
      "box": [top, right, bottom, left],
      "preview": "base64_encoded_image"
    }
  ]
}
```

### `POST /swap`
Performs face swap operation

**Request:**
- `source_image`: Source image file
- `target_faces`: JSON string mapping face IDs to base64 target images

**Response:**
- PNG image (binary)

## 🤝 Contributing

This is a personal project. Feel free to fork and modify for your needs!

## ⚠️ Disclaimer

This tool is for educational and entertainment purposes only. Please:
- Respect privacy and consent when using face swap technology
- Do not create misleading or harmful content
- Follow local laws regarding image manipulation
- Credit original creators when sharing results

## 📄 License

This project uses models from:
- Qwen Image Edit (Alibaba Cloud)
- BFS Best Face Swap (Alissonerdx)

Please review their respective licenses before commercial use.

## 🙏 Acknowledgments

- Qwen team for the amazing Image Edit model
- Alissonerdx for the BFS LoRA fine-tuning
- Hugging Face for the diffusers library
- Gradio team for the fantastic UI framework
