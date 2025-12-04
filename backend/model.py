"""
Model loading and management
"""
import torch
from diffusers import QwenImageEditPlusPipeline
from config import MODEL_NAME, LORA_REPO, LORA_WEIGHT_NAME, LORA_ADAPTER_NAME


class ModelManager:
    """Manages the face swap model pipeline"""
    
    def __init__(self):
        self.pipeline = None
        self._check_gpu()
    
    def _check_gpu(self):
        """Check and report GPU availability"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU Available: {gpu_name}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️ No GPU found! Processing will be slower on CPU")
            print("   For GPU support: Runtime → Change runtime type → Select GPU")
    
    def load(self):
        """Load the model pipeline with LoRA weights"""
        if self.pipeline is not None:
            print("ℹ️ Model already loaded")
            return
        
        print("🔄 Loading Qwen Image Edit Pipeline...")
        print("   This may take 3-5 minutes on first run...")
        
        try:
            # Load base model
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16
            )
            
            print(f"🔄 Loading LoRA weights ({LORA_WEIGHT_NAME})...")
            # Load LoRA
            self.pipeline.load_lora_weights(
                LORA_REPO,
                weight_name=LORA_WEIGHT_NAME,
                adapter_name=LORA_ADAPTER_NAME
            )
            self.pipeline.set_adapters([LORA_ADAPTER_NAME])
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline.to(device)
            
            print(f"✅ Model loaded successfully on {device.upper()}!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.pipeline is not None
    
    def generate(self, source_image, target_image, prompt, num_steps=30, guidance_scale=5.0):
        """
        Generate face swap
        
        Args:
            source_image: PIL Image of source (with face to replace)
            target_image: PIL Image of target face
            prompt: Text prompt for generation
            num_steps: Number of inference steps
            guidance_scale: Guidance scale for generation
        
        Returns:
            PIL Image with swapped face
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")
        
        inputs = {
            "image": [source_image, target_image],
            "prompt": prompt,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
        }
        
        with torch.inference_mode():
            output = self.pipeline(**inputs).images[0]
        
        return output


# Global model manager instance
model_manager = ModelManager()
