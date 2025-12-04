"""
Utility functions for image processing and conversion
"""
import io
import base64
from PIL import Image


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    return Image.open(io.BytesIO(base64.b64decode(base64_str)))


def calculate_crop_region(face_box, image_shape, multiplier=3):
    """
    Calculate expanded crop region around a face
    
    Args:
        face_box: Tuple of (top, right, bottom, left)
        image_shape: Tuple of (height, width)
        multiplier: How much to expand the crop (default: 3x face size)
    
    Returns:
        Tuple of (crop_top, crop_bottom, crop_left, crop_right)
    """
    top, right, bottom, left = face_box
    height, width = image_shape[:2]
    
    face_h = bottom - top
    face_w = right - left
    center_y = top + face_h // 2
    center_x = left + face_w // 2
    
    # Expand crop size
    crop_size = max(face_h, face_w) * multiplier
    
    crop_top = max(0, center_y - crop_size // 2)
    crop_bottom = min(height, center_y + crop_size // 2)
    crop_left = max(0, center_x - crop_size // 2)
    crop_right = min(width, center_x + crop_size // 2)
    
    return crop_top, crop_bottom, crop_left, crop_right
