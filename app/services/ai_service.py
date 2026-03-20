import logging
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple

# Import from the AI package in the root
from ai.inference.predictor import DRPredictor, PredictorConfig
from ai.explainability.gradcam import GradCAM, GradCAMConfig
from ai.explainability.heatmap import apply_colormap_on_image

logger = logging.getLogger(__name__)

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parents[3]

# Global singletons to avoid reloading the model on every request
_predictor = None
_gradcam = None

def _get_predictor_and_gradcam():
    """
    Initialize and return the AI predictor and Grad-CAM objects.
    """
    global _predictor, _gradcam
    if _predictor is None:
        try:
            logger.info("Initializing AI Predictor...")
            weights_path = BASE_DIR / "ai" / "model" / "weights" / "dr_model.pth"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            config = PredictorConfig(model_path=weights_path, device=device, model_name="ensemble")
            _predictor = DRPredictor(config)
            model = _predictor.model
            
            # Target the last convolutional block of the EfficientNet-B0 branch
            target_layer = "features.8.0" 
            if hasattr(model, "efficientnet"):
                target_layer = "efficientnet.features.8.0"
            
            g_config = GradCAMConfig(target_layer=target_layer)
            _gradcam = GradCAM(_predictor.model, g_config)
            
            logger.info("AI Infrastructure initialized successfully.")
        except Exception as e:
            logger.error(f"Critical error during AI initialization: {str(e)}", exc_info=True)
            _predictor = None
            _gradcam = None
            raise RuntimeError(f"AI components failed to load: {e}")
            
    return _predictor, _gradcam

def predict_dr_stage(image_path: str) -> Tuple[str, float, str]:
    """
    Predict DR stage and generate Heatmap.
    This is a computationally intensive synchronous call.
    """
    try:
        predictor, gradcam = _get_predictor_and_gradcam()
        
        # 1. Run inference
        result = predictor.predict(image_path)
        
        prediction = result["prediction"]
        confidence = result["confidence"]
        severity_idx = result["severity_level"]
        
        # 2. Generate Heatmap
        original_img = Image.open(image_path).convert("RGB")
        processed_arr = predictor._prepare_image(image_path)
        
        # TCHW tensor
        input_tensor = torch.from_numpy(processed_arr.transpose(2, 0, 1)).unsqueeze(0)
        input_tensor = input_tensor.to(predictor.device, dtype=torch.float32)
        
        # Grad-CAM
        heatmap_np = gradcam.generate(input_tensor, target_class=severity_idx)
        heatmap_img = apply_colormap_on_image(original_img, heatmap_np, alpha=0.5)
        
        # 3. Save Heatmap
        p = Path(image_path)
        heatmap_path = str(p.parent / f"{p.stem}_heatmap{p.suffix}")
        heatmap_img.save(heatmap_path)
        
        logger.info(f"Report generated for {p.name}: {prediction} ({confidence:.1%})")
        return prediction, confidence, heatmap_path

    except Exception as e:
        logger.error(f"AI Service error for image {image_path}: {str(e)}", exc_info=True)
        raise
