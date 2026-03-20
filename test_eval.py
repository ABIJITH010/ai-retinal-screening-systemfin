import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(r"c:\Users\ABIJITH\OneDrive\Desktop\ai-retinal-screening-system1\ai-retinal-screening-system")))

def check_model():
    path = Path(r"c:\Users\ABIJITH\OneDrive\Desktop\ai-retinal-screening-system1\ai-retinal-screening-system\ai\model\weights\dr_model.pth")
    state = torch.load(path, map_location="cpu")
    
    from ai.inference.predictor import DRPredictor, PredictorConfig
    
    config = PredictorConfig(model_path=path, model_name="ensemble", device="cpu")
    predictor = DRPredictor(config)
    img_path = Path(r"c:\Users\ABIJITH\OneDrive\Desktop\ai-retinal-screening-system1\ai-retinal-screening-system\backend\uploads\0e9d6c70eaf7.png")
    res = predictor.predict(img_path)
    
    print("=== PREDICTION RESULT ===")
    for key, value in res.items():
        print(f"{key}: {value}")
    print("=========================")

if __name__ == "__main__":
    check_model()
