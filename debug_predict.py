import logging
import sys
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ai_service import predict_dr_stage

def main():
    root = Path(__file__).resolve().parent.parent
    dataset_dir = root / "dataset" / "images"
    images = list(dataset_dir.glob("*.[pj][np][g]"))
    if not images:
        print("No images found to test.")
        return
    
    test_img = images[0]
    print(f"Testing on {test_img}")
    res = predict_dr_stage(str(test_img))
    print("Result:", res)

if __name__ == "__main__":
    main()
