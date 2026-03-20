from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# VENDOR_PATH = Path(r"C:\Users\ABIJITH\AppData\Local\ai-retinal-pydeps")
BACKEND_PATH = PROJECT_ROOT / "backend"

# Add PROJECT_ROOT and BACKEND_PATH to sys.path
for path in (str(PROJECT_ROOT), str(BACKEND_PATH)):
    if path not in sys.path:
        sys.path.insert(0, path)

import uvicorn


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True, reload_dirs=[str(BACKEND_PATH), str(PROJECT_ROOT / "ai")])
