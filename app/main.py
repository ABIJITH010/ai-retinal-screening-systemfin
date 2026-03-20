from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import auth, health, patients, reports
from app.db.database import Base, engine, ensure_sqlite_schema


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load AI infrastructure on startup
    from app.services.ai_service import _get_predictor_and_gradcam
    import anyio
    
    # Run heavy initialization in a separate thread to not block start-up
    await anyio.to_thread.run_sync(_get_predictor_and_gradcam)
    yield

app = FastAPI(lifespan=lifespan)

uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Backend running"}


app.include_router(health.router)
app.include_router(auth.router)
app.include_router(patients.router)
app.include_router(reports.router)

Base.metadata.create_all(bind=engine)
ensure_sqlite_schema()
