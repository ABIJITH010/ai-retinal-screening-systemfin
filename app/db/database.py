import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = BACKEND_ROOT / "database" / "retinal_screening.db"
DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"

engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


def ensure_sqlite_schema() -> None:
    """Upgrade the bundled SQLite database to match the ORM models."""
    if not DATABASE_URL.startswith("sqlite"):
        return

    expected_columns = {
        "patients": {
            "age": "INTEGER",
            "gender": "VARCHAR",
            "phone": "VARCHAR",
            "address": "VARCHAR",
            "doctor_id": "INTEGER",
        },
        "reports": {
            "heatmap_url": "VARCHAR",
            "prediction": "VARCHAR",
            "confidence": "FLOAT",
            "created_at": "DATETIME",
        },
        "users": {
            "name": "VARCHAR",
            "email": "VARCHAR",
            "password": "VARCHAR",
        },
    }

    with engine.begin() as connection:
        for table_name, columns in expected_columns.items():
            existing_rows = connection.execute(
                text(f"PRAGMA table_info({table_name})")
            ).mappings()
            existing_columns = {row["name"] for row in existing_rows}
            if not existing_columns:
                continue

            for column_name, column_type in columns.items():
                if column_name in existing_columns:
                    continue
                connection.execute(
                    text(
                        f"ALTER TABLE {table_name} "
                        f"ADD COLUMN {column_name} {column_type}"
                    )
                )
