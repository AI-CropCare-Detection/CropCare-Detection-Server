from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import router

# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Plant Disease Classifier API",
        description=(
            "YOLOv7-based plant-disease classifier with optional "
            "test-time augmentation (TTA). Upload a leaf image to get "
            "a disease prediction with confidence scores."
        ),
        version="2.0.0",
        docs_url="/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    app.include_router(router)
    return app


app = create_app()


# def main() -> None:
#     host = os.getenv("HOST", "0.0.0.0")
#     port = int(os.getenv("PORT", 8000))
#     uvicorn.run(app, host=host, port=port)


# if __name__ == "__main__":
#     main()