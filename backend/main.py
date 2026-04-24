"""
NewsForge — FastAPI Backend
==============================
Serves the fine-tuned DistilBERT bias classifier and connects to a local
Ollama instance running Mistral 7B for bias-style rewriting.

Endpoints:
    POST /analyze  — Classify bias and generate 5 style rewrites
    GET  /health   — Model and Ollama connectivity status

Startup:
    1. Loads DistilBERT from model/saved_model/
    2. Tests Ollama connectivity at http://localhost:11434

Usage:
    uvicorn main:app --reload --port 8000
"""

import os
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import torch
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "saved_model")
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"
MAX_LENGTH = 256

# ─────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────

model = None
tokenizer = None
label_names = None
device = None
model_loaded = False
ollama_available = False


# ─────────────────────────────────────────────
# Rewrite Style Prompts
# ─────────────────────────────────────────────
# Each style has a carefully engineered system prompt reflecting
# real journalistic conventions and editorial perspectives.

STYLE_PROMPTS = {
    "neutral": {
        "name": "The Neutral Wire",
        "system": (
            "You are a senior wire service editor at a Reuters-style news agency. "
            "Your mandate is absolute neutrality. Rewrite the following news text following "
            "these strict rules:\n"
            "- Use passive voice wherever possible\n"
            "- Attribution for every claim (e.g., 'according to officials', 'data shows')\n"
            "- Zero adjectives that carry emotional weight\n"
            "- No superlatives, no intensifiers\n"
            "- Present only verifiable facts\n"
            "- Remove all editorial commentary and opinion\n"
            "- Use measured, dispassionate language throughout\n"
            "- Structure: Who, What, When, Where — nothing more\n"
            "Output ONLY the rewritten text. No preamble, no explanation."
        ),
    },
    "corporate": {
        "name": "The Establishment Post",
        "system": (
            "You are the chief editor of an establishment-aligned broadsheet newspaper "
            "that frames all events through economic and institutional lenses. Rewrite the "
            "following news text following these editorial guidelines:\n"
            "- Frame every issue in terms of economic impact, GDP, market effects, and stakeholder interests\n"
            "- Use institutional language: 'stakeholders', 'fiscal responsibility', 'regulatory framework'\n"
            "- Present government and corporate institutions as stabilizing forces\n"
            "- Emphasize bipartisan consensus and pragmatic governance\n"
            "- Reference market reactions, investor confidence, and economic indicators\n"
            "- Use measured, authoritative tone with formal register\n"
            "- Prefer passive constructions that center institutions over individuals\n"
            "- Frame disruption as risk, stability as virtue\n"
            "Output ONLY the rewritten text. No preamble, no explanation."
        ),
    },
    "activist": {
        "name": "The People's Tribune",
        "system": (
            "You are a senior editor at an investigative publication that centers "
            "marginalized communities and challenges power structures. Rewrite the "
            "following news text following these editorial guidelines:\n"
            "- Center the experiences of affected communities and ordinary people\n"
            "- Name power structures and systemic forces at play\n"
            "- Use framing that highlights inequality, justice, and accountability\n"
            "- Challenge official narratives and institutional claims\n"
            "- Include context about historical patterns of oppression or exploitation\n"
            "- Use direct, urgent language that conveys the human cost\n"
            "- Frame issues as systemic rather than individual\n"
            "- Question who benefits and who is harmed\n"
            "Output ONLY the rewritten text. No preamble, no explanation."
        ),
    },
    "sensationalist": {
        "name": "Daily Sensation",
        "system": (
            "You are the editor of a high-energy tabloid newspaper that thrives on "
            "dramatic, attention-grabbing coverage. Rewrite the following news text "
            "following these editorial guidelines:\n"
            "- Use DRAMATIC capitalization for key phrases\n"
            "- Create urgency: 'BREAKING', 'SHOCKING', 'EXPLOSIVE'\n"
            "- Employ emotional, visceral language throughout\n"
            "- Use short, punchy sentences for maximum impact\n"
            "- Add exclamation marks generously!\n"
            "- Frame events as unprecedented crises or stunning revelations\n"
            "- Use clickbait-style hooks and cliffhangers\n"
            "- Appeal to fear, outrage, or awe\n"
            "- Make the mundane sound extraordinary\n"
            "Output ONLY the rewritten text. No preamble, no explanation."
        ),
    },
    "state": {
        "name": "State Gazette",
        "system": (
            "You are the editor of an official state-aligned newspaper that frames "
            "all events through the lens of national stability and government authority. "
            "Rewrite the following news text following these editorial guidelines:\n"
            "- Use passive deflection to obscure agency in negative outcomes\n"
            "- Present government actions as measured, responsible, and necessary\n"
            "- Frame dissent as destabilizing or foreign-influenced\n"
            "- Emphasize national unity, order, and institutional strength\n"
            "- Use bureaucratic language that distances readers from events\n"
            "- Present challenges as 'being addressed' by competent authorities\n"
            "- Minimize negative outcomes, emphasize government response\n"
            "- Frame the state as the guarantor of stability and progress\n"
            "Output ONLY the rewritten text. No preamble, no explanation."
        ),
    },
}


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """Input schema for the /analyze endpoint."""
    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Raw news text to analyze for bias",
        examples=["The radical left-wing agenda is destroying American values and traditions."],
    )


class BiasResult(BaseModel):
    """Bias classification results from DistilBERT."""
    bias_label: str
    confidence: float
    probabilities: dict[str, float]


class RewriteResult(BaseModel):
    """Rewritten text in a specific bias style."""
    style: str
    name: str
    text: Optional[str] = None
    error: Optional[str] = None


class AnalyzeResponse(BaseModel):
    """Complete response from the /analyze endpoint."""
    bias: BiasResult
    rewrites: dict[str, Optional[str]]
    rewrite_errors: dict[str, Optional[str]]


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    model_loaded: bool
    model_info: Optional[dict] = None
    ollama_available: bool
    ollama_model: str


# ─────────────────────────────────────────────
# App Lifecycle
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the DistilBERT model and check Ollama connectivity on startup.
    """
    global model, tokenizer, label_names, device, model_loaded, ollama_available

    print("\n" + "=" * 60)
    print("NEWSFORGE BACKEND — Starting up")
    print("=" * 60)

    # ── Load DistilBERT Model ──
    print("\n[1/2] Loading DistilBERT bias classifier...")
    resolved_model_dir = os.path.abspath(MODEL_DIR)

    if os.path.exists(resolved_model_dir):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = DistilBertTokenizer.from_pretrained(resolved_model_dir)
            model = DistilBertForSequenceClassification.from_pretrained(resolved_model_dir)
            model.to(device)
            model.eval()

            # Load label names from metadata
            metadata_path = os.path.join(resolved_model_dir, "newsforge_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                label_names = metadata.get("label_names", ["Left", "Neutral", "Right"])
            else:
                label_names = ["Left", "Neutral", "Right"]

            model_loaded = True
            print(f"       ✓ Model loaded on {device}")
            print(f"       Labels: {label_names}")
        except Exception as e:
            print(f"       ✗ Failed to load model: {e}")
            model_loaded = False
    else:
        print(f"       ✗ Model directory not found: {resolved_model_dir}")
        print(f"         Run 'python model/train.py' first to train the model.")
        # Set up fallback label names
        label_names = ["Left", "Neutral", "Right"]
        model_loaded = False

    # ── Check Ollama Connectivity ──
    print("\n[2/2] Checking Ollama connectivity...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                ollama_available = True
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                print(f"       ✓ Ollama connected — models: {model_names}")

                # Check if Mistral is available
                has_mistral = any("mistral" in name.lower() for name in model_names)
                if not has_mistral:
                    print(f"       ⚠ Mistral not found. Run: ollama pull mistral")
            else:
                ollama_available = False
                print(f"       ✗ Ollama returned status {response.status_code}")
    except Exception as e:
        ollama_available = False
        print(f"       ✗ Ollama not available: {e}")
        print(f"         Start Ollama and run: ollama pull mistral")

    print(f"\n{'=' * 60}")
    print(f"  Ready! Model: {'✓' if model_loaded else '✗'} | Ollama: {'✓' if ollama_available else '✗'}")
    print(f"{'=' * 60}\n")

    yield  # App is running

    # Cleanup
    print("\nShutting down NewsForge backend...")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="NewsForge API",
    description="AI-powered news bias detection and style rewriting",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Bias Classification
# ─────────────────────────────────────────────

def classify_bias(text: str) -> BiasResult:
    """
    Run text through the fine-tuned DistilBERT model.

    Returns:
        BiasResult with label, confidence, and probability distribution
    """
    if not model_loaded:
        # Fallback: return simulated neutral result if model not loaded
        return BiasResult(
            bias_label="Neutral",
            confidence=0.0,
            probabilities={name: 0.2 for name in label_names},
        )

    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0)

    # Extract results
    predicted_idx = torch.argmax(probs).item()
    confidence = probs[predicted_idx].item()

    probabilities = {
        label_names[i]: round(probs[i].item(), 4) for i in range(len(label_names))
    }

    return BiasResult(
        bias_label=label_names[predicted_idx],
        confidence=round(confidence, 4),
        probabilities=probabilities,
    )


# ─────────────────────────────────────────────
# Ollama Rewriting
# ─────────────────────────────────────────────

async def rewrite_with_style(text: str, style_key: str) -> RewriteResult:
    """
    Send text to Ollama Mistral 7B with a style-specific system prompt.

    Args:
        text: The original news text
        style_key: One of 'neutral', 'corporate', 'activist', 'sensationalist', 'state'

    Returns:
        RewriteResult with the rewritten text or error message
    """
    style = STYLE_PROMPTS[style_key]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": style["system"]},
            {"role": "user", "content": f"Rewrite this news text:\n\n{text}"},
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 512,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            rewritten_text = result.get("message", {}).get("content", "").strip()

            return RewriteResult(
                style=style_key,
                name=style["name"],
                text=rewritten_text if rewritten_text else None,
                error="Empty response from Mistral" if not rewritten_text else None,
            )
    except httpx.TimeoutException:
        return RewriteResult(
            style=style_key,
            name=style["name"],
            text=None,
            error="Ollama request timed out (120s). Mistral may still be loading.",
        )
    except Exception as e:
        return RewriteResult(
            style=style_key,
            name=style["name"],
            text=None,
            error=f"Ollama error: {str(e)}",
        )


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Analyze news text for bias and generate 5 style rewrites.

    1. Runs text through DistilBERT → bias label + confidence + probabilities
    2. Sends text to Ollama Mistral 7B with 5 style-specific prompts (in parallel)
    3. Returns combined results

    The bias classification returns instantly; rewrites may take 30-90 seconds
    depending on Mistral's speed.
    """
    text = request.text.strip()

    # Step 1: Classify bias (instant)
    bias_result = classify_bias(text)

    # Step 2: Generate rewrites in parallel via Ollama
    if ollama_available:
        rewrite_tasks = [
            rewrite_with_style(text, style_key)
            for style_key in STYLE_PROMPTS.keys()
        ]
        rewrite_results = await asyncio.gather(*rewrite_tasks)
    else:
        # Ollama not available — return nulls with error
        rewrite_results = [
            RewriteResult(
                style=key,
                name=STYLE_PROMPTS[key]["name"],
                text=None,
                error="Ollama is not available. Start Ollama and pull mistral.",
            )
            for key in STYLE_PROMPTS.keys()
        ]

    # Build response
    rewrites = {}
    rewrite_errors = {}
    for result in rewrite_results:
        rewrites[result.style] = result.text
        rewrite_errors[result.style] = result.error

    return AnalyzeResponse(
        bias=bias_result,
        rewrites=rewrites,
        rewrite_errors=rewrite_errors,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Check the health of the backend services.

    Returns:
        - Model load status and metadata
        - Ollama connectivity and model availability
    """
    # Re-check Ollama connectivity in real-time
    current_ollama_status = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            current_ollama_status = response.status_code == 200
    except Exception:
        current_ollama_status = False

    # Build model info
    model_info = None
    if model_loaded:
        metadata_path = os.path.join(os.path.abspath(MODEL_DIR), "newsforge_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                model_info = json.load(f)

    status = "healthy" if model_loaded and current_ollama_status else "degraded"
    if not model_loaded and not current_ollama_status:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_info=model_info,
        ollama_available=current_ollama_status,
        ollama_model=OLLAMA_MODEL,
    )
