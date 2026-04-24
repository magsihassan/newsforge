# NewsForge — AI-Powered News Bias Detector & Rewriter

> **Detect political bias in news text using a custom fine-tuned DistilBERT model, then see how the same story reads through 5 distinct editorial lenses — powered by Mistral 7B running locally via Ollama. No API keys required.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![React](https://img.shields.io/badge/React-18-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688)
![Ollama](https://img.shields.io/badge/Ollama-Mistral_7B-black)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Use Cases](#use-cases)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Setup & Installation](#setup--installation)
- [Dataset: BABE](#dataset-babe)
- [Model Training](#model-training)
- [Training Results](#training-results)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Limitations & Ethics](#limitations--ethics)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

NewsForge is a full-stack media literacy tool that combines two AI systems to analyze and transform news text:

1. **Bias Classifier** — A fine-tuned DistilBERT model trained on the BABE (Bias Annotations By Experts) dataset that classifies news sentences across 5 political bias categories: Left, Center-Left, Neutral, Center-Right, and Right.

2. **Style Rewriter** — Ollama running Mistral 7B locally with 5 carefully engineered editorial personas, each rewriting the input text according to distinct journalistic conventions.

Everything runs locally. No cloud APIs, no subscription keys, no data sent to external servers.

---

## Use Cases

- **Media Literacy Education** — Help students understand how framing, word choice, and perspective shape news coverage
- **Journalism Training** — Demonstrate how the same facts can be presented through different editorial lenses
- **Misinformation Research** — Study how bias manifests in news language at the sentence level
- **Content Analysis** — Quickly assess the political leaning of news passages
- **Writing Self-Check** — Journalists can check their own copy for unintentional bias

---

## Architecture

```
┌──────────────────────┐
│    React Frontend     │
│    (Vite, port 5173)  │
│                       │
│  ┌─────────────────┐  │
│  │  Input Panel     │  │      POST /analyze
│  │  Bias Panel      │──────────────────────────┐
│  │  Rewrite Cards   │  │                       │
│  └─────────────────┘  │                       ▼
└──────────────────────┘          ┌──────────────────────────┐
                                  │     FastAPI Backend       │
                                  │     (port 8000)           │
                                  │                           │
                                  │  ┌─────────────────────┐  │
                                  │  │  DistilBERT Model    │  │
                                  │  │  (saved_model/)      │  │
                                  │  │  → Bias Label        │  │
                                  │  │  → Confidence        │  │
                                  │  │  → 5-Class Probs     │  │
                                  │  └─────────────────────┘  │
                                  │                           │
                                  │  ┌─────────────────────┐  │
                                  │  │  Ollama / Mistral    │  │
                                  │  │  (localhost:11434)   │  │
                                  │  │  → 5 Rewrites        │  │
                                  │  │    (parallel)        │  │
                                  │  └─────────────────────┘  │
                                  └──────────────────────────┘
```

**Data Flow:**
1. User pastes news text into the Wire Dispatch Terminal
2. Frontend sends POST request to `/analyze`
3. Backend runs DistilBERT inference → returns bias classification instantly
4. Backend sends 5 parallel requests to Ollama with style-specific prompts
5. All 5 rewrites return and display as newspaper-styled cards

---

## How It Works

### Bias Classification (DistilBERT)

The model classifies text into 5 categories using a composite label derived from the BABE dataset:

| Label | Description | Source Mapping |
|-------|-------------|----------------|
| **Left** | Strong progressive/liberal framing | Biased text from left-leaning outlets |
| **Center-Left** | Mild progressive lean with editorial tone | Biased text from center outlets (opinion-heavy) |
| **Neutral** | Factual, balanced reporting | Non-biased text from any outlet |
| **Center-Right** | Mild conservative/establishment lean | Biased text from center outlets (factual-leaning) |
| **Right** | Strong conservative framing | Biased text from right-leaning outlets |

### Style Rewriting (Mistral 7B)

Each editorial voice has a dedicated system prompt reflecting real journalistic conventions:

| Style | Masthead | Characteristics |
|-------|----------|----------------|
| **Neutral** | The Neutral Wire | Reuters-style, passive voice, attribution only, zero adjectives |
| **Corporate** | The Establishment Post | Economic framing, stakeholder language, pro-institution |
| **Activist** | The People's Tribune | Systemic framing, communities centered, power structures challenged |
| **Sensationalist** | Daily Sensation | CAPS, urgency, emotional language, clickbait energy |
| **State** | State Gazette | Passive deflection, institutional framing, government as stabilizer |

---

## Setup & Installation

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **Ollama** — [Download here](https://ollama.com)
- **~6GB disk space** for Mistral 7B model
- **GPU recommended** for model training (CPU works but slower)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/newsforge.git
cd newsforge
```

### Step 2: Train the Model

```bash
cd model
pip install -r requirements.txt
python train.py
```

This will:
- Download the BABE dataset from HuggingFace (~948 KB)
- Fine-tune DistilBERT for ~10 epochs
- Save the model to `model/saved_model/`
- Print training metrics

**Training time:** ~10-15 min on GPU, ~1-2 hours on CPU

### Step 3: Set Up Ollama

```bash
# Install Ollama from https://ollama.com
ollama pull mistral
```

Verify it's running:
```bash
curl http://localhost:11434/api/tags
```

### Step 4: Start the Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The backend will load the DistilBERT model and check Ollama connectivity on startup.

### Step 5: Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Dataset: BABE

This project uses the **BABE (Bias Annotations By Experts)** dataset:

- **Source:** [mediabiasgroup/BABE on HuggingFace](https://huggingface.co/datasets/mediabiasgroup/BABE)
- **Size:** 4,121 sentences from US news articles
- **Annotation:** Expert-annotated for sentence-level bias
- **Topics:** Vaccine, elections, immigration, and other controversial subjects
- **Labels:** Binary bias labels + outlet political orientation
- **License:** Research use

### Citation

If you use the BABE dataset, please cite:

```
@inproceedings{spinde2021neural,
  title={Neural Media Bias Detection Using Distant Supervision With BABE},
  author={Spinde, Timo and Plank, Manuel and Krieger, Jan-David and Ruas, Terry and Gipp, Bela and Aizawa, Akiko},
  booktitle={Findings of EACL},
  year={2021}
}
```

---

## Model Training

The training pipeline (`model/train.py`) performs:

1. **Data Loading** — Downloads BABE from HuggingFace Datasets hub
2. **Label Mapping** — Creates 5 composite classes from binary bias + outlet type + opinion labels
3. **Tokenization** — DistilBERT tokenizer with max_length=256
4. **Stratified Splitting** — 80/10/10 train/val/test with class balance preservation
5. **Fine-Tuning** — 10 epochs with:
   - Class-weighted CrossEntropyLoss (handles Neutral class dominance)
   - AdamW optimizer (lr=2e-5, weight_decay=0.01)
   - Linear warmup scheduler (10% of total steps)
   - Gradient clipping (max_norm=1.0)
6. **Early Saving** — Best model checkpoint saved based on validation accuracy

### Evaluation

Run `python model/evaluate.py` after training to see:
- Overall accuracy and macro/weighted F1 scores  
- Per-class precision, recall, and F1
- Full confusion matrix
- Average confidence per predicted class

---

## Training Results

Expected approximate metrics (may vary with random seed):

| Metric | Value |
|--------|-------|
| Test Accuracy | ~72-78% |
| Macro F1 | ~0.55-0.65 |
| Weighted F1 | ~0.72-0.78 |

**Note:** The 5-class task is inherently challenging because:
- The BABE dataset was designed for binary (biased/non-biased) classification
- Our 5-class mapping adds nuance but has class imbalance
- Some bias categories have very few examples
- Sentence-level bias is highly context-dependent

The model performs best at distinguishing Neutral from Biased text, and reasonably well at Left vs Right distinctions.

---

## API Documentation

### POST /analyze

Analyze news text for bias and generate 5 editorial rewrites.

**Request:**
```json
{
  "text": "The radical agenda is threatening the very foundations of our economy."
}
```

**Response:**
```json
{
  "bias": {
    "bias_label": "Right",
    "confidence": 0.7823,
    "probabilities": {
      "Left": 0.0412,
      "Center-Left": 0.0234,
      "Neutral": 0.0891,
      "Center-Right": 0.064,
      "Right": 0.7823
    }
  },
  "rewrites": {
    "neutral": "Policy proposals face scrutiny from economic analysts...",
    "corporate": "Market stakeholders are evaluating the potential fiscal impact...",
    "activist": "Communities are pushing back against policies that deepen inequality...",
    "sensationalist": "SHOCKING new agenda THREATENS to DESTROY the economy!",
    "state": "Authorities are working to ensure economic stability..."
  },
  "rewrite_errors": {
    "neutral": null,
    "corporate": null,
    "activist": null,
    "sensationalist": null,
    "state": null
  }
}
```

### GET /health

Check backend service health.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": { "test_accuracy": 0.75, "best_epoch": 7 },
  "ollama_available": true,
  "ollama_model": "mistral"
}
```

---

## Limitations & Ethics

### Technical Limitations

- **Sentence-Level Only** — The model analyzes individual sentences or short paragraphs. It cannot assess bias across an entire article's structure, sourcing, or framing arc.
- **U.S. News Focus** — Trained on US news outlets; may not generalize well to international or non-English media.
- **Context Blindness** — Cannot understand sarcasm, irony, or context-dependent bias.
- **Small Dataset** — 4,121 samples is relatively small for NLP; expect some misclassifications.
- **Class Imbalance** — Center-Left and Center-Right classes have fewer training examples.

### Social Media

- This tool is **not designed for social media posts**, tweets, or informal text. Social media has different bias patterns than professional journalism.
- **Do not use this tool to label individuals or organizations** as biased — it analyzes text, not people.

### Ethical Considerations

- Bias is a spectrum, not a binary. The 5 categories are simplifications.
- The rewrites are AI-generated demonstrations, not ground truth of how biased outlets would actually cover a story.
- This tool is for **education and research** — not for censorship, content moderation, or automated news filtering.
- Mistral 7B's rewrites may contain hallucinated facts — they demonstrate style, not accuracy.

---

## Project Structure

```
newsforge/
├── model/
│   ├── train.py              # Full training pipeline
│   ├── evaluate.py           # Evaluation metrics & confusion matrix
│   ├── requirements.txt      # Python dependencies
│   └── saved_model/          # ← Generated after training
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       ├── special_tokens_map.json
│       ├── newsforge_metadata.json
│       └── test_data.json
├── backend/
│   ├── main.py               # FastAPI server
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── index.html            # Entry point with fonts & SEO
│   ├── package.json          # Node dependencies
│   ├── vite.config.js        # Vite configuration
│   └── src/
│       ├── main.jsx          # React entry
│       ├── App.jsx           # Main app with state management
│       ├── index.css         # Full design system
│       └── components/
│           ├── NewspaperBackground.jsx  # Parallax newspaper collage
│           ├── InputPanel.jsx           # Wire dispatch input
│           ├── BiasPanel.jsx            # Spectrum + signal + chart
│           ├── RewriteCards.jsx         # 5 newspaper-style cards
│           └── LoadingState.jsx         # Press printing animation
└── README.md                 # This file
```

---

## License

This project is for educational and research purposes. The BABE dataset is provided by the Media Bias Group at the University of Göttingen.

---

*Built with DistilBERT, Mistral 7B, FastAPI, React, and a deep appreciation for the Fourth Estate.*
