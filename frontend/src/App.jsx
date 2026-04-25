import { useState } from 'react';
import NewspaperBackground from './components/NewspaperBackground';
import InputPanel from './components/InputPanel';
import BiasPanel from './components/BiasPanel';
import RewriteCards from './components/RewriteCards';
import LoadingState from './components/LoadingState';

/**
 * App — NewsForge Main Application
 * 
 * AI-powered news bias detector and rewriter.
 * Sends text to FastAPI backend which runs:
 *   1. DistilBERT for bias classification (instant)
 *   2. Ollama Mistral 7B for 5 editorial style rewrites (30-90s)
 * 
 * State flow:
 *   idle → loading → biasReady → rewritesReady
 */

const API_BASE = 'http://localhost:8000';

function App() {
  // ─── State ───
  const [status, setStatus] = useState('idle'); // idle | loading | done | error
  const [biasResult, setBiasResult] = useState(null);
  const [rewrites, setRewrites] = useState(null);
  const [rewriteErrors, setRewriteErrors] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  // ─── Submit Handler ───
  const handleSubmit = async (text) => {
    setStatus('loading');
    setBiasResult(null);
    setRewrites({});
    setRewriteErrors({});
    setErrorMessage('');

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Server error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split('\n\n');
        buffer = parts.pop(); // Keep incomplete chunk in buffer

        for (const part of parts) {
          if (!part.trim()) continue;

          let eventType = '';
          let eventData = '';
          const lines = part.split('\n');
          for (const line of lines) {
            if (line.startsWith('event:')) eventType = line.slice(6).trim();
            if (line.startsWith('data:')) eventData = line.slice(5).trim();
          }

          if (eventType === 'bias') {
            const data = JSON.parse(eventData);
            setBiasResult(data);
            setStatus('done'); // Transition to done so UI shows bias + empty cards
          } else if (eventType === 'rewrite') {
            const data = JSON.parse(eventData);
            setRewrites((prev) => ({ ...prev, [data.style]: data.text }));
            setRewriteErrors((prev) => ({ ...prev, [data.style]: data.error }));
          } else if (eventType === 'done') {
            // Stream complete
          }
        }
      }
    } catch (err) {
      console.error('Analysis failed:', err);
      setErrorMessage(
        err.message === 'Failed to fetch'
          ? 'Cannot connect to backend. Make sure the FastAPI server is running on port 8000.'
          : err.message
      );
      setStatus('error');
    }
  };

  // ─── Date String ───
  const dateString = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  return (
    <>
      {/* Newspaper collage background with parallax */}
      <NewspaperBackground />

      {/* Main content */}
      <div className="app-container">
        {/* ─── Masthead ─── */}
        <header className="app-masthead">
          <h1 className="app-masthead__title">
            News<span>Forge</span>
          </h1>
          <div className="app-masthead__rule">
            <span className="app-masthead__subtitle">
              AI Bias Detector & Rewriter
            </span>
          </div>
          <p className="app-masthead__edition">
            {dateString} · DistilBERT Classification · Llama 3.2 Rewriting · No API Keys Required
          </p>
        </header>

        {/* ─── Input Panel ─── */}
        <InputPanel
          onSubmit={handleSubmit}
          isLoading={status === 'loading'}
        />

        {/* ─── Error Display ─── */}
        {status === 'error' && (
          <div
            className="bias-panel"
            style={{ marginTop: '24px', borderColor: 'rgba(192, 57, 43, 0.3)' }}
          >
            <div className="news-card__error" style={{ fontSize: '0.85rem' }}>
              ⚠ {errorMessage}
            </div>
          </div>
        )}

        {/* ─── Loading State ─── */}
        {status === 'loading' && (
          <LoadingState biasReady={biasResult !== null} />
        )}

        {/* ─── Bias Panel (appears when results arrive) ─── */}
        {biasResult && status === 'done' && (
          <BiasPanel
            biasLabel={biasResult.bias_label}
            confidence={biasResult.confidence}
            probabilities={biasResult.probabilities}
          />
        )}

        {/* ─── Rewrite Cards ─── */}
        {status === 'done' && (
          <RewriteCards
            rewrites={rewrites}
            rewriteErrors={rewriteErrors}
            originalBias={biasResult?.bias_label}
          />
        )}
      </div>
    </>
  );
}

export default App;
