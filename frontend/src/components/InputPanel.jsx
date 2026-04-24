import { useState } from 'react';

/**
 * InputPanel
 * 
 * Telegram/wire dispatch styled input area for submitting news text.
 * Features scanline texture, IBM Plex Mono typography, blinking indicator,
 * and a vintage "TRANSMIT" press button.
 * 
 * Props:
 *   onSubmit(text) — called when user clicks TRANSMIT
 *   isLoading      — disables input during analysis
 */

function InputPanel({ onSubmit, isLoading }) {
  const [text, setText] = useState('');

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (trimmed.length >= 10 && !isLoading) {
      onSubmit(trimmed);
    }
  };

  const handleKeyDown = (e) => {
    // Ctrl/Cmd + Enter to submit
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="input-panel" id="input-panel">
      {/* Header with live indicator */}
      <div className="input-panel__header">
        <div className="input-panel__indicator" />
        <span className="input-panel__label">Wire Dispatch Terminal</span>
      </div>

      {/* Textarea */}
      <textarea
        id="news-input"
        className="input-panel__textarea"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Paste or type a news article, headline, or paragraph to analyze for bias..."
        disabled={isLoading}
        maxLength={5000}
        aria-label="News text input"
      />

      {/* Footer with char count and submit */}
      <div className="input-panel__footer">
        <span className="input-panel__charcount">
          {text.length} / 5,000 chars{text.length > 0 && text.length < 10 ? ' (min 10)' : ''}
        </span>
        <button
          id="transmit-btn"
          className="transmit-btn"
          onClick={handleSubmit}
          disabled={isLoading || text.trim().length < 10}
          aria-label="Analyze text for bias"
        >
          {isLoading ? '⟳ ANALYZING...' : '◆ TRANSMIT'}
        </button>
      </div>
    </div>
  );
}

export default InputPanel;
