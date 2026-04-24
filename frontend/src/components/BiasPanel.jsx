import { useMemo } from 'react';

/**
 * BiasPanel
 * 
 * Displays bias classification results from DistilBERT:
 *   - Horizontal spectrum bar (Left → Center → Right) with animated needle
 *   - "SIGNAL STRENGTH" confidence indicator with radio-style bars
 *   - Probability breakdown as newspaper-infographic bar chart
 * 
 * Props:
 *   biasLabel    — string: "Left", "Center-Left", "Neutral", "Center-Right", "Right"
 *   confidence   — float: 0.0 to 1.0
 *   probabilities — object: { "Left": 0.12, "Center-Left": 0.08, ... }
 */

// Label positions on the spectrum bar (as percentage from left)
const LABEL_POSITIONS = {
  'Left': 5,
  'Neutral': 50,
  'Right': 95,
};

const LABEL_ORDER = ['Left', 'Neutral', 'Right'];

function BiasPanel({ biasLabel, confidence, probabilities }) {
  // Calculate needle position on spectrum
  const needlePosition = LABEL_POSITIONS[biasLabel] ?? 50;

  // Signal strength bars (10 bars, filled based on confidence)
  const signalBars = useMemo(() => {
    const totalBars = 10;
    const activeBars = Math.round(confidence * totalBars);
    return Array.from({ length: totalBars }, (_, i) => ({
      height: `${4 + (i + 1) * 2}px`,
      active: i < activeBars,
    }));
  }, [confidence]);

  // Find the highest probability label
  const maxProbLabel = useMemo(() => {
    if (!probabilities) return null;
    return Object.entries(probabilities).reduce(
      (max, [key, val]) => (val > max[1] ? [key, val] : max),
      ['', 0]
    )[0];
  }, [probabilities]);

  return (
    <div className="bias-panel" id="bias-panel">
      {/* Header */}
      <div className="bias-panel__header">
        <span className="bias-panel__header-icon">◈</span>
        <span className="bias-panel__title">Bias Detection Results</span>
      </div>

      {/* ─── Bias Spectrum Bar ─── */}
      <div className="bias-spectrum">
        <div className="bias-spectrum__labels">
          {LABEL_ORDER.map((label) => (
            <span
              key={label}
              className={`bias-spectrum__label ${
                label === biasLabel ? 'bias-spectrum__label--active' : ''
              }`}
            >
              {label}
            </span>
          ))}
        </div>

        <div className="bias-spectrum__bar">
          <div
            className="bias-spectrum__needle"
            style={{ left: `${needlePosition}%` }}
          />
        </div>

        <div className="bias-spectrum__detected">
          <span className="bias-spectrum__detected-label">
            {biasLabel}
          </span>
        </div>
      </div>

      {/* ─── Signal Strength (Confidence) ─── */}
      <div className="signal-strength" id="signal-strength">
        <span className="signal-strength__label">Signal Strength</span>
        <div className="signal-strength__bars">
          {signalBars.map((bar, i) => (
            <div
              key={i}
              className={`signal-strength__bar ${
                bar.active ? 'signal-strength__bar--active' : ''
              }`}
              style={{ height: bar.height }}
            />
          ))}
        </div>
        <span className="signal-strength__value">
          {(confidence * 100).toFixed(1)}%
        </span>
      </div>

      {/* ─── Probability Breakdown ─── */}
      {probabilities && (
        <div className="probability-chart" id="probability-chart">
          <div className="probability-chart__title">
            Probability Distribution
          </div>
          {LABEL_ORDER.map((label) => {
            const value = probabilities[label] ?? 0;
            const isMax = label === maxProbLabel;
            return (
              <div className="probability-chart__row" key={label}>
                <span className="probability-chart__label">{label}</span>
                <div className="probability-chart__bar-track">
                  <div
                    className={`probability-chart__bar-fill ${
                      isMax ? 'probability-chart__bar-fill--active' : ''
                    }`}
                    style={{ width: `${value * 100}%` }}
                  />
                </div>
                <span className="probability-chart__value">
                  {(value * 100).toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default BiasPanel;
