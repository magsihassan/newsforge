import { useState, useEffect } from 'react';

/**
 * LoadingState
 * 
 * Animated "PRESS PRINTING" sequence displayed while Mistral generates rewrites.
 * Features expanding ink blot animation and sequential step reveals.
 * 
 * Props:
 *   biasReady — boolean: whether bias classification is complete (instant)
 */

const LOADING_STEPS = [
  { text: 'CLASSIFYING BIAS SIGNAL...', delay: 0 },
  { text: 'BIAS DETECTED — LOCKING FREQUENCY', delay: 800 },
  { text: 'DISPATCHING TO EDITORIAL DESKS...', delay: 1500 },
  { text: 'NEUTRAL WIRE — COMPOSING...', delay: 2500 },
  { text: 'ESTABLISHMENT POST — DRAFTING...', delay: 3500 },
  { text: "PEOPLE'S TRIBUNE — WRITING...", delay: 4500 },
  { text: 'DAILY SENSATION — PRINTING...', delay: 5500 },
  { text: 'STATE GAZETTE — REVIEWING...', delay: 6500 },
  { text: 'PRESS RUNNING — STAND BY', delay: 7500 },
];

function LoadingState({ biasReady }) {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const timers = LOADING_STEPS.map((step, index) =>
      setTimeout(() => setActiveStep(index), step.delay)
    );

    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <div className="loading-overlay" id="loading-state">
      {/* Title */}
      <div className="loading-overlay__title">
        ◆ PRESS PRINTING ◆
      </div>

      {/* Ink Blot Animation */}
      <div className="ink-blot">
        <div className="ink-blot__circle" />
        <div className="ink-blot__circle" />
        <div className="ink-blot__circle" />
      </div>

      {/* Step Progress */}
      <div className="loading-overlay__steps">
        {LOADING_STEPS.map((step, index) => (
          <div
            key={index}
            className={`loading-overlay__step ${
              index < activeStep
                ? 'loading-overlay__step--done'
                : index === activeStep
                ? 'loading-overlay__step--active'
                : ''
            }`}
          >
            {index < activeStep ? '✓' : index === activeStep ? '▸' : '·'}{' '}
            {step.text}
          </div>
        ))}
      </div>
    </div>
  );
}

export default LoadingState;
