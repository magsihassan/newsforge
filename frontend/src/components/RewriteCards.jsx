import { useMemo } from 'react';

/**
 * RewriteCards
 * 
 * Displays 5 newspaper-styled front-page cards, each showing the same news
 * rewritten in a different editorial bias style by Mistral 7B.
 * 
 * Each card has:
 *   - Unique masthead name and styling
 *   - Fake publication date
 *   - The rewritten text in column layout
 *   - A badge showing bias shift from the original detected label
 *   - Staggered entry animation
 * 
 * Props:
 *   rewrites       — object: { neutral, corporate, activist, sensationalist, state }
 *   rewriteErrors  — object: { neutral, corporate, ... } with error strings or null
 *   originalBias   — string: the detected bias label of the original text
 */

// Card metadata for each style
const CARD_CONFIG = {
  neutral: {
    masthead: 'The Neutral Wire',
    cssClass: 'news-card--neutral',
    biasTarget: 'Neutral',
    motto: 'Just the facts.',
  },
  corporate: {
    masthead: 'The Establishment Post',
    cssClass: 'news-card--corporate',
    biasTarget: 'Center-Right',
    motto: 'Markets. Policy. Governance.',
  },
  activist: {
    masthead: "The People's Tribune",
    cssClass: 'news-card--activist',
    biasTarget: 'Left',
    motto: 'Truth to power.',
  },
  sensationalist: {
    masthead: 'Daily Sensation',
    cssClass: 'news-card--sensationalist',
    biasTarget: 'Sensationalist',
    motto: 'YOU WON\'T BELIEVE THIS!',
  },
  state: {
    masthead: 'State Gazette',
    cssClass: 'news-card--state',
    biasTarget: 'Center',
    motto: 'Order. Stability. Progress.',
  },
};

const STYLE_ORDER = ['neutral', 'corporate', 'activist', 'sensationalist', 'state'];

// Bias spectrum positions for calculating shift
const BIAS_POSITION = {
  'Left': 0,
  'Center-Left': 1,
  'Neutral': 2,
  'Center-Right': 3,
  'Right': 4,
  'Sensationalist': -1,
  'Center': 2.5,
};

function generateFakeDate() {
  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];
  const now = new Date();
  return `${months[now.getMonth()]} ${now.getDate()}, ${now.getFullYear()}`;
}

function getBiasShift(originalBias, targetBias) {
  const origPos = BIAS_POSITION[originalBias];
  const targetPos = BIAS_POSITION[targetBias];
  
  if (origPos === undefined || targetPos === undefined || targetPos === -1) {
    return { text: 'STYLE', isSame: false };
  }
  
  const diff = Math.abs(origPos - targetPos);
  if (diff === 0) return { text: 'SAME', isSame: true };
  if (diff <= 1) return { text: `SHIFT ±${diff}`, isSame: false };
  return { text: `SHIFT ±${diff}`, isSame: false };
}

function RewriteCards({ rewrites, rewriteErrors, originalBias }) {
  const pubDate = useMemo(() => generateFakeDate(), []);

  if (!rewrites) return null;

  // Check if any rewrites are available
  const hasAny = STYLE_ORDER.some((key) => rewrites[key]);
  if (!hasAny && !STYLE_ORDER.some((key) => rewriteErrors?.[key])) return null;

  return (
    <div className="rewrite-section" id="rewrite-section">
      <div className="rewrite-section__title">
        Editorial Rewrites by Mistral 7B
      </div>

      <div className="rewrite-grid">
        {STYLE_ORDER.map((styleKey) => {
          const config = CARD_CONFIG[styleKey];
          const text = rewrites[styleKey];
          const error = rewriteErrors?.[styleKey];
          const shift = getBiasShift(originalBias, config.biasTarget);

          return (
            <div
              key={styleKey}
              className={`news-card ${config.cssClass}`}
              id={`card-${styleKey}`}
            >
              {/* Bias Shift Badge */}
              <span
                className={`news-card__badge ${
                  shift.isSame ? 'news-card__badge--same' : 'news-card__badge--shifted'
                }`}
              >
                {shift.text}
              </span>

              {/* Masthead */}
              <div className="news-card__masthead">
                <div className="news-card__masthead-name">
                  {config.masthead}
                </div>
                <div className="news-card__masthead-date">
                  {pubDate} · {config.motto}
                </div>
                <div className="news-card__masthead-rule" />
              </div>

              {/* Body */}
              <div className="news-card__body">
                {text ? (
                  <p className="news-card__text">{text}</p>
                ) : error ? (
                  <div className="news-card__error">
                    ⚠ {error}
                  </div>
                ) : (
                  <p className="news-card__text" style={{ opacity: 0.3 }}>
                    Awaiting rewrite...
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default RewriteCards;
