import { useState, useEffect, useRef, useMemo } from 'react';

/**
 * NewspaperBackground
 * 
 * Full-viewport collage of newspaper elements with parallax mouse tracking.
 * Generates randomized fake headlines on every mount, creating the feel of
 * wallpaper in a war journalist's office.
 */

// ─── Headline Banks ───
// Mix of plausible-sounding fake headlines for atmosphere
const HEADLINE_BANKS = [
  // Broadsheet style
  "SENATE APPROVES LANDMARK INFRASTRUCTURE BILL IN LATE-NIGHT SESSION",
  "CENTRAL BANK HOLDS RATES STEADY AMID MOUNTING INFLATION CONCERNS",
  "TRADE NEGOTIATIONS STALL AS DIPLOMATS SEEK COMMON GROUND",
  "COMMITTEE LAUNCHES INVESTIGATION INTO FEDERAL SPENDING OVERSIGHT",
  "NEW CLIMATE ACCORD DRAWS PRAISE AND CRITICISM FROM WORLD LEADERS",
  "HOUSING MARKET SHOWS SIGNS OF COOLING AFTER RECORD QUARTER",
  "SUPREME COURT TO HEAR ARGUMENTS ON DIGITAL PRIVACY RIGHTS",
  "DEFENSE MINISTER OUTLINES NEW STRATEGY FOR REGIONAL SECURITY",
  "GLOBAL SUPPLY CHAINS FACE RENEWED PRESSURE FROM PORT DISRUPTIONS",
  "EDUCATION REFORM PACKAGE CLEARS FIRST LEGISLATIVE HURDLE",
  // Tabloid style
  "EXPERTS SOUND ALARM ON GROWING PENSION CRISIS",
  "WHISTLEBLOWER REVEALS EXTENT OF DATA COLLECTION PROGRAMS",
  "PROTESTS ERUPT IN CAPITAL OVER PROPOSED LABOR CHANGES",
  "TECH GIANTS FACE NEW WAVE OF ANTITRUST SCRUTINY",
  "RURAL COMMUNITIES BRACE FOR IMPACT OF FACTORY CLOSURES",
  "LEAKED MEMO EXPOSES RIFT WITHIN RULING COALITION",
  "ECONOMISTS WARN OF WIDENING WEALTH GAP IN NEW REPORT",
  "HEALTHCARE WORKERS DEMAND BETTER CONDITIONS NATIONWIDE",
  "ENERGY SECTOR SCRAMBLES TO MEET RENEWABLE TARGETS",
  "OPPOSITION LEADER CALLS FOR SNAP ELECTIONS AMID SCANDAL",
  // Wire service style  
  "OFFICIAL: CEASEFIRE AGREEMENT REACHED AFTER 72 HOURS OF TALKS",
  "U.N. REPORT DETAILS HUMANITARIAN SITUATION IN CONFLICT ZONE",
  "IMF RELEASES REVISED GROWTH FORECASTS FOR DEVELOPING ECONOMIES",
  "SOURCE: DIPLOMATIC CHANNELS OPEN BETWEEN RIVAL NATIONS",
];

const FILLER_TEXT = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.";

function NewspaperBackground() {
  const layerRef = useRef(null);
  
  // Generate random newspaper sheets on mount
  const sheets = useMemo(() => {
    const shuffled = [...HEADLINE_BANKS].sort(() => Math.random() - 0.5);
    const items = [];
    
    for (let i = 0; i < 14; i++) {
      const isLarge = Math.random() > 0.6;
      items.push({
        id: i,
        headline: shuffled[i % shuffled.length],
        left: `${Math.random() * 90}%`,
        top: `${Math.random() * 90}%`,
        rotation: (Math.random() - 0.5) * 20,
        width: isLarge ? `${280 + Math.random() * 120}px` : `${180 + Math.random() * 100}px`,
        fontSize: isLarge ? '14px' : '10px',
        zIndex: Math.floor(Math.random() * 5),
      });
    }
    return items;
  }, []);

  // Parallax mouse tracking
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!layerRef.current) return;
      const x = (e.clientX / window.innerWidth - 0.5) * 12;
      const y = (e.clientY / window.innerHeight - 0.5) * 12;
      layerRef.current.style.transform = `translate(${x}px, ${y}px)`;
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="newspaper-bg" aria-hidden="true">
      <div className="newspaper-bg__layer" ref={layerRef}>
        {sheets.map((sheet) => (
          <div
            key={sheet.id}
            className="newspaper-bg__sheet"
            style={{
              left: sheet.left,
              top: sheet.top,
              width: sheet.width,
              transform: `rotate(${sheet.rotation}deg)`,
              zIndex: sheet.zIndex,
            }}
          >
            <div
              className="newspaper-bg__headline"
              style={{ fontSize: sheet.fontSize }}
            >
              {sheet.headline}
            </div>
            <div className="newspaper-bg__text">
              {FILLER_TEXT}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default NewspaperBackground;
