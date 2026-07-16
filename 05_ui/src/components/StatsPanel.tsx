import React from 'react';

interface StatsPanelProps {
  networkPayloadMB: number;
  activeMemoryMB: number;
  activeSplatsCount: number;
}

export const StatsPanel: React.FC<StatsPanelProps> = ({
  networkPayloadMB,
  activeMemoryMB,
  activeSplatsCount
}) => {
  // Determine dot color based on memory thresholds
  let dotColorClass = 'dot-green';
  if (activeMemoryMB >= 2048) {
    dotColorClass = 'dot-red';
  } else if (activeMemoryMB >= 1024) {
    dotColorClass = 'dot-yellow';
  }

  const formatSplats = (count: number) => {
    if (count >= 1000000) {
      return `${(count / 1000000).toFixed(1)}M`;
    } else if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}K`;
    }
    return count.toString();
  };

  return (
    <div className="stats-panel">
      <div className="stat-row">
        <span className="stat-label">Total Download:</span>
        <span className="stat-value">{networkPayloadMB.toFixed(1)} MB</span>
      </div>
      <div className="stat-row">
        <span className="stat-label">Estimated Memory:</span>
        <div className="stat-value-container">
          <span className={`indicator-dot ${dotColorClass}`} />
          <span className="stat-value">~{Math.round(activeMemoryMB)} MB ({formatSplats(activeSplatsCount)} Splats)</span>
        </div>
      </div>
    </div>
  );
};
