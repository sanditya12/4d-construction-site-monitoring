import React from 'react';

export interface Session {
  id: string;
  date: string;
  url: string;
}

interface TimelineProps {
  sessions: Session[];
  activeSessionId: string | null;
  onSessionSelect: (id: string) => void;
}

export const Timeline: React.FC<TimelineProps> = ({ sessions, activeSessionId, onSessionSelect }) => {
  if (sessions.length === 0) return null;

  const activeIndex = sessions.findIndex((s) => s.id === activeSessionId);
  const progressPercentage = sessions.length > 1 ? (Math.max(0, activeIndex) / (sessions.length - 1)) * 100 : 0;
  const activeSession = activeIndex >= 0 ? sessions[activeIndex] : null;

  return (
    <div className="timeline-container">
      <div className="timeline-header">
        <span className="timeline-title">Recording Date</span>
        <span className="timeline-date">{activeSession?.date}</span>
      </div>
      <div className="timeline-track">
        <div className="timeline-progress" style={{ width: `${progressPercentage}%` }}></div>
        <div className="timeline-stops">
          {sessions.map((session, index) => (
            <div
              key={session.id}
              className={`timeline-stop ${session.id === activeSessionId ? 'active' : ''}`}
              onClick={() => onSessionSelect(session.id)}
              title={session.date}
              style={{ left: `${(index / (sessions.length - 1)) * 100}%`, transform: 'translateX(-50%)' }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};
