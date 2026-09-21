/** Show the current fleet mode, clock, and connection state. */

/** Return elapsed `seconds` as an MM:SS display value. */
function clock(seconds: number) {
  return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

/**
 * Return fleet status using `paused`, `elapsedS`, `mode`, and `connected`.
 * `onTogglePause` and `onReset` dispatch simulation-only operator actions.
 */
export function TopBar({
  paused, elapsedS, mode, connected, onTogglePause, onReset,
}: {
  paused: boolean;
  elapsedS: number;
  mode: "simulation" | "shadow" | "live";
  connected: boolean;
  onTogglePause: () => void;
  onReset: () => void;
}) {
  return <header className="topbar">
    <div className="brand-lockup"><div className="brand-mark"><span /><span /><span /></div>
      <div><p className="eyebrow">Group 2 · IoT Factory</p><h1>Robotino Fleet Manager</h1></div>
    </div>
    <div className="topbar-status">
      <span className="mode-badge">{mode === "simulation" ? "SIMULATION" : mode === "shadow" ? "SHADOW · NO TX" : "LIVE"}</span>
      <span className="connection-state"><span className={`connection-dot${connected ? "" : " is-offline"}`} />{connected ? "Backend WebSocket" : "Reconnecting"}</span>
      <span className="simulation-clock">T+ {clock(elapsedS)}</span>
      <button className="button button-secondary" disabled={!connected || mode !== "simulation"} onClick={onTogglePause}>{paused ? "Resume" : "Pause"}</button>
      <button className="button button-ghost" disabled={!connected || mode !== "simulation"} onClick={onReset}>Reset simulation</button>
    </div>
  </header>;
}
