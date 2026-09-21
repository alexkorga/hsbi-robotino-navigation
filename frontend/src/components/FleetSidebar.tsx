/** List Robotinos and aggregate fleet status for selection. */

import type { CSSProperties } from "react";
import type { Robotino } from "../types/fleet";

/** Return the fleet list for `robots`, highlighting `selectedRobotId` and reporting clicks via `onSelectRobot`. */
export function FleetSidebar({
  robots,
  selectedRobotId,
  onSelectRobot,
}: {
  robots: Robotino[];
  selectedRobotId: string;
  onSelectRobot: (id: string) => void;
}) {
  const navigating = robots.filter((robot) => robot.status === "NAVIGATING").length;
  const online = robots.filter((robot) => robot.online).length;
  return (
    <aside className="fleet-sidebar" aria-label="Robotino fleet">
      <div className="section-heading">
        <div><p className="eyebrow">Fleet</p><h2>{robots.length} Robotinos</h2></div>
        <span className="fleet-online">{online}/{robots.length} online</span>
      </div>
      <div className="fleet-summary">
        <div><strong>{navigating}</strong><span>Navigating</span></div>
        <div><strong>{robots.filter((r) => r.status === "WAITING").length}</strong><span>Waiting</span></div>
        <div><strong>{robots.length - navigating}</strong><span>Available</span></div>
      </div>
      <div className="robot-list">
        {robots.map((robot) => (
          <button
            type="button"
            key={robot.id}
            className={`robot-list-item${selectedRobotId === robot.id ? " is-selected" : ""}`}
            onClick={() => onSelectRobot(robot.id)}
          >
            <span
              className="robot-avatar"
              style={{ "--robot-color": robot.color } as CSSProperties}
            >{robot.ip.split(".").at(-1)}</span>
            <span className="robot-list-copy">
              <span className="robot-list-title">
                <strong>{robot.ip}</strong>
                <span className={`status-dot status-${robot.status.toLowerCase()}`} />
              </span>
              <span className="robot-list-meta">
                {robot.goalLocationId ? `To label ${robot.goalLocationId}` : robot.stopReason ?? "Ready"}
              </span>
            </span>
            <span className="robot-list-status">{robot.status}</span>
          </button>
        ))}
      </div>
      <div className="sidebar-note">
        <span className="note-index">01</span>
        <p>Robotino IP addresses are the fleet identities in simulation and in the lab.</p>
      </div>
    </aside>
  );
}
