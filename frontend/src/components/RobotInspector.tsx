/** Show the selected Robotino's status, telemetry, and goal controls. */

import type { CSSProperties } from "react";
import { SENSOR_MAX_RANGE_M } from "../config/robotino";
import type { MapLocation, Robotino } from "../types/fleet";

/** Return a clockwise proximity-sensor diagram for `robot`, or nothing when readings are absent. */
function SensorDiagram({ robot }: { robot: Robotino }) {
  const nearest = robot.sensors.reduce(
    (best, value) => value.rangeM < best.rangeM ? value : best,
    robot.sensors[0],
  );
  if (!nearest) return null;
  return (
    <div className="sensor-block">
      <div className="sensor-heading">
        <div><p className="eyebrow">Proximity</p><h3>Distance sensors</h3></div>
        <span>{`S${nearest.index} · ${nearest.rangeM.toFixed(3)} m`}</span>
      </div>
      <div className="sensor-orbit">
        {robot.sensors.map((sensor) => (
          <span
            key={sensor.index}
            className={`sensor-ray${sensor.rangeM >= SENSOR_MAX_RANGE_M - 0.005 ? " is-clear" : " is-hit"}`}
            style={{
              transform: `rotate(${sensor.index * 40}deg)`,
              "--sensor-length": `${24 + sensor.rangeM / SENSOR_MAX_RANGE_M * 36}px`,
            } as CSSProperties}
          ><i>{sensor.index}</i></span>
        ))}
        <span className="sensor-robot"><i /></span>
      </div>
    </div>
  );
}

/**
 * Return telemetry and goal controls for `robot` and available `locations`.
 * `selectedLocationId` controls the chosen label, `commandsEnabled` gates
 * actions during disconnection, and `physicalCommandsEnabled` labels live
 * output. `onSelectLocation`, `onDrive`, and `onStop` dispatch operator input.
 */
export function RobotInspector({
  robot,
  locations,
  selectedLocationId,
  commandsEnabled,
  physicalCommandsEnabled,
  onSelectLocation,
  onDrive,
  onStop,
}: {
  robot: Robotino;
  locations: MapLocation[];
  selectedLocationId: string | null;
  commandsEnabled: boolean;
  physicalCommandsEnabled: boolean;
  onSelectLocation: (id: string) => void;
  onDrive: () => void;
  onStop: () => void;
}) {
  const pathEfficiency = robot.pathEfficiency == null
    ? (robot.status === "NAVIGATING" || robot.status === "WAITING" ? "MEASURING" : "—")
    : `${(robot.pathEfficiency * 100).toFixed(1)}%`;
  const pathDistances = robot.straightPathLengthM == null
    ? "no completed assignment"
    : `${robot.straightPathLengthM.toFixed(2)} m straight / ${robot.drivenPathLengthM.toFixed(2)} m driven`;
  const travelTime = robot.travelTimeS == null
    ? (robot.status === "NAVIGATING" || robot.status === "WAITING" ? "MEASURING" : "—")
    : `${robot.travelTimeS.toFixed(1)} s`;
  return (
    <aside className="robot-inspector" aria-label={`Inspect ${robot.ip}`}>
      <div className="inspector-identity">
        <div className="inspector-avatar" style={{ "--robot-color": robot.color } as CSSProperties}>
          {robot.ip.split(".").at(-1)}
        </div>
        <div><p className="eyebrow">Selected Robotino</p><h2>{robot.ip}</h2>
          <span className={`status-label status-${robot.status.toLowerCase()}`}>{robot.status}</span>
        </div>
      </div>
      {robot.stopReason && <div className="fault-callout"><span>Motion stopped</span><p>{robot.stopReason}</p></div>}
      <div className="telemetry-grid">
        <div><span>Position</span><strong>{robot.x.toFixed(2)}, {robot.y.toFixed(2)}</strong><small>metres · map frame</small></div>
        <div><span>Heading</span><strong>{((robot.headingRad * 180) / Math.PI).toFixed(1)}°</strong><small>counter-clockwise</small></div>
        <div><span>Speed</span><strong>{robot.speedMps.toFixed(2)}</strong><small>m/s measured</small></div>
        <div><span>Battery</span><strong>{robot.batteryLow ? "LOW" : "OK"}</strong><small>batteryLow</small></div>
      </div>
      <div className="execution-grid">
        <div><span>Map pose</span><strong>{robot.poseValid ? "VALID" : "UNAVAILABLE"}</strong><small>{robot.poseSource} · {robot.poseAgeS.toFixed(2)} s</small></div>
        <div><span>Controller</span><strong>{robot.policyId ?? "direct baseline"}</strong><small>{robot.policyInferenceMs == null ? "no inference timing" : `${robot.policyInferenceMs.toFixed(2)} ms`}</small></div>
        <div><span>LiDAR</span><strong>{robot.lidar ? `${robot.lidar.validCount}/${robot.lidar.sampleCount}` : "NO SCAN"}</strong><small>{robot.lidar ? `${robot.lidar.ageS.toFixed(2)} s old` : "scan0 unavailable"}</small></div>
        <div><span>Output</span><strong>{physicalCommandsEnabled ? "LIVE" : "NO TX"}</strong><small>{physicalCommandsEnabled ? (robot.commandQueued ? "last command queued" : "no command queued") : "simulation / shadow"}</small></div>
        <div><span>Path efficiency</span><strong>{pathEfficiency}</strong><small>{pathDistances}</small></div>
        <div><span>Travel time</span><strong>{travelTime}</strong><small>assignment to ready to dock</small></div>
      </div>
      <section className="command-monitor">
        <p className="eyebrow">Velocity pipeline</p>
        <div><span>Proposed</span><code>{robot.proposedCommand.vx.toFixed(3)} / {robot.proposedCommand.vy.toFixed(3)} / {robot.proposedCommand.omega.toFixed(3)}</code></div>
        <div><span>Applied</span><code>{robot.appliedCommand.vx.toFixed(3)} / {robot.appliedCommand.vy.toFixed(3)} / {robot.appliedCommand.omega.toFixed(3)}</code></div>
        <small>vx / vy / ω</small>
      </section>
      <section className="route-control">
        <div className="control-heading"><div><p className="eyebrow">Assignment</p><h3>Machine label</h3></div></div>
        <select value={selectedLocationId ?? ""} onChange={(event) => onSelectLocation(event.target.value)}>
          <option value="" disabled>Select a label…</option>
          {locations.map((location) => <option key={location.id} value={location.id}>Label {location.id}</option>)}
        </select>
        <div className="control-actions">
          <button className="button button-primary" disabled={!commandsEnabled || !selectedLocationId} onClick={onDrive}>Drive to label</button>
          <button className="button button-danger" disabled={!commandsEnabled} onClick={onStop}>Stop</button>
        </div>
      </section>
      <SensorDiagram robot={robot} />
    </aside>
  );
}
