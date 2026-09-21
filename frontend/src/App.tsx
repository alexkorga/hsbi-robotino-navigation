/** Coordinate backend state, selected Robotino, and the dashboard panels. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  controlSimulation, driveToLabel, loadBackend, stopRobotino, subscribeToWorld,
} from "./api/backendFleet";
import { EventFeed } from "./components/EventFeed";
import { FleetSidebar } from "./components/FleetSidebar";
import { MapToolbar } from "./components/MapToolbar";
import { RobotInspector } from "./components/RobotInspector";
import { TopBar } from "./components/TopBar";
import { FactoryMap } from "./scene/FactoryMap";
import type {
  FleetEvent, HoverDetail, LayerVisibility, MapFixture, WorldState,
} from "./types/fleet";

const INITIAL_LAYERS: LayerVisibility = {
  occupancyMap: true,
  movementArea: true,
  staticObstacles: true,
  labels: true,
  footprints: true,
  trajectories: true,
  sensors: true,
  lidar: true,
  temporaryObstacles: true,
};

/** Own backend connection, selection, commands, and dashboard panels; return the page UI. */
export function App() {
  const [fixture, setFixture] = useState<MapFixture | null>(null);
  const [world, setWorld] = useState<WorldState | null>(null);
  const [selectedRobotId, setSelectedRobotId] = useState<string | null>(null);
  const [selectedLocationId, setSelectedLocationId] = useState<string | null>(null);
  const [layers, setLayers] = useState(INITIAL_LAYERS);
  const [connected, setConnected] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [fitSignal, setFitSignal] = useState(0);
  const [hoverDetail, setHoverDetail] = useState<HoverDetail | null>(null);
  const localEvent = useRef(1_000_000);

  /** Apply a new `next` snapshot while preserving a still-present robot selection. */
  const applyWorld = useCallback((next: WorldState) => {
    setWorld(next);
    setSelectedRobotId((current) =>
      current && next.robots.some((robot) => robot.id === current)
        ? current
        : next.robots[0]?.id ?? null
    );
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    /** Load initial map/world data, retrying after backend connection failures. */
    const connect = async () => {
      try {
        const loaded = await loadBackend();
        if (cancelled) return;
        setFixture(loaded.fixture);
        applyWorld(loaded.world);
        setBackendError(null);
      } catch (error) {
        if (cancelled) return;
        setBackendError(error instanceof Error ? error.message : "Backend unavailable");
        timer = window.setTimeout(connect, 1000);
      }
    };
    void connect();
    return () => { cancelled = true; if (timer) window.clearTimeout(timer); };
  }, [applyWorld]);

  useEffect(() => {
    if (!fixture) return;
    let cancelled = false;
    let socket: WebSocket | undefined;
    let timer: number | undefined;
    /** Subscribe to world updates and retry when this WebSocket closes. */
    const connect = () => {
      if (cancelled) return;
      socket = subscribeToWorld(
        (next) => { setConnected(true); setBackendError(null); applyWorld(next); },
        () => { setConnected(false); if (!cancelled) timer = window.setTimeout(connect, 1000); },
      );
    };
    connect();
    return () => { cancelled = true; socket?.close(); if (timer) window.clearTimeout(timer); };
  }, [fixture, applyWorld]);

  const selectedRobot = useMemo(
    () => world?.robots.find((robot) => robot.id === selectedRobotId) ?? world?.robots[0],
    [world, selectedRobotId],
  );

  /** Add a local event from `message` without changing backend fleet state. */
  const addError = (message: string) => {
    if (!world) return;
    const event: FleetEvent = {
      id: localEvent.current++,
      time: "--:--",
      robotId: selectedRobot?.id ?? null,
      type: "ERROR",
      message,
    };
    setWorld({ ...world, events: [event, ...world.events] });
  };

  /** Run `action` only while connected and apply its returned world snapshot. */
  const command = async (action: () => Promise<WorldState>) => {
    if (!connected) return addError("Backend world stream is unavailable");
    try { applyWorld(await action()); }
    catch (error) { addError(error instanceof Error ? error.message : "Command failed"); }
  };

  if (!fixture || !world || !selectedRobot) {
    return <main className="loading-screen">
      <div className="loading-mark" /><p className="eyebrow">Robotino Fleet Manager</p>
      <h1>Waiting for the fleet backend…</h1>
      <p>Run <code>uv run python run_simulation.py</code>. This page retries automatically.</p>
      {backendError && <p className="backend-error">{backendError}</p>}
    </main>;
  }

  return <div className="app-shell">
    <TopBar
      paused={world.paused}
      elapsedS={world.clockS}
      mode={world.mode}
      connected={connected}
      onTogglePause={() => void command(() => controlSimulation(world.paused ? "resume" : "pause"))}
      onReset={() => void command(async () => {
        const next = await controlSimulation("reset");
        setSelectedLocationId(null);
        setFitSignal((value) => value + 1);
        return next;
      })}
    />
    <div className="dashboard-grid">
      <FleetSidebar robots={world.robots} selectedRobotId={selectedRobot.id} onSelectRobot={setSelectedRobotId} />
      <main className="map-workspace">
        <div className="map-titlebar"><div><p className="eyebrow">Shared world model</p><h2>{fixture.map.name}</h2></div>
          <div className="map-facts">
            <span><strong>{fixture.locations.length}</strong> labels</span>
            <span><strong>{fixture.environmentGeometry.obstacles.length}</strong> obstacles</span>
            <span><strong>{fixture.map.resolutionMPerPx.toFixed(2)}</strong> m/px</span>
          </div>
        </div>
        <div className="map-stage">
          <FactoryMap
            fixture={fixture}
            robots={world.robots}
            goals={world.goals}
            temporaryObstacles={world.temporaryObstacles}
            selectedRobotId={selectedRobot.id}
            selectedLocationId={selectedLocationId}
            layers={layers}
            fitSignal={fitSignal}
            onSelectRobot={setSelectedRobotId}
            onSelectLocation={setSelectedLocationId}
            onHoverDetail={setHoverDetail}
          />
          <MapToolbar
            layers={layers}
            hoverDetail={hoverDetail}
            onFit={() => setFitSignal((value) => value + 1)}
            onToggleLayer={(layer) => setLayers((current) => ({ ...current, [layer]: !current[layer] }))}
          />
        </div>
      </main>
      <div className="right-rail">
        <RobotInspector
          robot={selectedRobot}
          locations={fixture.locations}
          selectedLocationId={selectedLocationId}
          commandsEnabled={connected}
          physicalCommandsEnabled={world.physicalCommandsEnabled}
          onSelectLocation={setSelectedLocationId}
          onDrive={() => selectedLocationId && void command(() => driveToLabel(selectedRobot.id, selectedLocationId))}
          onStop={() => void command(() => stopRobotino(selectedRobot.id))}
        />
        <EventFeed events={world.events} />
      </div>
    </div>
  </div>;
}
