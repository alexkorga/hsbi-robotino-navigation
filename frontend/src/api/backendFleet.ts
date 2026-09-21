/** Fetch fleet snapshots and send operator actions to the backend. */

import type { MapFixture, WorldState } from "../types/fleet";

export const FLEET_API_BASE = (
  import.meta.env.VITE_FLEET_API_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

/** Fetch JSON of type `T` from `path`, applying optional `init`; throw backend errors. */
async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${FLEET_API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { detail?: string };
    throw new Error(body.detail ?? `Fleet backend returned HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

/** Load the static map and initial world concurrently, returning both UI contracts. */
export async function loadBackend(): Promise<{ fixture: MapFixture; world: WorldState }> {
  const [fixture, world] = await Promise.all([
    requestJson<MapFixture>("/api/map"),
    requestJson<WorldState>("/api/world"),
  ]);
  if (fixture.map.imageUrl.startsWith("/")) {
    fixture.map.imageUrl = `${FLEET_API_BASE}${fixture.map.imageUrl}`;
  }
  return { fixture, world };
}

/** Open a world stream; call `onWorld` for snapshots and `onClosed` on disconnect. Return its socket. */
export function subscribeToWorld(
  onWorld: (world: WorldState) => void,
  onClosed: () => void,
) {
  const socket = new WebSocket(`${FLEET_API_BASE.replace(/^http/, "ws")}/api/world/stream`);
  socket.onmessage = (event) => onWorld(JSON.parse(event.data) as WorldState);
  socket.onerror = () => socket.close();
  socket.onclose = onClosed;
  return socket;
}

/** Assign `locationId` to `robotinoIp` and return the updated backend world. */
export function driveToLabel(robotinoIp: string, locationId: string) {
  return requestJson<WorldState>(`/api/robotinos/${robotinoIp}/goal`, {
    method: "POST",
    body: JSON.stringify({ locationId }),
  });
}
/** Cancel `robotinoIp`'s order and return the updated backend world. */
export function stopRobotino(robotinoIp: string) {
  return requestJson<WorldState>(`/api/robotinos/${robotinoIp}/stop`, { method: "POST" });
}
/** Apply simulation `action` and return the updated backend world. */
export function controlSimulation(action: "reset" | "pause" | "resume") {
  return requestJson<WorldState>(`/api/simulation/${action}`, { method: "POST" });
}
