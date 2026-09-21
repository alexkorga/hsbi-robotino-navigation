/** Verify backend availability and API request behavior in the dashboard. */

import { afterEach, describe, expect, it, vi } from "vitest";
import { FLEET_API_BASE, driveToLabel, loadBackend } from "./backendFleet";

/** Return an HTTP JSON response containing `value` with optional `status`. */
function jsonResponse(value: unknown, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("backend fleet client", () => {
  it("loads static and dynamic contracts and resolves the map image URL", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse({
          schemaVersion: 1,
          map: { imageUrl: "/assets/iot_factory_map.png" },
          paths: [],
          graph: { nodes: [], edges: [] },
          locations: [],
          stations: [],
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          schemaVersion: 1,
          sequence: 4,
          clockS: 1.2,
          paused: false,
          mode: "simulation",
          physicalCommandsEnabled: false,
          robots: [],
          routes: [],
          events: [],
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    const result = await loadBackend();

    expect(result.world.sequence).toBe(4);
    expect(result.fixture.map.imageUrl).toBe(`${FLEET_API_BASE}/assets/iot_factory_map.png`);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("sends waypoint commands to the backend rather than planning in the browser", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        robotId: "172.21.20.90",
        destinationLocationId: "13",
        points: [],
        resourceIds: [],
        lengthM: 0,
        estimatedDurationS: 0,
        progressM: 0,
        state: "PREVIEW",
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await driveToLabel("172.21.20.90", "13");

    const request = fetchMock.mock.calls[0][1] as RequestInit;

    expect(fetchMock).toHaveBeenCalledWith(
      `${FLEET_API_BASE}/api/robotinos/172.21.20.90/goal`,
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ locationId: "13" }),
      }),
    );
    expect(new Headers(request.headers).get("Content-Type")).toBe("application/json");
  });
});
