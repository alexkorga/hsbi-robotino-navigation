"""Check dashboard API responses and goal actions against the fleet runtime."""

import unittest

from fastapi.testclient import TestClient

from robotino_fleet.api.app import create_app
from robotino_fleet.bootstrap import build_fleet_runtime


class FleetApiTests(unittest.TestCase):
    def setUp(self) -> None:
        """Create a simulated runtime and API client for endpoint assertions."""

        self.fleet = build_fleet_runtime("simulation")
        self.client = TestClient(create_app(self.fleet, start_background=False))

    def test_contract_contains_geometry_without_graph_resources(self) -> None:
        """Expose geometry but not removed graph-based traffic resources."""

        data = self.client.get("/api/map").json()
        self.assertIn("environmentGeometry", data)
        self.assertNotIn("graph", data)
        self.assertNotIn("trafficResources", data)

    def test_world_contract_reports_command_queue_status(self) -> None:
        """Keep the world schema aligned with dashboard command-queue fields."""

        data = self.client.get("/api/world").json()
        self.assertEqual(data["schemaVersion"], 3)
        self.assertIn("commandQueued", data["robots"][0])
        self.assertNotIn("commandTransmitted", data["robots"][0])

    def test_one_goal_endpoint_and_stop(self) -> None:
        """Assign and cancel a label goal through the same API used by the UI."""

        robot_id = next(iter(self.fleet.robots))
        response = self.client.post(
            f"/api/robotinos/{robot_id}/goal", json={"locationId": "13"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["goals"][robot_id]["locationId"], "13")
        stopped = self.client.post(f"/api/robotinos/{robot_id}/stop")
        self.assertEqual(stopped.status_code, 200)
        self.assertNotIn(robot_id, stopped.json()["goals"])


if __name__ == "__main__":
    unittest.main()
