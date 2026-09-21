/** Backend map, telemetry, and dashboard state contracts. */

export interface Point { x: number; y: number }
export interface Bounds { minX: number; minY: number; maxX: number; maxY: number }
export interface FactoryMap {
  id: string; name: string; imageUrl: string; widthPx: number; heightPx: number;
  resolutionMPerPx: number; origin: Point & { theta: number }; bounds: Bounds;
}
export interface MapLocation extends Point { id: string; headingRad: number }
export interface Station extends Point {
  id: string; type: string; approachLocation: string | null; headingRad: number;
}
export interface EnvironmentObstacle {
  id: string; name: string; category: "MACHINE" | "WALL"; points: Point[];
}
export interface EnvironmentGeometry {
  source: string; movementArea: Point[]; obstacles: EnvironmentObstacle[];
}
export interface MapFixture {
  schemaVersion: number; map: FactoryMap; locations: MapLocation[]; stations: Station[];
  environmentGeometry: EnvironmentGeometry;
  robotDiameterM: number; distanceSensorMaxM: number;
}
export type RobotStatus =
  | "IDLE" | "NAVIGATING" | "WAITING" | "ARRIVED"
  | "READY_TO_DOCK" | "STOPPED" | "OFFLINE" | "FAULT";
export interface VelocityCommand { vx: number; vy: number; omega: number }
export interface OdometryState extends VelocityCommand {
  // Odometry has its own local origin; Robotino.x/y are in the factory-map frame.
  x: number; y: number; headingRad: number; sequence: number; timestampS: number;
}
export interface DistanceSensor { index: number; rangeM: number }
export interface LidarPoint extends Point { rangeM: number; intensity: number | null }
export interface LidarScanState {
  sequence: number; timestampS: number; ageS: number; rangeMinM: number; rangeMaxM: number;
  sampleCount: number; validCount: number; points: LidarPoint[];
}
export interface Robotino extends Point {
  id: string; ip: string; name: string; color: string; headingRad: number;
  status: RobotStatus; online: boolean; poseValid: boolean; poseSource: string;
  poseAgeS: number; goalLocationId: string | null; speedMps: number;
  sensors: DistanceSensor[]; distanceSensorAgeS: number | null;
  lidar: LidarScanState | null; bumperPressed: boolean; batteryLow: boolean;
  odometry: OdometryState | null; measuredVelocity: VelocityCommand;
  proposedCommand: VelocityCommand; appliedCommand: VelocityCommand;
  commandQueued: boolean; stopReason: string | null; policyId: string | null;
  policyInferenceMs: number | null; trajectory: Point[];
  straightPathLengthM: number | null; drivenPathLengthM: number;
  pathEfficiency: number | null; travelTimeS: number | null;
}
export interface NavigationGoal extends Point {
  locationId: string; headingRad: number | null; positionToleranceM: number;
  headingToleranceRad: number; dockingHandoff: string | null;
}
export interface TemporaryObstacle extends Point { id: string; radiusM: number }
export interface FleetEvent {
  id: number; time: string; robotId: string | null; type: string; message: string;
}
export interface WorldState {
  schemaVersion: number; sequence: number; clockS: number;
  mode: "simulation" | "shadow" | "live"; paused: boolean;
  physicalCommandsEnabled: boolean; robots: Robotino[];
  goals: Record<string, NavigationGoal>; temporaryObstacles: TemporaryObstacle[];
  errors: Record<string, Record<string, string>>; events: FleetEvent[];
}
export interface LayerVisibility {
  occupancyMap: boolean; movementArea: boolean; staticObstacles: boolean;
  labels: boolean; footprints: boolean; trajectories: boolean; sensors: boolean;
  lidar: boolean; temporaryObstacles: boolean;
}
export interface HoverDetail { title: string; detail: string }
