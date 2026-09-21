/** Render the factory map, geometry, goals, and Robotino sensor layers. */

import { Html, Line, MapControls, useTexture } from "@react-three/drei";
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo } from "react";
import * as THREE from "three";
import type {
  Bounds, HoverDetail, LayerVisibility, MapFixture, NavigationGoal,
  Robotino, TemporaryObstacle,
} from "../types/fleet";

interface Props {
  fixture: MapFixture;
  robots: Robotino[];
  goals: Record<string, NavigationGoal>;
  temporaryObstacles: TemporaryObstacle[];
  selectedRobotId: string;
  selectedLocationId: string | null;
  layers: LayerVisibility;
  fitSignal: number;
  onSelectRobot: (id: string) => void;
  onSelectLocation: (id: string) => void;
  onHoverDetail: (detail: HoverDetail | null) => void;
}

/** Return map-frame bounds enclosing the movement area, obstacles, and labels in `fixture`. */
/** Return map-frame bounds enclosing the movement area, obstacles, and labels in `fixture`. */
function sceneBounds(fixture: MapFixture): Bounds {
  const points = [
    ...fixture.environmentGeometry.movementArea,
    ...fixture.environmentGeometry.obstacles.flatMap((obstacle) => obstacle.points),
    ...fixture.locations,
  ];
  if (points.length === 0) return fixture.map.bounds;
  return points.reduce<Bounds>((bounds, point) => ({
    minX: Math.min(bounds.minX, point.x), minY: Math.min(bounds.minY, point.y),
    maxX: Math.max(bounds.maxX, point.x), maxY: Math.max(bounds.maxY, point.y),
  }), {
    minX: Number.POSITIVE_INFINITY,
    minY: Number.POSITIVE_INFINITY,
    maxX: Number.NEGATIVE_INFINITY,
    maxY: Number.NEGATIVE_INFINITY,
  });
}

/** Fit the orthographic camera to `fixture`; changes to `signal` request a fresh fit. Returns no visible JSX. */
/** Fit the orthographic camera to `fixture`; changes to `signal` request a fresh fit. Returns no visible JSX. */
function FitCamera({ fixture, signal }: { fixture: MapFixture; signal: number }) {
  const { camera, size } = useThree();
  const bounds = useMemo(() => sceneBounds(fixture), [fixture]);
  useEffect(() => {
    if (!(camera instanceof THREE.OrthographicCamera)) return;
    if (size.width <= 0 || size.height <= 0) return;
    const width = Math.max(1, bounds.maxX - bounds.minX + 2);
    const height = Math.max(1, bounds.maxY - bounds.minY + 2);
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    camera.position.set(centerX, centerY, 50);
    camera.lookAt(centerX, centerY, 0);
    camera.zoom = Math.max(0.1, Math.min(size.width / width, size.height / height));
    camera.updateProjectionMatrix();
  }, [bounds, camera, signal, size.height, size.width]);
  return null;
}

/** Render `fixture`'s occupancy image at its calibrated map-frame bounds. */
/** Render `fixture`'s occupancy image at its calibrated map-frame bounds. */
function MapImage({ fixture }: { fixture: MapFixture }) {
  const texture = useTexture(fixture.map.imageUrl);
  const bounds = fixture.map.bounds;
  const width = bounds.maxX - bounds.minX;
  const height = bounds.maxY - bounds.minY;
  useEffect(() => {
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
  }, [texture]);
  return <mesh position={[bounds.minX + width / 2, bounds.minY + height / 2, -0.4]}>
    <planeGeometry args={[width, height]} />
    <meshBasicMaterial map={texture} color="#75818d" opacity={0.82} transparent />
  </mesh>;
}

/** Draw a map polygon from `points` with `color`, `opacity`, and layer depth `z`. */
/** Draw a map polygon from `points` with `color`, `opacity`, and layer depth `z`. */
function Polygon({
  points, color, opacity, z,
}: {
  points: Array<{ x: number; y: number }>; color: string; opacity: number; z: number;
}) {
  const shape = useMemo(
    () => new THREE.Shape(points.map((point) => new THREE.Vector2(point.x, point.y))),
    [points],
  );
  if (points.length < 3) return null;
  return <>
    <mesh position={[0, 0, z]}><shapeGeometry args={[shape]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
    </mesh>
    <Line points={[...points, points[0]].map((point) => [point.x, point.y, z + 0.01])} color={color} lineWidth={1.2} />
  </>;
}

/** Render `robot`'s projected LiDAR points in that Robotino's display color. */
/** Render `robot`'s projected LiDAR points in that Robotino's display color. */
function Lidar({ robot }: { robot: Robotino }) {
  const positions = useMemo(() => {
    const values = new Float32Array((robot.lidar?.points.length ?? 0) * 3);
    robot.lidar?.points.forEach((point, index) => {
      values[index * 3] = point.x;
      values[index * 3 + 1] = point.y;
      values[index * 3 + 2] = 0.12;
    });
    return values;
  }, [robot.lidar]);
  return <points>
    <bufferGeometry><bufferAttribute attach="attributes-position" args={[positions, 3]} /></bufferGeometry>
    <pointsMaterial color={robot.color} size={0.055} sizeAttenuation />
  </points>;
}

/** Render one `robot` footprint, selected/goal markers, trajectory, and enabled sensor layers. */
/**
 * Render `robot` at physical `radius`, with `selected` and `goal` markers.
 * Layer flags control footprint, sensors, LiDAR, and trajectory; `onSelect`
 * and `onHover` report interaction to the dashboard.
 */
function Robot({
  robot, radius, selected, showFootprint, showSensors, showLidar, showTrajectory, goal,
  onSelect, onHover,
}: {
  robot: Robotino; radius: number; selected: boolean; showFootprint: boolean; showSensors: boolean;
  showLidar: boolean; showTrajectory: boolean; goal?: NavigationGoal;
  onSelect: () => void; onHover: (detail: HoverDetail | null) => void;
}) {
  return <>
    {showTrajectory && robot.trajectory.length > 1 && (
      <Line points={robot.trajectory.map((point) => [point.x, point.y, 0.12])} color={robot.color} lineWidth={2} />
    )}
    {showTrajectory && goal && <>
      <Line points={[[robot.x, robot.y, 0.10], [goal.x, goal.y, 0.10]]} color={robot.color} lineWidth={1} dashed transparent opacity={0.45} />
      <mesh position={[goal.x, goal.y, 0.13]}><ringGeometry args={[0.13, 0.18, 24]} /><meshBasicMaterial color={robot.color} /></mesh>
    </>}
    <group position={[robot.x, robot.y, 0.2]} rotation={[0, 0, robot.headingRad]}>
      {showFootprint && <mesh
        onClick={(event: ThreeEvent<MouseEvent>) => { event.stopPropagation(); onSelect(); }}
        onPointerOver={(event) => { event.stopPropagation(); onHover({ title: robot.ip, detail: robot.status }); }}
        onPointerOut={() => onHover(null)}
      >
        <circleGeometry args={[radius, 40]} />
        <meshBasicMaterial color={robot.color} transparent opacity={robot.poseValid ? 0.9 : 0.35} />
      </mesh>}
      {showFootprint && <Line points={[[0, 0, 0.03], [radius, 0, 0.03]]} color="#101820" lineWidth={2} />}
      {showFootprint && selected && <mesh><ringGeometry args={[radius + 0.05, radius + 0.08, 40]} /><meshBasicMaterial color="#ffffff" /></mesh>}
      {showSensors && robot.sensors.map((sensor) => {
        const angle = -sensor.index * 40 * Math.PI / 180;
        const end = radius + Math.min(sensor.rangeM, 0.410);
        return <Line key={sensor.index} points={[
          [Math.cos(angle) * radius, Math.sin(angle) * radius, 0.06],
          [Math.cos(angle) * end, Math.sin(angle) * end, 0.06],
        ]} color={sensor.rangeM < 0.405 ? "#ff5964" : robot.color} lineWidth={1} transparent opacity={0.55} />;
      })}
      {showLidar && robot.lidar && <Lidar robot={robot} />}
    </group>
  </>;
}

/** Compose `props`' factory geometry and live world state into the interactive Three.js scene. */
/** Compose `props`' factory geometry and live world state into the interactive Three.js scene. */
function Scene(props: Props) {
  const { fixture, layers } = props;
  const bounds = useMemo(() => sceneBounds(fixture), [fixture]);
  const center = useMemo(
    () => [
      (bounds.minX + bounds.maxX) / 2,
      (bounds.minY + bounds.maxY) / 2,
      0,
    ] as const,
    [bounds],
  );
  return <>
    <color attach="background" args={["#111820"]} />
    <FitCamera fixture={fixture} signal={props.fitSignal} />
    {layers.occupancyMap && <MapImage fixture={fixture} />}
    {layers.movementArea && <Polygon points={fixture.environmentGeometry.movementArea} color="#45dfa2" opacity={0.10} z={-0.15} />}
    {layers.staticObstacles && fixture.environmentGeometry.obstacles.map((obstacle) => (
      <Polygon key={obstacle.id} points={obstacle.points} color={obstacle.category === "WALL" ? "#ff6571" : "#bf9f62"} opacity={0.32} z={-0.05} />
    ))}
    {layers.labels && fixture.locations.map((location) => (
      <group key={location.id} position={[location.x, location.y, 0.25]}>
        <mesh onClick={(event) => { event.stopPropagation(); props.onSelectLocation(location.id); }}>
          <circleGeometry args={[props.selectedLocationId === location.id ? 0.19 : 0.15, 24]} />
          <meshBasicMaterial color={props.selectedLocationId === location.id ? "#31d7c5" : "#17212b"} />
        </mesh>
        <Html center style={{ pointerEvents: "none" }}><span className="map-node-label">{location.id}</span></Html>
      </group>
    ))}
    {layers.temporaryObstacles && props.temporaryObstacles.map((obstacle) => (
      <mesh key={obstacle.id} position={[obstacle.x, obstacle.y, 0.18]}>
        <circleGeometry args={[obstacle.radiusM, 32]} /><meshBasicMaterial color="#ff6673" transparent opacity={0.75} />
      </mesh>
    ))}
    {props.robots.filter((robot) => robot.poseValid).map((robot) => <Robot
      key={robot.id}
      robot={robot}
      radius={fixture.robotDiameterM / 2}
      selected={robot.id === props.selectedRobotId}
      showFootprint={layers.footprints}
      showSensors={layers.sensors}
      showLidar={layers.lidar}
      showTrajectory={layers.trajectories}
      goal={props.goals[robot.id]}
      onSelect={() => props.onSelectRobot(robot.id)}
      onHover={props.onHoverDetail}
    />)}
    <MapControls
      key={props.fitSignal}
      makeDefault
      target={center}
      enableRotate={false}
      screenSpacePanning
      minZoom={1}
      maxZoom={180}
    />
  </>;
}

/** Return the orthographic canvas for the map and live fleet `props`. */
/** Return the orthographic canvas for the map and live fleet `props`. */
export function FactoryMap(props: Props) {
  return <Canvas orthographic camera={{ position: [0, 0, 50], near: 0.1, far: 200 }}>
    <Suspense fallback={null}><Scene {...props} /></Suspense>
  </Canvas>;
}
