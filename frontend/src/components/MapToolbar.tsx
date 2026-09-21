/** Control factory-map framing and optional visualization layers. */

import type { HoverDetail, LayerVisibility } from "../types/fleet";

const LAYERS: Array<{ key: keyof LayerVisibility; label: string }> = [
  { key: "occupancyMap", label: "Map image" },
  { key: "movementArea", label: "Valid movement area" },
  { key: "staticObstacles", label: "Machines and walls" },
  { key: "labels", label: "Machine labels" },
  { key: "footprints", label: "Robotino footprints" },
  { key: "trajectories", label: "Goals and trajectories" },
  { key: "sensors", label: "Distance sensors" },
  { key: "lidar", label: "LiDAR scans" },
  { key: "temporaryObstacles", label: "Temporary obstacles" },
];

/**
 * Return map controls for visible `layers` and the current `hoverDetail`.
 * `onToggleLayer` changes one display layer; `onFit` requests camera refit.
 */
export function MapToolbar({
  layers, hoverDetail, onToggleLayer, onFit,
}: {
  layers: LayerVisibility;
  hoverDetail: HoverDetail | null;
  onToggleLayer: (layer: keyof LayerVisibility) => void;
  onFit: () => void;
}) {
  return <>
    <div className="map-toolbar">
      <button className="button button-map" onClick={onFit}>Fit factory</button>
      <details className="layer-menu"><summary className="button button-map">Layers</summary>
        <div className="layer-menu-popover"><p>Map layers</p>
          {LAYERS.map((layer) => <label key={layer.key}>
            <input type="checkbox" checked={layers[layer.key]} onChange={() => onToggleLayer(layer.key)} />
            <span>{layer.label}</span>
          </label>)}
        </div>
      </details>
      <span className="map-help">Drag to pan · Scroll to zoom</span>
    </div>
    <div className="map-legend">
      <span><i className="legend-line line-route" />Trajectory</span>
      <span><i className="legend-line line-reserved" />Goal</span>
      <span><i className="legend-line line-waiting" />Safety stop</span>
    </div>
    <div className={`map-hover-detail${hoverDetail ? " is-visible" : ""}`}>
      {hoverDetail ? <><strong>{hoverDetail.title}</strong><span>{hoverDetail.detail}</span></> : <span>Select a Robotino or label</span>}
    </div>
  </>;
}
