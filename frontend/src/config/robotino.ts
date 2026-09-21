/** Dashboard sensor-display constant and duration formatting. */

export const SENSOR_MAX_RANGE_M = 0.41;

/** Return finite `seconds` as a compact duration, or an em dash when unavailable. */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return "—";
  const rounded = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(rounded / 60);
  const remainder = rounded % 60;
  return minutes > 0 ? `${minutes}m ${remainder}s` : `${remainder}s`;
}
