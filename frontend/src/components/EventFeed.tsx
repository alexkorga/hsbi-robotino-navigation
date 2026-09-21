/** Present recent fleet events from the backend. */

import type { FleetEvent } from "../types/fleet";

/** Return the six most recent `events` as the activity list. */
export function EventFeed({ events }: { events: FleetEvent[] }) {
  return (
    <section className="event-feed" aria-labelledby="event-feed-title">
      <div className="event-feed-heading">
        <div>
          <p className="eyebrow">Activity</p>
          <h2 id="event-feed-title">Fleet events</h2>
        </div>
        <span>Latest first</span>
      </div>
      <ol>
        {events.slice(0, 6).map((event) => (
          <li key={event.id}>
            <time>{event.time}</time>
            <span className={`event-marker event-${event.type.toLowerCase()}`} />
            <p>
              {event.robotId && <strong>{event.robotId} </strong>}
              {event.message}
            </p>
          </li>
        ))}
      </ol>
    </section>
  );
}
