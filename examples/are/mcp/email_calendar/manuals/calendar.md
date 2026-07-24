# Tool Metadata
id: CalendarApp
category: Productivity / Scheduling

# Functional Description
A simulated calendar of tagged events with a title, start/end time, description, location, and
attendee list. Supports adding, reading, searching, and deleting events, and querying by date range,
tag, or "today".

# Observable Properties
The app's state resource is a snapshot of the calendar (all events); it is exposed as one observable
property, `state`.

# Signals
`state_changed` fires whenever the calendar changes — an event is added or deleted.

# Operations
- add_calendar_event(title, start_datetime, end_datetime, tag, description, location, attendees):
  creates an event; `start_datetime`/`end_datetime` are `YYYY-MM-DD HH:MM:SS`. Unless the task says
  otherwise, the week starts Monday and ends Sunday — resolve a relative day ("next Monday") against
  that convention before calling.
- get_calendar_events_from_to(start_datetime, end_datetime, offset, limit): events overlapping a date
  range (excludes events that only touch at the range's boundaries) — the natural call for "am I
  free Monday afternoon?" before proposing a time.
- read_today_calendar_events(): today's events, no arguments.
- search_events(query): partial match against title, description, location, or attendees.
- get_calendar_event(event_id) / get_calendar_events_by_tag(tag): look up a specific event, or every
  event sharing a tag.
- delete_calendar_event(event_id): permanently removes an event.

# Usage Protocols & Safety
Check `get_calendar_events_from_to` (or `read_today_calendar_events`) for conflicts in the target
window before calling `add_calendar_event` — the app does not reject overlapping events itself.

`event_id` for `delete_calendar_event`/`get_calendar_event` comes from a prior
`get_calendar_events_from_to`/`search_events`/`get_calendar_events_by_tag` call, never invented.

`delete_calendar_event` is destructive and not reversible within the session — reserve it for an
explicit cancellation request, not routine calendar management.
