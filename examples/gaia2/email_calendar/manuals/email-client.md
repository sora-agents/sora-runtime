# Tool Metadata
id: EmailClientApp
category: Communication / Email

# Functional Description
A simulated mailbox with four folders (INBOX, SENT, DRAFT, TRASH). Supports listing, searching, and
reading individual emails, and sending, replying to, forwarding, moving, and deleting them.

# Observable Properties
```yaml
- name: state
```
The app's state resource is a JSON snapshot of the whole mailbox, exposed as one observable
property, `state`. No JSON Schema for this shape is available anywhere (MCP resources carry no
schema field, unlike MCP tools), so it's spelled out here instead:
- `user_email` (string): the mailbox owner's address.
- `view_limit` (integer): the default page size `list_emails` uses when no `limit` is given.
- `folders` (object): keyed by folder name (`INBOX`, `SENT`, `DRAFT`, `TRASH`). Each folder is an
  object with:
  - `folder_name` (string): the same key, repeated.
  - `emails` (array): each entry is an object with `sender` (string), `recipients` (array of
    strings), `subject` (string), `content` (string), `email_id` (string — the id every operation
    above takes), `parent_id` (string or null — the id of the email this one is a reply to, if any),
    `cc` (array of strings), `attachments` (object mapping filename to content), `timestamp` (number,
    Unix epoch seconds), and `is_read` (boolean).

# Signals
```yaml
- name: state_changed
```
`state_changed` fires whenever the mailbox changes — a new email arrives, one is sent, replied to,
forwarded, moved, or deleted. Same schema gap as `state` above (no native schema anywhere), so again
spelled out here. The payload is an object with:
- `uri` (string): the resource that changed, always `app://EmailClientApp/state`.
- `value` (object): the refreshed `state` snapshot at the moment of the change — same shape as the
  `state` observable property described above (`user_email`, `view_limit`, `folders`).

# Operations
```yaml
- name: list_emails
- name: search_emails
  required: [query]
- name: get_email_by_id
  required: [email_id]
- name: get_email_by_index
  required: [idx]
- name: send_email
- name: reply_to_email
  required: [email_id]
- name: forward_email
  required: [email_id]
- name: move_email
  required: [email_id]
- name: delete_email
  required: [email_id]
- name: download_attachments
  required: [email_id]
```
- list_emails(folder_name, offset, limit): lists emails in a folder (INBOX/SENT/DRAFT/TRASH,
  defaults to INBOX), paginated by offset/limit. The natural first call for "any unread mail" /
  "what's in my inbox" style tasks.
- search_emails(query, folder_name): finds emails by a partial match against sender, recipients,
  subject, or content, within one folder.
- get_email_by_id(email_id, folder_name) / get_email_by_index(idx, folder_name): reads one email in
  full (marks it read). Prefer `get_email_by_id` once an id is known from `list_emails` or
  `search_emails`.
- send_email(recipients, subject, content, cc, attachment_paths): composes and sends a brand-new
  email; returns the new email's id.
- reply_to_email(email_id, folder_name, content, attachment_paths): replies in-thread to an existing
  email; returns the reply's id.
- forward_email(email_id, recipients, folder_name): forwards an existing email; returns the new
  email's id.
- move_email(email_id, source_folder_name, dest_folder_name): moves an email between folders.
- delete_email(email_id, folder_name): permanently removes an email from a folder.
- download_attachments(email_id, folder_name, path_to_save): saves an email's attachments to disk.

# Usage Protocols & Safety
`email_id` is never invented — it always comes from a prior `list_emails`, `search_emails`, or
`get_email_by_id`/`get_email_by_index` call against the same folder. `reply_to_email` and
`forward_email` fail if the id doesn't resolve in the given `folder_name` (defaults to INBOX), so
pass the folder the email was actually listed from if it's not INBOX.

`delete_email` is destructive and not reversible within the session — prefer `move_email` to TRASH
when the task only implies tidying up, and reserve `delete_email` for an explicit delete request.

`send_email`/`reply_to_email`/`forward_email` all commit immediately — there is no draft-review step
in this simulation, so compose the final content before invoking.
