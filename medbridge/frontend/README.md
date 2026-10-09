# MedBridge Streamlit Frontend (legacy / fallback)

The primary user interface is the FastAPI-served single-page app in
`medbridge/web/` (served at `http://127.0.0.1:8000/`). This Streamlit app is
kept as a lightweight fallback and for demos where Streamlit is preferred.

```bash
streamlit run medbridge/frontend/app.py   # http://localhost:8501
```

It talks to the backend only over HTTP (`MEDBRIDGE_API_URL`, default
`http://localhost:8000`):

- `POST /api/sessions` creates a session; the returned `session_token` is kept
  in `st.session_state` (never in the URL) and sent as
  `Authorization: Bearer <token>` on every session call.
- If the backend rejects the token (expired, erased, or lost after a full page
  reload), the app starts a new session and resends the message once.
- Each answer shows its routing action badge (ANSWER, SOFT-ASK, GENERALIZE,
  ABSTAIN, ESCALATE) and a collapsible citations panel.

No web fonts or CDN resources are loaded.
