"""
Streamlit real‑time dashboard (WebSocket-ready, Mock feed for now)

Features
- Live view of incoming JSON (mock generator today; WebSocket client scaffold included)
- Start/Stop experiment recording (recording window only affects saved dataset)
- Export recorded data to CSV via a download button
- Live charts for selected numeric fields (auto-detected from payload)
- Configurable refresh rate and buffer window
- Generic, stores the **full** JSON each tick (flattened keys) so you can chart any field later

Run locally
  pip install streamlit pandas streamlit-autorefresh websocket-client
  streamlit run dashboard.py

Notes
- Default mode is **Mock** (generates Shelly Pro 3EM-like payloads similar to your example).
- WebSocket mode is scaffolded with a background thread using `websocket-client`. Enable later by switching mode and providing a ws:// or wss:// URL.
"""
from __future__ import annotations

import json
import math
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional
from collections import deque

import pandas as pd
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    def st_autorefresh(*_args, **_kwargs):
        return None
import plotly.express as px
import plotly.graph_objects as go
import dashboard_config as cfg
import streamlit.components.v1 as components

# Optional dependency for real WebSocket mode. Safe to import even if unused.
try:
    from websocket import WebSocketApp  # type: ignore
except Exception:  # pragma: no cover
    WebSocketApp = None  # type: ignore

# ----------------------------- Utilities ----------------------------- #

def now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .astimezone()  # local tz
        .isoformat(timespec="milliseconds")
    )


def flatten(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: List[Tuple[str, Any]] = []
    for k, v in (d or {}).items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ----------------------------- Mock data generator ----------------------------- #

def mock_payload(last: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Generate a Shelly Pro 3EM-like payload close to the user's example."""
    # Baselines
    base_voltage_c = 220.0
    base_freq = 50.0

    # Small random walks to look alive
    def jitt(v: float, spread: float) -> float:
        return v + random.uniform(-spread, spread)

    # Derive prior values for smoother series
    if last and isinstance(last.get("em_status"), dict):
        prev = last["em_status"]
        c_voltage = jitt(float(prev.get("c_voltage", base_voltage_c)), 0.6)
        c_current = max(0.0, jitt(float(prev.get("c_current", 0.03)), 0.01))
        a_current = max(0.0, jitt(float(prev.get("a_current", 0.03)), 0.015))
        b_current = max(0.0, jitt(float(prev.get("b_current", 0.03)), 0.015))
        freq = jitt(float(prev.get("c_freq", base_freq)), 0.05)
    else:
        c_voltage = jitt(base_voltage_c, 1.0)
        c_current = 0.03
        a_current = 0.03
        b_current = 0.03
        freq = base_freq

    # Apparent powers VA = V * A (simplified, per-phase)
    c_aprt = c_voltage * c_current
    a_aprt = c_voltage * a_current * 0.0  # assume phase A not energized in example
    b_aprt = c_voltage * b_current * 0.0  # assume phase B not energized

    total_current = a_current + b_current + c_current
    total_aprt = a_aprt + b_aprt + c_aprt

    payload = {
        "ip": "192.168.0.129",
        "device_info": {
            "name": None,
            "id": "shellypro3em-841fe891e27c",
            "mac": "841FE891E27C",
            "slot": 0,
            "model": "SPEM-003CEBEU",
            "gen": 2,
            "fw_id": "20250924-062749/1.7.1-gd336f31",
            "ver": "1.7.1",
            "app": "Pro3EM",
            "auth_en": False,
            "auth_domain": None,
            "profile": "triphase",
        },
        "em_status": {
            "id": 0,
            "a_current": round(a_current, 3),
            "a_voltage": 0.0,
            "a_act_power": 0.0,
            "a_aprt_power": round(a_aprt, 1),
            "a_pf": 0.0,
            "a_freq": 0.0,
            "b_current": round(b_current, 3),
            "b_voltage": 0.0,
            "b_act_power": 0.0,
            "b_aprt_power": round(b_aprt, 1),
            "b_pf": 0.0,
            "b_freq": 0.0,
            "c_current": round(c_current, 3),
            "c_voltage": round(c_voltage, 1),
            "c_act_power": 0.0,
            "c_aprt_power": round(c_aprt, 1),
            "c_pf": 0.0,
            "c_freq": round(freq, 2),
            "n_current": None,
            "total_current": round(total_current, 3),
            "total_act_power": 0.0,
            "total_aprt_power": round(total_aprt, 3),
            "user_calibrated_phase": [],
        },
    }
    return payload


# ----------------------------- WebSocket client (scaffold) ----------------------------- #

class WSClient:
    def __init__(self, url: str, queue: deque):
        self.url = url
        self.queue = queue
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws: WebSocketApp | None = None

    def start(self):
        if WebSocketApp is None:
            raise RuntimeError("websocket-client is not installed")
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    def _on_message(self, _, message: str):
        try:
            data = json.loads(message)
            self.queue.append((time.time(), data))
            # Trim queue to avoid unbounded growth
            if len(self.queue) > 10000:
                for _ in range(len(self.queue) - 10000):
                    self.queue.popleft()
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                self._ws = WebSocketApp(self.url, on_message=self._on_message)
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                time.sleep(1.0)
            finally:
                self._ws = None


# ----------------------------- Streamlit App ----------------------------- #

st.set_page_config(page_title="H2 / Power Dashboard", layout="wide")

# Preserve scroll position across reruns for smoother UX
components.html(
    """
<script>
(function() {
  const key = 'st-scroll-pos:' + (window.location.pathname || 'root');
  function save() {
    try { sessionStorage.setItem(key, String(window.scrollY || window.pageYOffset || 0)); } catch (e) {}
  }
  function getY() {
    try { return parseInt(sessionStorage.getItem(key) || '0', 10) || 0; } catch (e) { return 0; }
  }
  function restore() {
    const y = getY();
    if (y >= 0) {
      window.requestAnimationFrame(() => window.scrollTo(0, y));
    }
  }
  try { history.scrollRestoration = 'manual'; } catch (e) {}
  // Attempt restoration immediately and after layout settles
  restore();
  ;[50, 150, 300, 600, 1200, 2000].forEach(t => setTimeout(restore, t));
  const start = Date.now();
  const obs = new MutationObserver(() => {
    if (Date.now() - start < 3000) {
      restore();
    } else {
      obs.disconnect();
    }
  });
  obs.observe(document.documentElement || document.body, { childList: true, subtree: true });
  document.addEventListener('scroll', save, { passive: true });
  window.addEventListener('beforeunload', save);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) setTimeout(restore, 100); });
})();
</script>
    """,
    height=0,
)

# Session state bootstrap with optimized defaults
ss = st.session_state
ss.setdefault("live_rows", [])           # list of flattened dicts with timestamp
ss.setdefault("recorded_rows", [])       # subset captured during recording
ss.setdefault("recording", False)
ss.setdefault("buffer_seconds", 10 * 60)  # 10 minutes live window
ss.setdefault("max_rows", 20000)
ss.setdefault("ws_queue", deque())
ss.setdefault("ws_client", None)
ss.setdefault("mode", "Mock")            # or 'WebSocket'
ss.setdefault("refresh_ms", 500)         # Increased default refresh rate for stability
ss.setdefault("last_mock_payload", None)
ss.setdefault("chart_data_cache", None)  # Cache for chart data to prevent flashing
ss.setdefault("last_chart_update", 0)    # Track last chart update time
ss.setdefault("selected_fields", [])     # Cache selected fields
ss.setdefault("last_ingest_time", 0.0)

# Config-driven runtime (no sidebar)
ss.mode = cfg.DATA_SOURCE
ss.refresh_ms = cfg.REFRESH_MS
ss.buffer_seconds = cfg.BUFFER_SECONDS
ss.max_rows = cfg.MAX_ROWS
ws_url = cfg.WS_URL if ss.mode == "WebSocket" else None

# Auto-rerun while page is open (uses config)
if not getattr(cfg, "SMOOTH_UPDATES", False):
    st_autorefresh(interval=ss.refresh_ms, key="_autorefresh")

# Sidebar controls
with st.sidebar:
    st.subheader("Controls")
    btn_cols = st.columns([1,1])
    with btn_cols[0]:
        if st.button("Start Recording", use_container_width=True, key="start_rec"):
            ss.recording = True
            ss.chart_data_cache = None
    with btn_cols[1]:
        if st.button("Stop Recording", use_container_width=True, key="stop_rec"):
            ss.recording = False
    if st.button("Clear Live", use_container_width=True, key="clear_live"):
        ss.live_rows = []
        ss.chart_data_cache = None
    if st.button("Clear Recorded", use_container_width=True, key="clear_rec"):
        ss.recorded_rows = []
    if ss.recorded_rows:
        if not hasattr(ss, 'last_csv_data') or ss.last_csv_data != len(ss.recorded_rows):
            df_dl = pd.DataFrame(ss.recorded_rows)
            ss.csv_bytes = df_dl.to_csv(index=False).encode("utf-8")
            ss.csv_filename = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            ss.last_csv_data = len(ss.recorded_rows)
        st.download_button(
            "Export CSV",
            data=ss.csv_bytes,
            file_name=ss.csv_filename,
            mime="text/csv",
            use_container_width=True,
            key="export_csv"
        )
    else:
        st.button("Export CSV", use_container_width=True, disabled=True, key="export_csv_disabled")
# (Status line removed by request)

# ----------------------------- Ingest step per rerun ----------------------------- #

def migrate_existing_data():
    """Migrate existing data to include epoch_time field."""
    if ss.live_rows and len(ss.live_rows) > 0 and "epoch_time" not in ss.live_rows[0]:
        current_time = time.time()
        for i, row in enumerate(ss.live_rows):
            if "epoch_time" not in row:
                # Estimate epoch time based on position (newer data gets more recent time)
                estimated_time = current_time - (len(ss.live_rows) - i) * 0.5  # Assume 0.5s intervals
                row["epoch_time"] = estimated_time

def append_row_from_payload(payload: Dict[str, Any]):
    """Optimized data ingestion with efficient trimming."""
    current_time = time.time()
    row = {"timestamp": now_iso(), "epoch_time": current_time}
    row.update(flatten(payload))
    
    # Add to live data
    ss.live_rows.append(row)
    
    # Optimized trimming: only trim when we exceed limits
    if len(ss.live_rows) > ss.max_rows or (ss.live_rows and 
        current_time - ss.live_rows[0].get("epoch_time", current_time) > ss.buffer_seconds):
        
        cutoff_time = current_time - ss.buffer_seconds
        # Keep only recent data that fits within limits
        ss.live_rows = [
            r for r in ss.live_rows[-ss.max_rows:] 
            if r.get("epoch_time", current_time) >= cutoff_time
        ]
    
    # Add to recorded data if recording
    if ss.recording:
        ss.recorded_rows.append(row)


# Migrate existing data if needed
migrate_existing_data()

if ss.mode == "Mock":
    # One payload per rerun
    p = mock_payload(ss.last_mock_payload)
    append_row_from_payload(p)
    ss.last_mock_payload = p
else:
    # Drain queued ws messages collected by background client
    # (scaffold; to enable, create WSClient below and start/stop using buttons or auto-connect)
    if ws_url and WebSocketApp is not None:
        if ss.ws_client is None:
            ss.ws_client = WSClient(ws_url, ss.ws_queue)
            # Auto-start on first use
            try:
                ss.ws_client.start()
            except Exception as e:  # pragma: no cover
                st.warning(f"WebSocket not started: {e}")
        # Pull up to N messages per rerun
        pulls = 0
        while ss.ws_queue and pulls < 100:
            _ts, msg = ss.ws_queue.popleft()
            append_row_from_payload(msg)
            pulls += 1
    else:
        st.info("Install websocket-client and provide a ws:// URL to enable WebSocket ingestion.")

# ----------------------------- Display ----------------------------- #

if not ss.live_rows:
    st.info("Waiting for data…")
else:
    # Use cached data if available and recent
    current_time = time.time()
    use_cache = (ss.chart_data_cache is not None and 
                current_time - ss.last_chart_update < 2.0)  # Cache for 2 seconds
    
    if not use_cache:
        # Process data only when needed
        df = pd.DataFrame(ss.live_rows)
        
        # Auto-detect numeric columns (prefer em_status.*)
        numeric_cols = [
            c for c in df.columns
            if c not in ["timestamp", "epoch_time"] and pd.api.types.is_numeric_dtype(df[c])
        ]
        preferred = [c for c in numeric_cols if c.startswith("em_status.")]
        default_sel = [c for c in preferred if any(x in c for x in ["c_voltage", "c_current", "total_aprt_power", "total_current"])]
        
        # Cache the processed data
        ss.chart_data_cache = {
            'df': df,
            'numeric_cols': numeric_cols,
            'preferred': preferred,
            'default_sel': default_sel
        }
        ss.last_chart_update = current_time
    else:
        # Use cached data
        cache = ss.chart_data_cache
        df = cache['df']
        numeric_cols = cache['numeric_cols']
        preferred = cache['preferred']
        default_sel = cache['default_sel']
    
    # Field selection with persistence
    if not ss.selected_fields:
        ss.selected_fields = default_sel[:4] if default_sel else (preferred[:3] if preferred else numeric_cols[:3])
    
    sel = st.multiselect(
        "Fields to plot",
        options=preferred or numeric_cols,
        default=ss.selected_fields,
        key="field_selector"
    )
    
    # Update cached selection
    ss.selected_fields = sel

    # Display mode: Separate vs Overlay (default Separate)
    display_mode = st.radio(
        "Display mode",
        ["Separate", "Overlay"],
        index=(0 if cfg.DEFAULT_DISPLAY_MODE == "Separate" else 1),
        horizontal=True,
        key="chart_display_mode",
    )

    # Convert time and plot with stable rendering
    if sel:
        # Create a stable chart container
        charts_placeholder = st.empty()

        def render_charts_from_rows(rows: List[Dict[str, Any]]):
            with charts_placeholder.container():
                df_local = pd.DataFrame(rows)
                df_local = df_local[["timestamp"] + sel].copy()
                df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], errors="coerce")
                df_local = df_local.dropna(subset=["timestamp"]) 
                if len(df_local) == 0:
                    st.info("No valid data points available for plotting")
                    return
                df_local = df_local.set_index("timestamp")
                # Light decimation when many points present
                if len(df_local) > 1200:
                    stride = max(1, len(df_local) // 300)
                    df_local = df_local.iloc[::stride]
                if display_mode == "Overlay" and len(sel) > 0:
                    df_reset = df_local.reset_index()
                    df_melt = df_reset.melt(
                        id_vars="timestamp",
                        value_vars=sel,
                        var_name="field",
                        value_name="value",
                    )
                    fig = px.line(
                        df_melt,
                        x="timestamp",
                        y="value",
                        color="field",
                        title="Selected fields",
                        height=400,
                    )
                    fig.update_layout(
                        yaxis=dict(type="linear", showgrid=True, zeroline=True),
                        xaxis=dict(showgrid=True),
                        uirevision="keep",  # preserve zoom/viewport
                        transition=dict(duration=200),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    for field_name in sel:
                        chart_data_reset = df_local[[field_name]].reset_index()
                        fig = px.line(
                            chart_data_reset,
                            x="timestamp",
                            y=field_name,
                            title=f"{field_name} over time",
                            height=300,
                        )
                        fig.update_layout(
                            yaxis=dict(type="linear", showgrid=True, zeroline=True),
                            xaxis=dict(showgrid=True),
                            uirevision="keep",
                            transition=dict(duration=200),
                        )
                        st.plotly_chart(fig, use_container_width=True)

        if getattr(cfg, "SMOOTH_UPDATES", False):
            end_time = time.time() + getattr(cfg, "SMOOTH_BURST_SECONDS", 6)
            sleep_s = max(0.05, ss.refresh_ms / 1000.0)
            ingest_interval = max(0.1, getattr(cfg, "INGEST_INTERVAL_S", 2.0))
            while time.time() < end_time:
                now = time.time()
                if now - ss.last_ingest_time >= ingest_interval:
                    if ss.mode == "Mock":
                        p = mock_payload(ss.last_mock_payload)
                        append_row_from_payload(p)
                        ss.last_mock_payload = p
                    else:
                        pulls = 0
                        while ss.ws_queue and pulls < 100:
                            _ts, msg = ss.ws_queue.popleft()
                            append_row_from_payload(msg)
                            pulls += 1
                    ss.last_ingest_time = now
                    render_charts_from_rows(ss.live_rows)
                time.sleep(sleep_s)
            # continue updating seamlessly
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()  # type: ignore
        else:
            # One-shot render (legacy mode)
            render_charts_from_rows(ss.live_rows)
    else:
        st.info("Select at least one numeric field.")

# (Performance tips removed by request)
