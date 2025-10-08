# Central configuration for the dashboard
# Adjust these settings as needed; UI controls are removed from the app.

# Data source: "Mock" or "WebSocket"
DATA_SOURCE = "Mock"

# WebSocket URL used when DATA_SOURCE == "WebSocket"
WS_URL = "ws://localhost:8000/stream"

# Auto-refresh interval in milliseconds (used when smooth updates are disabled)
REFRESH_MS = 2000

# Live window size in seconds for in-memory data
BUFFER_SECONDS = 10 * 60  # 10 minutes

# Maximum number of rows kept in the live buffer
MAX_ROWS = 20000

# Default chart display mode: "Separate" or "Overlay"
DEFAULT_DISPLAY_MODE = "Separate"

# Smooth update settings to minimize redraw flicker
# When True, charts update in-place in a short local loop without page reruns
SMOOTH_UPDATES = True
# How long each smooth update burst should run (seconds)
SMOOTH_BURST_SECONDS = 8
## Ingestion interval in seconds
INGEST_INTERVAL_S = 2.0
