# Excluded preliminary multistream run

These 18 short runs were collected before the clean matrix in `../raw/`. The board API test harness launched `edge_station.server` through an intermediate shell; terminating the shell left the Python service and its Worker alive. A later `/proc` audit found both processes still running. Their overlap with the multistream matrix could alter NPU scheduling, CPU, RSS, and temperature, so the results are preserved here for audit but are excluded from `summary.csv`, `summary.json`, the SVG, and all report conclusions.

The service and Worker were stopped after verifying their command lines belonged to this isolated staging directory. The board then reported no matching process. The idle NPU check still returned `100@300000000Hz` with runtime status `suspended`, so utilization remained unavailable. All 1/2/4/8/16/32 conditions were rerun three times after that cleanup.

Do not compare these `run-index.json` values against the clean matrix as if they were an independent measurement.
