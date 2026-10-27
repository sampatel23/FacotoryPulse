import os
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="FactoryPulse | Industrial Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
DATABASE_NAME = "factorypulse"
COLLECTION_NAME = "machine_predictions"
REFRESH_INTERVAL = 5000

# --------------------------------------------------
# 2. Custom styling
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: #0b1020;
        color: #e8edf7;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #263249;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9;
    }

    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }

    .fp-eyebrow {
        color: #56d6c8;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .fp-title {
        color: #f8fafc;
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -1.2px;
        margin: 0;
        line-height: 1.2;
    }

    .fp-subtitle {
        color: #9aa9c0;
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }

    .fp-card {
        background: linear-gradient(145deg, #172238, #111a2b);
        border: 1px solid #29364d;
        border-radius: 15px;
        padding: 20px;
        min-height: 145px;
        margin-bottom: 8px;
    }

    .fp-card-label {
        color: #a7b6cc;
        font-size: 0.84rem;
        font-weight: 500;
        margin-bottom: 13px;
    }

    .fp-card-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 750;
        letter-spacing: -0.8px;
        line-height: 1.2;
    }

    .fp-card-note {
        color: #8394ae;
        font-size: 0.76rem;
        margin-top: 12px;
    }

    .fp-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 7px;
    }

    .fp-section {
        color: #f1f5f9;
        font-size: 1.18rem;
        font-weight: 700;
        margin: 0.6rem 0 0.2rem 0;
    }

    .fp-muted {
        color: #91a0b6;
        font-size: 0.83rem;
    }

    div[data-testid="stPlotlyChart"] {
        background: #111827;
        border: 1px solid #263249;
        border-radius: 14px;
        padding: 8px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #29364d;
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #29364d;
        border-radius: 12px;
        padding: 12px;
    }

    .stButton > button {
        border-radius: 9px;
        border: 1px solid #34445e;
        min-height: 42px;
        font-weight: 600;
    }

    hr {
        border-color: #263249;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. Automatic refresh
# --------------------------------------------------
st_autorefresh(
    interval=REFRESH_INTERVAL,
    key="factorypulse_refresh",
)

st.session_state["refresh_count"] = (
    st.session_state.get("refresh_count", 0) + 1
)

# --------------------------------------------------
# 4. MongoDB data loading
# --------------------------------------------------
@st.cache_data(ttl=4, show_spinner=False)
def load_records():
    """Load the most recent prediction records from MongoDB."""
    client = None

    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
        )

        client.admin.command("ping")

        records = list(
            client[DATABASE_NAME][COLLECTION_NAME]
            .find({}, {"_id": 0})
            .sort("_id", -1)
            .limit(2000)
        )

        return records, None

    except PyMongoError as exc:
        return [], str(exc)

    finally:
        if client is not None:
            client.close()


# --------------------------------------------------
# 5. Header
# --------------------------------------------------
header_left, header_right = st.columns([5, 1.3])

with header_left:
    st.markdown(
        '<div class="fp-eyebrow">Industrial intelligence platform</div>'
        '<div class="fp-title">🏭 FactoryPulse</div>'
        '<div class="fp-subtitle">'
        'Real-time equipment health monitoring & predictive analytics'
        '</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.write("")
    if st.button("↻ Refresh dashboard", use_container_width=True):
        load_records.clear()
        st.rerun()

st.write("")

records, error = load_records()

if error:
    st.error(f"MongoDB connection failed: {error}")
    st.info(
        "Check that MongoDB is running and reachable at "
        "127.0.0.1:27017."
    )
    st.stop()

if not records:
    st.info(
        "No predictions found yet. Start Kafka, the sensor producer, "
        "and the ML inference consumer."
    )
    st.caption(
        "The dashboard will check MongoDB again automatically."
    )
    st.stop()

# --------------------------------------------------
# 6. Prepare records
# --------------------------------------------------
df = pd.json_normalize(records, sep="_")

df["machine_id"] = (
    df.get("machine_id", pd.Series("unknown", index=df.index))
    .fillna("unknown")
    .astype(str)
)

df["alert_level"] = (
    df.get("alert_level", pd.Series("UNKNOWN", index=df.index))
    .fillna("UNKNOWN")
    .astype(str)
    .str.upper()
)

df["failure_probability"] = pd.to_numeric(
    df.get(
        "failure_probability",
        pd.Series(index=df.index, dtype=float),
    ),
    errors="coerce",
)

df["timestamp_parsed"] = pd.to_datetime(
    df.get(
        "timestamp",
        pd.Series(index=df.index, dtype=object),
    ),
    errors="coerce",
    utc=True,
)

# Sort newest first for tables and KPI calculations.
df = df.sort_values("timestamp_parsed", ascending=False, na_position="last")

# --------------------------------------------------
# 7. Sidebar filters
# --------------------------------------------------
st.sidebar.markdown("## Dashboard filters")
st.sidebar.caption("Customize your monitoring view.")

machines = sorted(df["machine_id"].unique().tolist())
alerts = sorted(df["alert_level"].unique().tolist())

selected_machines = st.sidebar.multiselect(
    "Machines",
    machines,
    default=machines,
)

selected_alerts = st.sidebar.multiselect(
    "Alert levels",
    alerts,
    default=alerts,
)

st.sidebar.divider()

st.sidebar.markdown("### System information")
st.sidebar.caption("Data source: MongoDB")
st.sidebar.caption(f"Database: {DATABASE_NAME}")
st.sidebar.caption(f"Collection: {COLLECTION_NAME}")
st.sidebar.caption("Dashboard refresh: 5 seconds")

data = df[
    df["machine_id"].isin(selected_machines)
    & df["alert_level"].isin(selected_alerts)
].copy()

# --------------------------------------------------
# 8. KPI cards
# --------------------------------------------------
latest_risks = data.dropna(subset=["failure_probability"])
latest_risk = (
    float(latest_risks.iloc[0]["failure_probability"])
    if not latest_risks.empty
    else None
)

reading_count = len(data)
machine_count = data["machine_id"].nunique()
warning_count = int((data["alert_level"] == "WARNING").sum())
warning_pct = (
    warning_count / reading_count * 100
    if reading_count else 0
)

last_timestamp = (
    data["timestamp_parsed"].max()
    if not data.empty else pd.NaT
)

st.markdown(
    '<div class="fp-section">Operations overview</div>',
    unsafe_allow_html=True,
)
st.caption(
    f"Last dashboard refresh: "
    f"{datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M:%S UTC')}"
)

k1, k2, k3, k4 = st.columns(4)

cards = [
    (
        k1, "READINGS LOADED", f"{reading_count:,}",
        "Records matching filters", "#60a5fa"
    ),
    (
        k2, "MACHINES MONITORED", f"{machine_count:,}",
        "Distinct selected machines", "#56d6c8"
    ),
    (
        k3, "WARNING ALERTS", f"{warning_count:,}",
        f"{warning_pct:.1f}% of displayed readings", "#fbbf24"
    ),
    (
        k4, "LATEST DISPLAYED RISK",
        f"{latest_risk:.1%}" if latest_risk is not None else "N/A",
        "Model-estimated failure probability", "#c4b5fd"
    ),
]

for col, label, value, note, accent in cards:
    with col:
        st.markdown(
            f"""
            <div class="fp-card" style="border-top: 3px solid {accent};">
                <div class="fp-card-label">{label}</div>
                <div class="fp-card-value">{value}</div>
                <div class="fp-card-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if pd.notna(last_timestamp):
    st.caption(
        "Newest timestamp in selected records: "
        + last_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    )

# --------------------------------------------------
# 9. Shared chart theme
# --------------------------------------------------
PLOT_BG = "#111827"
PAPER_BG = "#111827"
GRID_COLOR = "#29364d"
TEXT_COLOR = "#dce5f2"
MACHINE_COLORS = px.colors.qualitative.Safe

def style_chart(fig, height=350):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR, family="Arial", size=12),
        height=height,
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_color="#ffffff",
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=GRID_COLOR,
    )

    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zeroline=False,
    )

    return fig

# --------------------------------------------------
# 10. Risk trend and alert distribution
# --------------------------------------------------
st.markdown(
    '<div class="fp-section">Equipment health analytics</div>',
    unsafe_allow_html=True,
)
st.caption("Explore model-estimated risk and the distribution of alerts.")

left, right = st.columns([1.65, 1])

with left:
    st.markdown("#### Failure risk over time")

    trend = data.dropna(
        subset=["failure_probability", "timestamp_parsed"]
    ).copy()

    # Bound the chart size and reduce overplotting.
    trend = trend.sort_values("timestamp_parsed").tail(500)

    if not trend.empty:
        risk_fig = px.line(
            trend,
            x="timestamp_parsed",
            y="failure_probability",
            color="machine_id",
            color_discrete_sequence=MACHINE_COLORS,
            labels={
                "timestamp_parsed": "Timestamp (UTC)",
                "failure_probability": "Estimated failure probability",
                "machine_id": "Machine",
            },
        )

        risk_fig.update_yaxes(
            tickformat=".0%",
            rangemode="tozero",
            title="Failure probability",
        )

        risk_fig.update_xaxes(
            tickformat="%d %b %H:%M",
            nticks=6,
            title="Time (UTC)",
        )

        style_chart(risk_fig, 390)

        st.plotly_chart(
            risk_fig,
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )
    else:
        st.info("No timestamped risk data matches the selected filters.")

with right:
    st.markdown("#### Alert distribution")

    alert_counts = (
        data["alert_level"]
        .value_counts()
        .rename_axis("Alert")
        .reset_index(name="Readings")
    )

    if not alert_counts.empty:
        alert_color_map = {
            "NORMAL": "#56d6c8",
            "WARNING": "#fbbf24",
        }

        alert_fig = px.bar(
            alert_counts,
            x="Alert",
            y="Readings",
            color="Alert",
            color_discrete_map=alert_color_map,
            text="Readings",
            labels={"Readings": "Number of readings"},
        )

        alert_fig.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        style_chart(alert_fig, 390)
        alert_fig.update_layout(showlegend=False)

        st.plotly_chart(
            alert_fig,
            use_container_width=True,
            config={"displaylogo": False, "responsive": True},
        )
    else:
        st.info("No alerts match these filters.")

# --------------------------------------------------
# 11. Sensor trends
# --------------------------------------------------
st.markdown(
    '<div class="fp-section">Sensor telemetry</div>',
    unsafe_allow_html=True,
)
st.caption("Inspect sensor readings over time for each selected machine.")

sensors = {
    "sensor_data_air_temp_K": "Air Temperature (K)",
    "sensor_data_process_temp_K": "Process Temperature (K)",
    "sensor_data_rotational_speed_rpm": "Rotational Speed (rpm)",
    "sensor_data_torque_nm": "Torque (Nm)",
    "sensor_data_tool_wear_min": "Tool Wear (min)",
    "sensor_data_vibration_mm_s": "Vibration (mm/s)",
    "sensor_data_hydraulic_pressure_bar": "Hydraulic Pressure (bar)",
}

available_sensors = {
    key: label
    for key, label in sensors.items()
    if key in data.columns
}

if available_sensors:
    sensor_left, sensor_right = st.columns([1, 3])

    with sensor_left:
        selected_sensor = st.selectbox(
            "Choose sensor",
            list(available_sensors.keys()),
            format_func=lambda key: available_sensors[key],
        )

        st.markdown("#### Sensor details")
        st.caption(available_sensors[selected_sensor])
        st.caption("Use hover to inspect individual readings.")

    with sensor_right:
        sensor_data = data.copy()

        sensor_data[selected_sensor] = pd.to_numeric(
            sensor_data[selected_sensor],
            errors="coerce",
        )

        sensor_data = sensor_data.dropna(
            subset=[selected_sensor, "timestamp_parsed"]
        ).sort_values("timestamp_parsed").tail(500)

        if not sensor_data.empty:
            
            sensor_fig = px.area(
                sensor_data,
                x="timestamp_parsed",
                y=selected_sensor,
                color="machine_id",
                color_discrete_sequence=["#56d6c8", "#60a5fa", "#c4b5fd"],
                labels={
                    "timestamp_parsed": "Timestamp (UTC)",
                    selected_sensor: available_sensors[selected_sensor],
                    "machine_id": "Machine",
                },
                groupnorm=None,
            )

# Add a dashed reference line for the average sensor reading.
            sensor_average = sensor_data[selected_sensor].mean()

            sensor_fig.add_hline(
                y=sensor_average,
                line_dash="dash",
                line_color="#fbbf24",
                annotation_text=f"Average: {sensor_average:.2f}",
                annotation_position="top left",
                annotation_font_color="#fbbf24",
            )

            sensor_fig.update_xaxes(
                tickformat="%d %b %H:%M",
                nticks=6,
                title="Time (UTC)",
            )

            style_chart(sensor_fig, 330)

            st.plotly_chart(
                sensor_fig,
                use_container_width=True,
                config={"displaylogo": False, "responsive": True},
            )
        else:
            st.info("No sensor values available for this selection.")
else:
    st.info("No supported sensor fields were found in the database.")

# --------------------------------------------------
# 12. Recent predictions
# --------------------------------------------------
st.markdown(
    '<div class="fp-section">Recent predictions</div>',
    unsafe_allow_html=True,
)
st.caption("Latest stored inference results matching your filters.")

display_columns = [
    "machine_id",
    "timestamp",
    "alert_level",
    "failure_probability",
    "threshold",
    "predicted_failure",
    "sensor_data_air_temp_K",
    "sensor_data_process_temp_K",
    "sensor_data_rotational_speed_rpm",
    "sensor_data_torque_nm",
    "sensor_data_tool_wear_min",
]

display_columns = [
    column for column in display_columns if column in data.columns
]

recent = data[display_columns].head(100).copy()

rename_columns = {
    "machine_id": "Machine",
    "timestamp": "Timestamp (UTC)",
    "alert_level": "Alert",
    "failure_probability": "Failure Risk",
    "threshold": "Threshold",
    "predicted_failure": "Predicted Failure",
    "sensor_data_air_temp_K": "Air Temp (K)",
    "sensor_data_process_temp_K": "Process Temp (K)",
    "sensor_data_rotational_speed_rpm": "Speed (rpm)",
    "sensor_data_torque_nm": "Torque (Nm)",
    "sensor_data_tool_wear_min": "Tool Wear (min)",
}

recent = recent.rename(columns=rename_columns)

if "Failure Risk" in recent.columns:
    recent["Failure Risk"] = recent["Failure Risk"].map(
        lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
    )

if "Threshold" in recent.columns:
    recent["Threshold"] = recent["Threshold"].map(
        lambda value: f"{value:.0%}" if pd.notna(value) else "N/A"
    )

if not recent.empty:
    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True,
        height=380,
    )
else:
    st.info("No predictions match the selected filters.")

# --------------------------------------------------
# 13. Model limitation notice and footer
# --------------------------------------------------
st.markdown("---")

st.info(
    "Model estimates are not confirmed equipment faults. The classifier "
    "was trained on the AI4I 2020 benchmark dataset. Predictions on "
    "synthetic telemetry demonstrate pipeline integration and are not "
    "verified real-world fault diagnoses."
)

st.markdown(
    '<div style="text-align:center;color:#718096;font-size:0.78rem;'
    'padding:10px 0;">'
    'FactoryPulse · Industrial Equipment Health Monitoring · '
    'Local monitoring prototype'
    '</div>',
    unsafe_allow_html=True,
)
