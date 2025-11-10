import os
import streamlit as st
import pandas as pd
import plotly.express as px
from math import radians, sin, cos, sqrt, atan2
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# -------------------------------------
# FIX: Ensure Kaleido works on Streamlit Cloud
# -------------------------------------
os.environ["KALIEDO_SCOPE"] = "plotly"
os.environ["PATH"] += os.pathsep + "/usr/bin"
os.environ["KALIEDO_CHROME_PATH"] = "/usr/bin/chromium"

# -------------------------------------
# Safe defaults
# -------------------------------------
base_df, summary, range_df = None, None, None
pd.options.display.float_format = "{:.10f}".format


# -------------------------------------
# Haversine distance
# -------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# -------------------------------------
# Helpers for column detection
# -------------------------------------
EXPECTED_LABELS = {
    "emp_id": ["employee number", "emp id", "employee code", "emp no"],
    "emp_name": ["employee name", "name"],
    "timestamp": ["time stamp", "timestamp", "punch time", "time", "captured time", "date time", "date/time", "datetime"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "long", "lng", "lon"],
    "address": ["address", "location", "place"]
}


def _norm(s):
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def _find_column(cols, candidates):
    m = {c: _norm(c) for c in cols}
    for c, cn in m.items():
        for cand in candidates:
            if cand in cn:
                return c
    return None


# -------------------------------------
# Robust Excel reader
# -------------------------------------
def load_keka_sheet(file):
    tried = []
    for h in (1, 0, 2):
        try:
            df = pd.read_excel(file, header=h)
            tried.append(h)
            if df is None or len(df.columns) <= 1:
                continue

            emp_id_col = _find_column(df.columns, EXPECTED_LABELS["emp_id"])
            emp_nm_col = _find_column(df.columns, EXPECTED_LABELS["emp_name"])
            ts_col = _find_column(df.columns, EXPECTED_LABELS["timestamp"])
            lat_col = _find_column(df.columns, EXPECTED_LABELS["lat"])
            lon_col = _find_column(df.columns, EXPECTED_LABELS["lon"])
            addr_col = _find_column(df.columns, EXPECTED_LABELS["address"])

            if not all([ts_col, lat_col, lon_col]):
                continue

            out = pd.DataFrame({
                "EmployeeID": df[emp_id_col] if emp_id_col else None,
                "EmployeeName": df[emp_nm_col] if emp_nm_col else None,
                "Timestamp": pd.to_datetime(df[ts_col], errors="coerce", dayfirst=True),
                "Latitude": pd.to_numeric(df[lat_col], errors="coerce"),
                "Longitude": pd.to_numeric(df[lon_col], errors="coerce"),
            })
            out["Address"] = df[addr_col].astype(str) if addr_col else ""

            # Drop invalid rows
            out = out.dropna(subset=["Timestamp", "Latitude", "Longitude"])
            out = out[out["Latitude"].between(-90, 90) & out["Longitude"].between(-180, 180)]

            if out["EmployeeName"].isna().all() and emp_id_col:
                out["EmployeeName"] = out["EmployeeID"].astype(str)

            out["Date"] = out["Timestamp"].dt.date
            return out
        except Exception:
            continue
    raise ValueError(f"Could not parse Excel (tried headers: {tried})")


# -------------------------------------
# Distance calculations
# -------------------------------------
def add_interval_distances(day_df):
    day_df = day_df.sort_values("Timestamp").reset_index(drop=True)
    dists = [0.0]
    for i in range(len(day_df) - 1):
        d = haversine(day_df.loc[i, "Latitude"], day_df.loc[i, "Longitude"],
                      day_df.loc[i + 1, "Latitude"], day_df.loc[i + 1, "Longitude"])
        dists.append(d)
    day_df["Distance_from_prev_km"] = dists
    return day_df, sum(dists)


def summarize_range(df_range):
    rows = []
    for dt, sub in df_range.groupby("Date"):
        sub_sorted = sub.sort_values("Timestamp")
        total = 0.0
        prev = None
        for _, r in sub_sorted.iterrows():
            if prev is not None:
                total += haversine(prev["Latitude"], prev["Longitude"], r["Latitude"], r["Longitude"])
            prev = r
        rows.append({"Date": dt, "Points": len(sub_sorted), "Distance_km": total})
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


# -------------------------------------
# Map generation (dynamic zoom)
# -------------------------------------
def make_map_figure(day_df):
    if day_df.empty:
        return None

    lat_min, lat_max = day_df["Latitude"].min(), day_df["Latitude"].max()
    lon_min, lon_max = day_df["Longitude"].min(), day_df["Longitude"].max()
    max_dist_km = haversine(lat_min, lon_min, lat_max, lon_max)

    def compute_zoom(d):
        if d < 0.5: return 15
        elif d < 1: return 14
        elif d < 3: return 13
        elif d < 8: return 12
        elif d < 20: return 11
        elif d < 40: return 10
        elif d < 80: return 9
        elif d < 150: return 8
        else: return 7

    zoom = compute_zoom(max_dist_km)
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    fig = px.line_mapbox(day_df, lat="Latitude", lon="Longitude", hover_name="Timestamp", height=500)
    fig.add_scattermapbox(
        lat=day_df["Latitude"],
        lon=day_df["Longitude"],
        mode="markers+text",
        marker={"size": 9},
        text=[str(i + 1) for i in range(len(day_df))],
        textposition="top center",
        hoverinfo="text",
        name="Points",
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=zoom,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    return fig


# -------------------------------------
# FIXED: Safer image export with fallback
# -------------------------------------
def fig_to_png_bytes(fig, width=800, height=600, scale=2):
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=scale, engine="kaleido")
        return BytesIO(img_bytes)
    except Exception as e:
        print(f"[WARN] Failed to render figure: {e}")
        return None


# -------------------------------------
# PDF generator
# -------------------------------------
def build_pdf_all_days(employee_name, start_date, end_date, summary_df, range_df):
    styles = getSampleStyleSheet()
    buff = BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    story = []

    # Summary page
    story += [
        Paragraph("<b>Travel Report</b>", styles["Title"]),
        Paragraph(f"<b>Employee:</b> {employee_name}", styles["Normal"]),
        Paragraph(f"<b>Period:</b> {start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}", styles["Normal"]),
        Spacer(1, 12),
    ]
    story.append(Paragraph("<b>Daily Summary (Displacement)</b>", styles["Heading3"]))

    summary_tbl_data = [["Date", "Points", "Distance (km)"]]
    for _, r in summary_df.iterrows():
        summary_tbl_data.append([r["Date"].strftime("%d-%b-%Y"), str(int(r["Points"])), f"{r['Distance_km']:.2f}"])
    summary_tbl_data.append(["TOTAL", "", f"{summary_df['Distance_km'].sum():.2f}"])

    tbl = Table(summary_tbl_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story += [tbl, Spacer(1, 12)]

    # Each day page
    for dt, sub in sorted(range_df.groupby("Date"), key=lambda x: x[0]):
        story.append(PageBreak())
        story.append(Paragraph(f"<b>{pd.to_datetime(dt).strftime('%d-%b-%Y')} — Detailed Intervals</b>", styles["Heading3"]))

        day_df = sub.sort_values("Timestamp").reset_index(drop=True)
        day_df, _ = add_interval_distances(day_df)

        det_tbl_data = [["#", "Timestamp", "Latitude", "Longitude", "Δ Distance (km)"]]
        for j, r in day_df.iterrows():
            det_tbl_data.append([
                str(j + 1),
                pd.to_datetime(r["Timestamp"]).strftime("%d-%b-%Y %H:%M:%S"),
                f"{r['Latitude']:.8f}",
                f"{r['Longitude']:.8f}",
                f"{r['Distance_from_prev_km']:.4f}",
            ])
        det_tbl = Table(det_tbl_data, repeatRows=1)
        det_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (4, 1), (4, -1), "RIGHT"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ]))
        story.append(det_tbl)
        story.append(Spacer(1, 10))

        fig = make_map_figure(day_df)
        if fig:
            png_buf = fig_to_png_bytes(fig)
            if png_buf:
                png_buf.seek(0)
                story.append(Image(png_buf, width=420, height=315))
            else:
                story.append(Paragraph(f"(Map could not render for {dt})", styles["Normal"]))
        else:
            story.append(Paragraph(f"(No map data for {dt})", styles["Normal"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    pdf = buff.getvalue()
    buff.close()
    return pdf


# -------------------------------------
# Streamlit UI
# -------------------------------------
st.title("📍 Keka Unified — Travel Summary + Maps + PDF")

uploaded = st.file_uploader("Upload Keka Excel file", type=["xlsx"])
if uploaded is None:
    st.info("👆 Upload your Keka Excel file to get started.")
    st.stop()

try:
    base_df = load_keka_sheet(uploaded)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

display_name = base_df["EmployeeName"].fillna(base_df["EmployeeID"].astype(str))
employees = sorted(display_name.dropna().unique())
emp = st.selectbox("Select Employee", employees)

emp_df = base_df[display_name == emp].copy()
if emp_df.empty:
    st.warning("No records for this employee.")
    st.stop()

min_d, max_d = emp_df["Date"].min(), emp_df["Date"].max()
c1, c2 = st.columns(2)
start_date = c1.date_input("Start date", min_d, min_value=min_d, max_value=max_d)
end_date = c2.date_input("End date", max_d, min_value=min_d, max_value=max_d)

if start_date > end_date:
    st.error("Start date must be before End date.")
    st.stop()

mask = (emp_df["Date"] >= start_date) & (emp_df["Date"] <= end_date)
range_df = emp_df[mask].copy().sort_values("Timestamp")

if range_df.empty:
    st.warning("No punches found in selected date range.")
    st.stop()

summary = summarize_range(range_df)
grand_total = summary["Distance_km"].sum()

st.subheader("📊 Daily Summary (Displacement)")
st.dataframe(summary.assign(Date=summary["Date"].astype(str)), use_container_width=True)
st.success(f"Total distance for range: **{grand_total:.2f} km**")

available_dates = sorted(summary["Date"].tolist())
sel_date = st.selectbox("🔍 View details for a specific day", available_dates)

day_df = range_df[range_df["Date"] == sel_date].copy().sort_values("Timestamp").reset_index(drop=True)
if not day_df.empty:
    day_df, total_for_day = add_interval_distances(day_df)
    st.info(f"{emp} — {sel_date.strftime('%d-%b-%Y')} — {len(day_df)} points, {total_for_day:.2f} km")
    st.dataframe(day_df[["Timestamp", "Latitude", "Longitude", "Distance_from_prev_km"]],
                 use_container_width=True)
    fig = make_map_figure(day_df)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("📄 Export Report")
if st.button("Generate & Download PDF"):
    with st.spinner("Generating report..."):
        try:
            pdf_bytes = build_pdf_all_days(emp, pd.to_datetime(start_date), pd.to_datetime(end_date), summary, range_df)
            fname = f"{emp.replace(' ', '_')}_{start_date.strftime('%d-%b-%Y')}_to_{end_date.strftime('%d-%b-%Y')}.pdf"
            st.download_button("Download PDF", data=pdf_bytes, file_name=fname, mime="application/pdf")
        except Exception as e:
            st.error(f"Failed to build PDF: {e}")
