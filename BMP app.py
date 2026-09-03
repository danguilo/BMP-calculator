from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64
import html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="BMP Analyzer",
    page_icon="BMP",
    layout="wide",
)


REFERENCE_VALUES = {
    "Cellulose": 360,
    "Cow Manure": 200,
    "Food Waste": 450,
    "Grass Silage": 350,
    "Corn Silage": 320,
    "Pig Manure": 300,
    "Wastewater Sludge": 240,
}


@st.cache_data(show_spinner=False)
def read_workbook(uploaded_file: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    workbook = pd.ExcelFile(BytesIO(uploaded_file), engine="openpyxl")
    sheet_lookup = {name.strip().lower(): name for name in workbook.sheet_names}

    raw_name = sheet_lookup.get("raw data")
    doe_name = sheet_lookup.get("doe")
    if raw_name is None or doe_name is None:
        raise ValueError(
            "The workbook must contain sheets named 'Raw data' and 'DOE'. "
            f"Found: {', '.join(workbook.sheet_names)}"
        )

    raw_sheet = pd.read_excel(workbook, sheet_name=raw_name, header=None)
    header_rows = raw_sheet.apply(
        lambda row: row.astype(str).str.strip().str.lower().isin({"run id", "data"}).any(),
        axis=1,
    )
    if not header_rows.any():
        raise ValueError("Could not find the header row in the Raw data sheet.")
    header_index = header_rows[header_rows].index[0]
    raw_data = raw_sheet.iloc[header_index + 1 :].copy()
    raw_data.columns = [
        str(value).strip() if pd.notna(value) else f"Unnamed_{index}"
        for index, value in enumerate(raw_sheet.iloc[header_index])
    ]
    raw_data = raw_data.rename(
        columns={column: "Run ID" for column in raw_data.columns if str(column).lower() == "data"}
    )
    raw_data = raw_data.dropna(how="all").reset_index(drop=True)

    doe_sheet = pd.read_excel(workbook, sheet_name=doe_name, header=None)
    doe_header_rows = doe_sheet.apply(
        lambda row: row.astype(str).str.strip().str.lower().isin({"type", "run id"}).sum() >= 2,
        axis=1,
    )
    if not doe_header_rows.any():
        raise ValueError("Could not find the header row in the DOE sheet.")
    doe_header_index = doe_header_rows[doe_header_rows].index[0]
    doe_data = doe_sheet.iloc[doe_header_index + 1 :].copy()
    doe_data.columns = [
        str(value).strip() if pd.notna(value) else f"Unnamed_{index}"
        for index, value in enumerate(doe_sheet.iloc[doe_header_index])
    ]
    return raw_data.dropna(how="all").reset_index(drop=True), doe_data.dropna(how="all").reset_index(drop=True)


def clean_raw_data(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"Run ID", "Day", "Daily Biogas (ml)", "CH4%"}
    missing = sorted(required - set(raw_data.columns))
    if missing:
        raise ValueError("Missing required Raw data columns: " + ", ".join(missing))

    data = raw_data.copy()
    data["Run ID"] = data["Run ID"].astype(str).str.strip()
    for column in ["Day", "Daily Biogas (ml)", "CH4%", "CO2%"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    summaries = []
    cleaned_runs = []
    for run_id, run_data in data.dropna(subset=["Run ID"]).groupby("Run ID", sort=False):
        run_data = run_data.sort_values("Day").copy()
        duplicate_mask = run_data.duplicated("Day", keep="first")
        duplicate_count = int(duplicate_mask.sum())
        run_data = run_data.loc[~duplicate_mask].copy()
        days = sorted(run_data["Day"].dropna().unique())
        expected_days = list(range(int(min(days)), int(max(days)) + 1)) if days else []
        missing_days = sorted(set(expected_days) - set(days))

        interpolated = 0
        for column in ["Daily Biogas (ml)", "CH4%"]:
            null_count = int(run_data[column].isna().sum())
            if null_count:
                run_data[column] = run_data[column].interpolate(method="linear", limit_area="inside")
                interpolated += null_count

        run_data["Daily Methane (ml)"] = run_data["Daily Biogas (ml)"] * run_data["CH4%"] / 100
        run_data["Cumulative Methane (mL)"] = run_data["Daily Methane (ml)"].cumsum()
        run_data["Cumulative Biogas (mL)"] = run_data["Daily Biogas (ml)"].cumsum()
        cleaned_runs.append(run_data)
        summaries.append(
            {
                "Run ID": run_id,
                "First day": min(days) if days else np.nan,
                "Last day": max(days) if days else np.nan,
                "Recorded timepoints": len(days),
                "Missing days": ", ".join(map(str, missing_days)) or "None",
                "Duplicate rows": duplicate_count,
                "Values interpolated": interpolated,
            }
        )

    if not cleaned_runs:
        raise ValueError("No valid run rows were found.")
    return pd.concat(cleaned_runs, ignore_index=True), pd.DataFrame(summaries)


def plateau_results(processed: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    incomplete = set(checks.loc[checks["Missing days"] != "None", "Run ID"])
    results = []
    for run_id, run_data in processed.groupby("Run ID", sort=False):
        run_data = run_data.sort_values("Day")
        total = run_data["Cumulative Methane (mL)"].iloc[-1]
        last_three = run_data.tail(3)["Daily Methane (ml)"].mean()
        criterion = last_three / total * 100 if total and pd.notna(total) else np.nan
        if run_id in incomplete:
            status = "INCOMPLETE TIME SERIES"
            reached = False
        elif len(run_data) < 3 or pd.isna(criterion):
            status = "NOT ENOUGH VALID DATA"
            reached = False
        else:
            reached = bool(criterion < 1)
            status = "REACHED PLATEAU" if reached else "NOT REACHED PLATEAU"
        results.append(
            {
                "Run ID": run_id,
                "Status": status,
                "Time series complete": run_id not in incomplete,
                "Days recorded": len(run_data),
                "Total Methane (mL)": total,
                "% of Total in last 3 days": criterion,
            }
        )
    return pd.DataFrame(results)


def calculate_bmp(processed: pd.DataFrame, doe: pd.DataFrame, plateau: pd.DataFrame) -> pd.DataFrame:
    required = {"Type", "Run ID"}
    if not required.issubset(doe.columns):
        raise ValueError("DOE must contain Type and Run ID columns.")

    doe = doe.copy()
    doe["Run ID"] = doe["Run ID"].astype(str).str.strip()
    blanks = doe[doe["Type"].astype(str).str.contains("Blank", case=False, na=False)]["Run ID"].unique()
    blank_data = processed[processed["Run ID"].isin(blanks)]
    if blank_data.empty:
        raise ValueError("No blank controls were found in the raw data.")

    blank_average = blank_data.groupby("Day")["Cumulative Methane (mL)"].mean().rename("Blank methane")
    corrected = processed.join(blank_average, on="Day")
    corrected["Net methane"] = corrected["Cumulative Methane (mL)"] - corrected["Blank methane"]

    records = []
    for _, row in doe.iterrows():
        run_id = row["Run ID"]
        if run_id in blanks:
            continue
        mass = pd.to_numeric(row.get("g.1"), errors="coerce")
        vs_percent = pd.to_numeric(row.get("VS_FS"), errors="coerce")
        if pd.isna(mass) or pd.isna(vs_percent) or mass <= 0 or vs_percent <= 0:
            continue
        vs_grams = mass * vs_percent / 100
        run_data = corrected[corrected["Run ID"] == run_id].sort_values("Day")
        if run_data.empty or vs_grams <= 0:
            continue
        records.append(
            {
                "Run ID": run_id,
                "Type": row.get("Type", ""),
                "VS (g)": vs_grams,
                "Final net methane (mL)": run_data["Net methane"].iloc[-1],
                "BMP (mL CH4/g VS)": run_data["Net methane"].iloc[-1] / vs_grams,
            }
        )
    return pd.DataFrame(records)


def make_report(processed: pd.DataFrame, checks: pd.DataFrame, plateau: pd.DataFrame, bmp: pd.DataFrame, source_name: str) -> str:
    figure, axis = plt.subplots(figsize=(10, 5))
    for run_id, run_data in processed.groupby("Run ID"):
        axis.plot(run_data["Day"], run_data["Cumulative Methane (mL)"], marker="o", label=run_id)
    axis.set_title("Cumulative methane by run")
    axis.set_xlabel("Day")
    axis.set_ylabel("Cumulative methane (mL)")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    chart = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>BMP report</title>
<style>body{{font-family:Arial,sans-serif;max-width:1150px;margin:32px auto;padding:0 24px;color:#202124}}table{{border-collapse:collapse;width:100%;margin:12px 0 24px}}th,td{{border:1px solid #d8e0e3;padding:8px;text-align:right}}th{{background:#e7f2f0}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%}}</style></head><body>
<h1>BMP analysis report</h1><p>Source workbook: {html.escape(source_name)}</p>
<h2>Data checks</h2>{checks.to_html(index=False)}<h2>Plateau results</h2>{plateau.to_html(index=False)}
<h2>BMP results</h2>{bmp.to_html(index=False) if not bmp.empty else '<p>No BMP results were generated.</p>'}
<h2>Cumulative methane</h2><img src='{chart}' alt='Cumulative methane by run'></body></html>"""


def main() -> None:
    st.title("BMP Analyzer")
    st.caption("Upload an Excel workbook containing Raw data and DOE sheets.")
    uploaded = st.file_uploader("Excel workbook", type=["xlsx", "xlsm", "xls"])
    if uploaded is None:
        st.info("Upload a workbook to begin.")
        return

    try:
        raw_data, doe_data = read_workbook(uploaded.getvalue())
        processed, checks = clean_raw_data(raw_data)
        plateau = plateau_results(processed, checks)
    except Exception as exc:
        st.error(str(exc))
        return

    bmp = pd.DataFrame()
    with st.sidebar:
        st.header("Analysis")
        calculate = st.checkbox("Calculate blank-corrected BMP", value=True)
        report_name = f"{Path(uploaded.name).stem}_BMP_report.html"

    if calculate:
        try:
            bmp = calculate_bmp(processed, doe_data, plateau)
        except Exception as exc:
            st.warning(f"BMP calculation unavailable: {exc}")

    tab_overview, tab_series, tab_plateau, tab_bmp, tab_report = st.tabs(
        ["Overview", "Time series", "Plateau", "BMP", "Report"]
    )
    with tab_overview:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Runs", processed["Run ID"].nunique())
        metric_columns[1].metric("Raw rows", len(raw_data))
        metric_columns[2].metric("First day", int(processed["Day"].min()))
        metric_columns[3].metric("Last day", int(processed["Day"].max()))
        st.subheader("Data checks")
        st.dataframe(checks, use_container_width=True, hide_index=True)
        st.subheader("Raw data")
        st.dataframe(raw_data, use_container_width=True, hide_index=True)

    with tab_series:
        selected_runs = st.multiselect(
            "Runs to display",
            options=processed["Run ID"].unique().tolist(),
            default=processed["Run ID"].unique().tolist(),
        )
        figure, axis = plt.subplots(figsize=(12, 6))
        for run_id in selected_runs:
            run_data = processed[processed["Run ID"] == run_id]
            axis.plot(run_data["Day"], run_data["Cumulative Biogas (mL)"], marker="o", label=run_id)
        axis.set_xlabel("Day")
        axis.set_ylabel("Cumulative biogas (mL)")
        axis.set_title("Cumulative biogas by run")
        axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        axis.grid(alpha=0.25)
        st.pyplot(figure)
        plt.close(figure)

    with tab_plateau:
        st.dataframe(plateau, use_container_width=True, hide_index=True)

    with tab_bmp:
        if bmp.empty:
            st.info("No BMP results are available. Check the DOE mass and VS_FS columns.")
        else:
            st.dataframe(bmp, use_container_width=True, hide_index=True)
            figure, axis = plt.subplots(figsize=(10, 5))
            axis.bar(bmp["Run ID"], bmp["BMP (mL CH4/g VS)"])
            axis.set_ylabel("BMP (mL CH4/g VS)")
            axis.set_xlabel("Run ID")
            axis.tick_params(axis="x", rotation=45)
            st.pyplot(figure)
            plt.close(figure)

    with tab_report:
        report = make_report(processed, checks, plateau, bmp, uploaded.name)
        st.download_button(
            "Download HTML report",
            data=report,
            file_name=report_name,
            mime="text/html",
        )
        st.components.v1.html(report, height=700, scrolling=True)


if __name__ == "__main__":
    main()
