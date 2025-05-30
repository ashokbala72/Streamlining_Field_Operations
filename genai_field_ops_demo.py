import streamlit as st
import pandas as pd
import datetime
import random
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env
load_dotenv()
client = OpenAI()

# ---------- CSS FIX FOR TABS ----------
st.markdown("""
<style>
.stTabs [role="tablist"] {
    flex-wrap: wrap;
    justify-content: start;
}
.stTabs [data-baseweb="tab"] {
    font-size: 15px;
    padding: 10px;
    margin-right: 5px;
}
</style>
""", unsafe_allow_html=True)

# ---------- UTILITIES ----------
def normalize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def convert_to_numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def simulate_new_fault(n=1):
    if "simulated_faults" not in st.session_state:
        st.session_state.simulated_faults = []

    fault_codes = ["TRF101", "GEN203", "CB404", "INVT555"]
    equipments = ["Transformer", "Generator", "Circuit Breaker", "Inverter"]
    locations = ["Zone A", "Zone B", "Zone C", "Zone D"]

    for _ in range(n):
        timestamp = datetime.datetime.now() - datetime.timedelta(minutes=random.randint(0, 120))
        fault = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "fault_code": random.choice(fault_codes),
            "equipment": random.choice(equipments),
            "location": random.choice(locations),
            "status": "New",
            "assigned_time": None,
            "in_progress_time": None,
            "closed_time": None,
            "technician": None
        }
        st.session_state.simulated_faults.append(fault)

# Real GenAI call using updated OpenAI client
def genai_response(prompt: str, stream: bool = True) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful field operations assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.7,
            stream=stream
        )

        if stream:
            collected_chunks = []
            for chunk in response:
                chunk_message = getattr(chunk.choices[0].delta, "content", "") or ""
                collected_chunks.append(chunk_message)
            return "".join(collected_chunks).strip()
        else:
            return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❗️ GenAI API error: {e}"

def calculate_ineffective_techs(faults_df):
    tech_perf = faults_df.groupby("technician").apply(
        lambda df: pd.Series({
            "unresolved_faults": df["closed_time"].isnull().sum(),
            "pending_assignments": df["status"].isin(["New", "Assigned"]).sum()
        })
    ).reset_index()
    tech_perf["total_issues"] = tech_perf["unresolved_faults"] + tech_perf["pending_assignments"]
    return tech_perf.sort_values("total_issues", ascending=False)

# ---------- Load CSV Data ----------
fault_history = normalize_columns(pd.read_csv("fault_history_uk_template.csv"))
technicians = normalize_columns(pd.read_csv("technicians_uk_template.csv"))

# ---------- TABS ----------
tabs = st.tabs([
    "📊 Overview",
    "🚨 Live Faults",
    "🛠 Work Orders",
    "📈 Dashboard",
    "🤖 Supervisor Chat",
    "📉 Ineffective Techs",
    "🏢 Management",
    "📊 Staffing Adequacy",
    "🧠 Predictive Risk Advisory"
])

# ---------- Overview ----------
with tabs[0]:
    st.title("GenAI Field Operations Assistant")
    st.markdown("""
    This GenAI Field Operations Assistant is designed to streamline end-to-end fault management for field teams using real-time data and AI. Here’s what it includes:

    - 🔍 **Root Cause Analysis**: GenAI explains likely causes of faults based on historical patterns.
    - 🔧 **Maintenance Recommendations**: Work orders include fix suggestions and risk summaries.
    - 👷 **Technician Suitability & Assignment**: Technicians are matched using skills and proximity.
    - 🚨 **Live Fault Simulation**: Simulates new faults, tracks status changes from New → Assigned → In Progress → Closed.
    - 📈 **Dashboard Insights**: Summarizes operational trends and leverages GenAI for data-driven recommendations.
    - 💬 **Supervisor Chat**: Allows open-ended Q&A with GenAI. Use it to get strategic advice, summaries, or troubleshoot issues. *Note: GPT-3.5 is used for faster and cost-effective answers.*
    - 📉 **Ineffective Technician Tracking**: Identifies technicians with frequent unresolved faults or pending tasks.
    - 🏢 **Executive Management Tab**: Provides KPIs, zone breakdowns, and AI-recommended actions.

    Use the tabs above to explore each function.
    """)

# ---------- Live Faults ----------
with tabs[1]:
    st.header("📍 Live Fault Feed")

    if "simulated_faults" not in st.session_state or not st.session_state.simulated_faults:
        simulate_new_fault(5)
        st.session_state["last_refresh_time"] = datetime.datetime.now()

    if st.button("Refresh Now"):
        simulate_new_fault(1)
        st.session_state["last_refresh_time"] = datetime.datetime.now()
        st.rerun()

    now = datetime.datetime.now()
    last_refresh = st.session_state.get("last_refresh_time", now)
    elapsed = (now - last_refresh).total_seconds()

    if elapsed > 60:
        simulate_new_fault(1)
        st.session_state["last_refresh_time"] = now
        st.rerun()

    st.markdown(f"**Last refreshed at:** {st.session_state['last_refresh_time'].strftime('%Y-%m-%d %H:%M:%S')}")

    df_live = pd.DataFrame(st.session_state.simulated_faults)
    df_live["timestamp"] = pd.to_datetime(df_live["timestamp"], errors='coerce')
    for col in ["assigned_time", "in_progress_time", "closed_time"]:
        df_live[col] = pd.to_datetime(df_live[col], errors='coerce')

    st.dataframe(df_live.sort_values("timestamp", ascending=False).fillna("-"))

# ---------- Work Orders ----------
with tabs[2]:
    st.header("🛠️ Work Order Generator")
    if "simulated_faults" not in st.session_state or not st.session_state.simulated_faults:
        st.info("No faults to process. Please simulate faults first.")
    else:
        df_sim = pd.DataFrame(st.session_state.simulated_faults)
        df_sim["timestamp"] = pd.to_datetime(df_sim["timestamp"], errors='coerce')
        selected_index = st.selectbox("Select a fault to process", df_sim.index[::-1])
        selected = df_sim.loc[selected_index]
        available = technicians[technicians["status"].str.lower() == "available"].copy()

        if not available.empty:
            fault_code = selected["fault_code"]
            available["skill_match"] = available["skills"].apply(lambda x: 1 if fault_code in x else 0)
            best_tech = available.sort_values(["skill_match", "distance_km"], ascending=[False, True]).iloc[0]
            tech_name = best_tech["name"]

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.simulated_faults[selected_index]["status"] = "Assigned"
            st.session_state.simulated_faults[selected_index]["assigned_time"] = now_str
            st.session_state.simulated_faults[selected_index]["technician"] = tech_name

            st.markdown(f"**🧑‍🔧 Assigned:** {tech_name} ({best_tech['skills']})")
            prompt = f"Why did fault code {selected['fault_code']} occur in equipment {selected['equipment']} located at {selected['location']}?"
            st.info(genai_response(prompt))

            match = fault_history[fault_history["fault_code"] == selected["fault_code"]]
            row = match.iloc[0] if not match.empty else {}

            st.markdown(f"""
**Work Order Summary**

- Fault Code: {selected['fault_code']}
- Equipment: {selected['equipment']}
- Zone: {selected['location']}
- Fix: {row.get('common_fix', 'N/A')}
- Risk: {row.get('risk_notes', 'N/A')}
- Assigned: {tech_name} at {now_str}
""")
        else:
            st.info("No available technicians at the moment.")

# ---------- Dashboard ----------
with tabs[3]:
    st.header("📊 Operational Dashboard & Insights")
    fault_history = convert_to_numeric(fault_history, ["avg_resolution_time"])

    st.subheader("📍 Fault Volume by Zone")
    zone_stats = fault_history["location"].value_counts().reset_index()
    zone_stats.columns = ["Zone", "Faults"]
    st.bar_chart(zone_stats.set_index("Zone"))

    st.subheader("⏱ Avg. Resolution Time by Equipment")
    avg_res_by_eq = fault_history.groupby("equipment")["avg_resolution_time"].mean().reset_index()
    st.bar_chart(avg_res_by_eq.set_index("equipment"))

    st.subheader("📌 Top Fault Types")
    top_faults = fault_history["fault_code"].value_counts().head(5)
    st.table(top_faults.reset_index().rename(columns={"index": "Fault Code", "fault_code": "Occurrences"}))

    st.subheader("🧠 GenAI Insights")
    top_zones = zone_stats.sort_values("Faults", ascending=False).head(3).to_dict(orient="records")
    top_equipment = avg_res_by_eq.sort_values("avg_resolution_time", ascending=False).head(3).to_dict(orient="records")
    top_faults_list = top_faults.head(3).to_dict()

    data_prompt = f"""
You are a field operations strategist reviewing fault management metrics. Based on the following:

Top Fault Zones:
{top_zones}

Equipment with Highest Avg. Resolution Time:
{top_equipment}

Most Common Fault Codes:
{top_faults_list}

Generate:
1. Specific insights (e.g., which zone/equipment is underperforming)
2. Root cause hypotheses
3. Actionable next steps (e.g., staffing, training, spares, alerts)
Avoid generic advice. Refer only to this data.
"""

    st.success(genai_response(data_prompt))

# ---------- Supervisor Chat ----------
with tabs[4]:
    st.header("💬 Ask GenAI (Supervisor Assistant)")
    question = st.text_input("Ask a field operations question:")
    if question:
        with st.spinner("Thinking..."):
            response = genai_response(question, stream=False)
        if response.startswith("❗️"):
            st.error(response)
        elif response.strip() == "":
            st.warning("⚠️ GenAI returned an empty response.")
        else:
            st.success(response)

# ---------- Ineffective Techs ----------
with tabs[5]:
    st.header("📉 Ineffective Technician Tracker")
    if "simulated_faults" not in st.session_state or not st.session_state.simulated_faults:
        st.info("No technician activity recorded yet.")
    else:
        df = pd.DataFrame(st.session_state.simulated_faults)
        df["assigned_time"] = pd.to_datetime(df["assigned_time"], errors='coerce')
        df["closed_time"] = pd.to_datetime(df["closed_time"], errors='coerce')
        df["status"] = df["status"].fillna("New")
        report = calculate_ineffective_techs(df)
        st.dataframe(report)
        st.download_button("📥 Download Fault Assignment Log", df.to_csv(index=False), file_name="fault_log.csv")

# ---------- Staffing Adequacy ----------
with tabs[7]:
    st.header("📊 Staffing Adequacy Analysis")
    if not fault_history.empty:
        zone_faults = fault_history["location"].value_counts().to_dict()
        tech_distribution = technicians["preferred_zone"].value_counts().to_dict() if "preferred_zone" in technicians.columns else {}
        unresolved_by_zone = fault_history[fault_history["closed_time"].isnull()]["location"].value_counts().to_dict()

        staffing_prompt = f"""
You are analyzing field technician adequacy using real data:

📊 Fault Summary:
- Total Faults: {len(fault_history)}
- Unresolved Faults: {fault_history['closed_time'].isnull().sum()}

📍 Faults by Zone:
{zone_faults}

👷‍♂️ Technicians by Preferred Zone:
{tech_distribution}

🛑 Unresolved Faults by Zone:
{unresolved_by_zone}

✅ Assess:
- Which zones have a technician shortfall?
- Where are unresolved faults piling up?
- Recommend zone reassignments or hiring.
- Suggest cross-training if needed.

Focus strictly on the data above. Avoid generic advice.
"""
        st.success(genai_response(staffing_prompt))
    else:
        st.info("Historical fault data is missing or empty.")

# ---------- Predictive Risk Advisory ----------
with tabs[8]:
    st.header("🧠 Predictive Risk Advisory")
    if not fault_history.empty:
        top_faults = fault_history["fault_code"].value_counts().head(5).index.tolist()
        risk_context = fault_history[fault_history["fault_code"].isin(top_faults)]

        risk_by_zone = risk_context["location"].value_counts().to_dict()
        risk_by_equipment = risk_context["equipment"].value_counts().to_dict()
        fix_notes = risk_context[["fault_code", "common_fix", "risk_notes"]].dropna().drop_duplicates().to_dict(orient="records")

        risk_prompt = f"""
You are a predictive risk strategist. Use the following real fault data:

🔥 Top Fault Codes: {top_faults}

🗺️ Faults by Zone:
{risk_by_zone}

⚙️ Faults by Equipment:
{risk_by_equipment}

🛠 Past Fixes and Risk Notes:
{fix_notes}

Generate:
1. Risk predictions specific to fault+zone+equipment patterns
2. Suggested preventive tasks and inspections
3. Insights backed by recurrence and past fixes

Avoid any guesswork or generic output. Use data only.
"""
        st.success(genai_response(risk_prompt))
    else:
        st.info("Historical fault data is missing or empty.")

# ---------- Management ----------
with tabs[6]:
    st.header("📈 Executive Management View")
    if "simulated_faults" not in st.session_state or not st.session_state.simulated_faults:
        st.info("No data available yet.")
    else:
        df = pd.DataFrame(st.session_state.simulated_faults)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df["assigned_time"] = pd.to_datetime(df["assigned_time"], errors='coerce')
        df["closed_time"] = pd.to_datetime(df["closed_time"], errors='coerce')
        df["in_progress_time"] = pd.to_datetime(df.get("in_progress_time"), errors='coerce')

        st.subheader("📊 Fault Summary")
        st.metric("Total Faults", len(df))
        st.metric("Unresolved Faults", df["closed_time"].isnull().sum())

        valid_mask = df["closed_time"].notna() & df["in_progress_time"].notna()
        if valid_mask.any():
            avg_resolution = (df.loc[valid_mask, "closed_time"] - df.loc[valid_mask, "in_progress_time"]).mean()
        else:
            avg_resolution = None
        st.metric("Avg. Resolution Time", str(avg_resolution) if avg_resolution is not None else "-")

        st.subheader("🗺️ Faults by Zone")
        zone_counts = df["location"].value_counts().reset_index()
        zone_counts.columns = ["Zone", "Count"]
        st.bar_chart(zone_counts.set_index("Zone"))

        st.subheader("⚙️ Technician Performance")
        if "technician" in df.columns:
            tech_summary = df["technician"].value_counts().reset_index()
            tech_summary.columns = ["Technician", "Faults Assigned"]
            st.table(tech_summary)

        st.subheader("🧠 GenAI Recommended Actions")
        prompt = "Recommend executive actions based on fault and technician performance data."
        st.markdown(genai_response(prompt))

        st.download_button("📥 Download Executive Summary CSV", df.to_csv(index=False), file_name="management_summary.csv")
