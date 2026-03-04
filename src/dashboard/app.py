"""
Streamlit dashboard for CFPB Consumer Complaint Cross-Linked Analysis.

Seven interactive sections:
1. National Overview
2. Institution Explorer
3. Complaint Analysis
4. Enforcement Timeline
5. Geographic Analysis
6. CRA vs. Complaints
7. Trends & Patterns

Built by Nathan Goldberg | nathanmauricegoldberg@gmail.com | www.linkedin.com/in/nathan-goldberg-62a44522a/
"""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CFPB Complaint Tracker",
    page_icon="\U0001f3e6",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants & Theme
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_DEPLOY_DB = _DATA_DIR / "cfpb_tracker_deploy.db"
DB_PATH = _DEPLOY_DB if _DEPLOY_DB.exists() else _DATA_DIR / "cfpb_tracker.db"

COLOR_PRIMARY = "#0984E3"
COLOR_BG = "#0E1117"
COLOR_SECONDARY_BG = "#1B2A4A"
COLOR_TEXT = "#E2E8F0"
COLOR_ACCENT_1 = "#00CEC9"
COLOR_ACCENT_2 = "#6C5CE7"
COLOR_ACCENT_3 = "#FD79A8"
COLOR_ACCENT_4 = "#FDCB6E"
COLOR_ACCENT_5 = "#55EFC4"
COLOR_DANGER = "#D63031"

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLOR_TEXT),
    margin=dict(l=40, r=40, t=50, b=40),
)

# US state abbreviation to full name mapping for choropleth
STATE_ABBREV = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "AS": "American Samoa", "GU": "Guam", "MP": "Northern Mariana Islands",
    "PR": "Puerto Rico", "VI": "Virgin Islands",
}

# Full state names (e.g., "Connecticut") to abbreviation mapping
STATE_NAME_TO_ABBREV = {}
for abbr, full in STATE_ABBREV.items():
    STATE_NAME_TO_ABBREV[full] = abbr
    STATE_NAME_TO_ABBREV[full.upper()] = abbr
    STATE_NAME_TO_ABBREV[full.lower()] = abbr
    STATE_NAME_TO_ABBREV[abbr] = abbr
    STATE_NAME_TO_ABBREV[abbr.lower()] = abbr


def _normalize_state(s: str) -> str:
    """Return 2-letter state abbreviation from either abbrev or full name."""
    if not s:
        return s
    s_stripped = s.strip()
    return STATE_NAME_TO_ABBREV.get(s_stripped, STATE_NAME_TO_ABBREV.get(s_stripped.title(), s_stripped))


# ---------------------------------------------------------------------------
# Database Connection Helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Return a read-only SQLite connection with Row factory."""
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        # DB file doesn't exist yet — use an in-memory stub
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        return conn


def _table_exists(conn, name):
    try:
        r = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return r is not None
    except Exception:
        return False


def _safe_query(sql, conn, params=()):
    try:
        import pandas as _pd
        return _pd.read_sql_query(sql, conn, params=params if params else None)
    except Exception:
        import pandas as _pd
        return _pd.DataFrame()


def _safe_fetchone(conn, sql, params=(), default=0):
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else default
    except Exception:
        return default


@st.cache_data(ttl=300, show_spinner=False)
def run_query(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a SQL query and return results as a list of dicts."""
    conn = _get_conn()
    try:
        cursor = conn.execute(sql, params)
        cols = [d[0] for d in cursor.description] if cursor.description else []
        return [dict(zip(cols, row)) for row in cursor.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def run_scalar(sql: str, params: tuple = ()):
    """Execute a SQL query and return a single scalar value."""
    conn = _get_conn()
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None
    except Exception:
        return None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def fmt_number(n, decimals: int = 0) -> str:
    """Format a number with comma separators."""
    if n is None:
        return "N/A"
    if decimals > 0:
        return f"{n:,.{decimals}f}"
    return f"{int(n):,}"


def fmt_currency(n) -> str:
    """Format a number as USD currency."""
    if n is None or n == 0:
        return "$0"
    if abs(n) >= 1_000_000_000:
        return f"${n / 1_000_000_000:,.1f}B"
    if abs(n) >= 1_000_000:
        return f"${n / 1_000_000:,.1f}M"
    if abs(n) >= 1_000:
        return f"${n / 1_000:,.1f}K"
    return f"${n:,.0f}"


def render_footer():
    """Render the standard footer at the bottom of every page."""
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; padding: 10px 0;'>"
        "Built by Nathan Goldberg | "
        "<a href='mailto:nathanmauricegoldberg@gmail.com' style='color: #0984E3;'>"
        "nathanmauricegoldberg@gmail.com</a> | "
        "<a href='https://www.linkedin.com/in/nathan-goldberg-62a44522a/' style='color: #0984E3;'>"
        "LinkedIn</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

st.sidebar.title("CFPB Complaint Tracker")
st.sidebar.markdown("Cross-linked consumer complaint analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "National Overview",
        "Institution Explorer",
        "Complaint Analysis",
        "Enforcement Timeline",
        "Geographic Analysis",
        "CRA vs. Complaints",
        "Trends & Patterns",
        "Financial Health",
        "Multi-Agency Enforcement",
        "Institution Risk Profiles",
        "Fair Lending Analysis",
    ],
)

# Sidebar data-freshness indicator
total_complaints = run_scalar("SELECT COUNT(*) FROM cfpb_complaints") or 0
total_institutions = run_scalar("SELECT COUNT(*) FROM financial_institutions") or 0
total_enforcement = run_scalar("SELECT COUNT(*) FROM enforcement_actions") or 0
total_financials = run_scalar("SELECT COUNT(DISTINCT cert_number) FROM institution_financials") or 0
total_risk_profiles = run_scalar("SELECT COUNT(*) FROM institution_risk_profiles") or 0

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Summary**")
st.sidebar.markdown(f"Complaints: **{fmt_number(total_complaints)}**")
st.sidebar.markdown(f"Institutions: **{fmt_number(total_institutions)}**")
st.sidebar.markdown(f"Enforcement: **{fmt_number(total_enforcement)}**")
if total_financials > 0:
    st.sidebar.markdown(f"w/ Financials: **{fmt_number(total_financials)}**")
if total_risk_profiles > 0:
    st.sidebar.markdown(f"Risk Profiles: **{fmt_number(total_risk_profiles)}**")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built by Nathan Goldberg**  \n"
    "nathanmauricegoldberg@gmail.com  \n"
    "[LinkedIn](https://www.linkedin.com/in/nathan-goldberg-62a44522a/)"
)


# ============================================================================
# SECTION 1: NATIONAL OVERVIEW
# ============================================================================

if page == "National Overview":
    st.title("National Overview")
    st.markdown(
        "High-level metrics across CFPB complaints, FDIC institutions, "
        "and enforcement actions."
    )

    _conn = _get_conn()
    if not _table_exists(_conn, "cfpb_complaints"):
        st.warning("No data loaded. Run the pipeline first: `python -m src.cli pipeline --full`")
        render_footer()
        _conn.close()
        st.stop()
    _conn.close()

    # --- KPI Cards ---
    total_cross_links = run_scalar("SELECT COUNT(*) FROM institution_complaints") or 0
    total_cra = run_scalar("SELECT COUNT(*) FROM cra_ratings") or 0
    total_penalties = run_scalar(
        "SELECT SUM(penalty_amount) FROM enforcement_actions WHERE penalty_amount > 0"
    ) or 0
    avg_quality = run_scalar(
        "SELECT AVG(quality_score) FROM financial_institutions WHERE quality_score > 0"
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Complaints", fmt_number(total_complaints))
    k2.metric("FDIC Institutions", fmt_number(total_institutions))
    k3.metric("Enforcement Actions", fmt_number(total_enforcement))
    k4.metric("Total Penalties", fmt_currency(total_penalties))

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Cross-Linked Records", fmt_number(total_cross_links))
    k6.metric("CRA Ratings", fmt_number(total_cra))
    k7.metric(
        "Avg Quality Score",
        f"{avg_quality:.2f}" if avg_quality else "N/A",
    )
    date_range = run_query(
        "SELECT MIN(date_received) as min_d, MAX(date_received) as max_d "
        "FROM cfpb_complaints"
    )
    if date_range and date_range[0]["min_d"]:
        k8.metric(
            "Date Range",
            f"{date_range[0]['min_d'][:7]} to {date_range[0]['max_d'][:7]}",
        )
    else:
        k8.metric("Date Range", "N/A")

    st.markdown("---")

    # --- Complaint Volume Over Time ---
    st.subheader("Complaint Volume Over Time")

    time_gran = st.radio(
        "Granularity",
        ["Yearly", "Monthly"],
        horizontal=True,
        key="overview_time_gran",
    )

    if time_gran == "Yearly":
        time_data = run_query(
            "SELECT substr(date_received, 1, 4) AS period, COUNT(*) AS cnt "
            "FROM cfpb_complaints "
            "WHERE date_received IS NOT NULL "
            "GROUP BY period ORDER BY period"
        )
    else:
        time_data = run_query(
            "SELECT substr(date_received, 1, 7) AS period, COUNT(*) AS cnt "
            "FROM cfpb_complaints "
            "WHERE date_received IS NOT NULL "
            "GROUP BY period ORDER BY period"
        )

    if time_data:
        df_time = pd.DataFrame(time_data)
        fig_time = px.bar(
            df_time,
            x="period",
            y="cnt",
            labels={"period": "Period", "cnt": "Complaints"},
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        fig_time.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Complaints by {time_gran[:-2] if time_gran.endswith('ly') else time_gran}",
            xaxis_title="",
            yaxis_title="Number of Complaints",
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No complaint data available yet.")

    st.markdown("---")

    # --- Top 10 Companies ---
    st.subheader("Top 10 Most-Complained-About Companies")

    top_companies = run_query(
        "SELECT company, COUNT(*) AS cnt FROM cfpb_complaints "
        "GROUP BY company ORDER BY cnt DESC LIMIT 10"
    )

    if top_companies:
        df_companies = pd.DataFrame(top_companies)
        fig_companies = px.bar(
            df_companies,
            x="cnt",
            y="company",
            orientation="h",
            labels={"cnt": "Complaints", "company": ""},
            color_discrete_sequence=[COLOR_ACCENT_1],
        )
        fig_companies.update_layout(
            **PLOTLY_LAYOUT,
            title="Top 10 Companies by Complaint Volume",
            yaxis=dict(autorange="reversed"),
            height=450,
        )
        st.plotly_chart(fig_companies, use_container_width=True)

        # Also show as a table
        with st.expander("View as table"):
            df_display = df_companies.rename(
                columns={"company": "Company", "cnt": "Complaints"}
            )
            df_display.index = range(1, len(df_display) + 1)
            st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No complaint data available yet.")

    render_footer()


# ============================================================================
# SECTION 2: INSTITUTION EXPLORER
# ============================================================================

elif page == "Institution Explorer":
    st.title("Institution Explorer")
    st.markdown(
        "Search and explore FDIC-insured financial institutions with linked "
        "complaint data, CRA ratings, and enforcement history."
    )

    _conn = _get_conn()
    if not _table_exists(_conn, "financial_institutions"):
        st.warning("No institution data loaded. Run the pipeline first: `python -m src.cli pipeline --full`")
        render_footer()
        _conn.close()
        st.stop()
    _conn.close()

    # --- Filters ---
    col_search, col_state, col_active = st.columns([3, 1, 1])

    with col_search:
        search_name = st.text_input(
            "Search by institution name",
            placeholder="e.g., Wells Fargo, JPMorgan Chase",
            key="inst_search",
        )

    # Get distinct states for institutions
    inst_states = run_query(
        "SELECT DISTINCT state FROM financial_institutions "
        "WHERE state IS NOT NULL AND state <> '' ORDER BY state"
    )
    state_options = ["All States"] + [r["state"] for r in inst_states]

    with col_state:
        selected_state = st.selectbox(
            "State", state_options, key="inst_state_filter"
        )

    with col_active:
        active_filter = st.selectbox(
            "Status", ["All", "Active", "Inactive"], key="inst_active_filter"
        )

    # Asset size range
    col_min_asset, col_max_asset = st.columns(2)
    with col_min_asset:
        min_assets = st.number_input(
            "Min Total Assets ($K)", min_value=0, value=0, step=10000,
            key="inst_min_assets",
        )
    with col_max_asset:
        max_assets = st.number_input(
            "Max Total Assets ($K)", min_value=0, value=0, step=10000,
            help="Set to 0 for no upper limit",
            key="inst_max_assets",
        )

    # Build query
    where_clauses = []
    params: list = []

    if search_name:
        where_clauses.append("institution_name LIKE ?")
        params.append(f"%{search_name}%")
    if selected_state != "All States":
        where_clauses.append("state = ?")
        params.append(selected_state)
    if active_filter == "Active":
        where_clauses.append("active = 1")
    elif active_filter == "Inactive":
        where_clauses.append("active = 0")
    if min_assets > 0:
        where_clauses.append("total_assets >= ?")
        params.append(min_assets * 1000)
    if max_assets > 0:
        where_clauses.append("total_assets <= ?")
        params.append(max_assets * 1000)

    where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    institutions = run_query(
        f"SELECT cert_number, institution_name, city, state, total_assets, "
        f"regulator, active, quality_score "
        f"FROM financial_institutions{where_sql} "
        f"ORDER BY COALESCE(total_assets, 0) DESC LIMIT 200",
        tuple(params),
    )

    st.markdown(f"**{len(institutions)}** institutions found (showing up to 200)")

    if institutions:
        df_inst = pd.DataFrame(institutions)
        df_inst["total_assets_display"] = df_inst["total_assets"].apply(
            lambda x: fmt_currency(x) if x and x > 0 else "N/A"
        )
        df_inst["status"] = df_inst["active"].apply(
            lambda x: "Active" if x == 1 else "Inactive"
        )

        display_cols = {
            "cert_number": "CERT #",
            "institution_name": "Institution",
            "city": "City",
            "state": "State",
            "total_assets_display": "Total Assets",
            "status": "Status",
            "quality_score": "Quality Score",
        }
        df_show = df_inst[list(display_cols.keys())].rename(columns=display_cols)
        df_show.index = range(1, len(df_show) + 1)

        st.dataframe(df_show, use_container_width=True, height=400)

        # --- Institution Detail Drill-Down ---
        st.markdown("---")
        st.subheader("Institution Detail")

        cert_options = {
            f"{r['institution_name']} (CERT: {r['cert_number']})": r["cert_number"]
            for r in institutions
        }

        if cert_options:
            selected_label = st.selectbox(
                "Select an institution for details",
                list(cert_options.keys()),
                key="inst_detail_select",
            )
            selected_cert = cert_options[selected_label]

            detail = run_query(
                "SELECT * FROM financial_institutions WHERE cert_number = ?",
                (selected_cert,),
            )

            if detail:
                inst = detail[0]
                d1, d2, d3 = st.columns(3)
                d1.metric("CERT Number", inst["cert_number"])
                d2.metric("Total Assets", fmt_currency(inst["total_assets"]) if inst["total_assets"] else "N/A")
                d3.metric("Quality Score", f"{inst['quality_score']:.2f}" if inst["quality_score"] else "N/A")

                d4, d5, d6 = st.columns(3)
                d4.metric("Location", f"{inst['city'] or 'N/A'}, {inst['state'] or 'N/A'}")
                d5.metric("Regulator", inst["regulator"] or "N/A")
                d6.metric("Status", "Active" if inst["active"] == 1 else "Inactive")

                # Linked complaints count
                linked_count = run_scalar(
                    "SELECT COUNT(*) FROM institution_complaints WHERE cert_number = ?",
                    (selected_cert,),
                ) or 0
                st.markdown(f"**Linked Complaints:** {fmt_number(linked_count)}")

                if linked_count > 0:
                    linked_complaints = run_query(
                        "SELECT c.complaint_id, c.date_received, c.product, c.issue, "
                        "c.company_response, ic.match_confidence, ic.match_method "
                        "FROM cfpb_complaints c "
                        "JOIN institution_complaints ic ON c.complaint_id = ic.complaint_id "
                        "WHERE ic.cert_number = ? "
                        "ORDER BY c.date_received DESC LIMIT 50",
                        (selected_cert,),
                    )
                    if linked_complaints:
                        df_linked = pd.DataFrame(linked_complaints)
                        df_linked["match_confidence"] = df_linked["match_confidence"].apply(
                            lambda x: f"{x:.2f}" if x else "N/A"
                        )
                        st.dataframe(df_linked, use_container_width=True, height=300)

                # CRA Ratings for this institution
                cra_data = run_query(
                    "SELECT cra_rating, exam_date, regulator "
                    "FROM cra_ratings WHERE cert_number = ? ORDER BY exam_date DESC",
                    (selected_cert,),
                )
                if cra_data:
                    st.markdown("**CRA Rating History:**")
                    st.dataframe(pd.DataFrame(cra_data), use_container_width=True)
                else:
                    st.caption("No CRA ratings on file for this institution.")

                # Enforcement actions
                enforcement_data = run_query(
                    "SELECT respondent_name, action_date, action_type, penalty_amount, description "
                    "FROM enforcement_actions WHERE institution_cert = ? "
                    "ORDER BY action_date DESC",
                    (selected_cert,),
                )
                if enforcement_data:
                    st.markdown("**Enforcement Actions:**")
                    st.dataframe(pd.DataFrame(enforcement_data), use_container_width=True)
                else:
                    st.caption("No enforcement actions linked to this institution.")
    else:
        st.info("No institutions match the current filters.")

    render_footer()


# ============================================================================
# SECTION 3: COMPLAINT ANALYSIS
# ============================================================================

elif page == "Complaint Analysis":
    st.title("Complaint Analysis")
    st.markdown(
        "Detailed breakdown of consumer complaints by product, issue, "
        "company response, and geographic distribution."
    )

    _conn = _get_conn()
    if not _table_exists(_conn, "cfpb_complaints"):
        st.warning("No complaint data loaded. Run the pipeline first: `python -m src.cli pipeline --full`")
        render_footer()
        _conn.close()
        st.stop()
    _conn.close()

    # --- Filters ---
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        complaint_states = run_query(
            "SELECT DISTINCT state FROM cfpb_complaints "
            "WHERE state IS NOT NULL AND state <> '' ORDER BY state"
        )
        complaint_state_options = ["All States"] + [r["state"] for r in complaint_states]
        ca_state = st.selectbox("State", complaint_state_options, key="ca_state_filter")

    with filter_col2:
        products = run_query(
            "SELECT DISTINCT product FROM cfpb_complaints "
            "WHERE product IS NOT NULL ORDER BY product"
        )
        product_options = ["All Products"] + [r["product"] for r in products]
        ca_product = st.selectbox("Product", product_options, key="ca_product_filter")

    # Build WHERE clause for filters
    ca_wheres = []
    ca_params: list = []
    if ca_state != "All States":
        ca_wheres.append("state = ?")
        ca_params.append(ca_state)
    if ca_product != "All Products":
        ca_wheres.append("product = ?")
        ca_params.append(ca_product)

    ca_where_sql = (" WHERE " + " AND ".join(ca_wheres)) if ca_wheres else ""
    ca_and_sql = (" AND " + " AND ".join(ca_wheres)) if ca_wheres else ""

    # Filtered complaint count
    filtered_total = run_scalar(
        f"SELECT COUNT(*) FROM cfpb_complaints{ca_where_sql}",
        tuple(ca_params),
    ) or 0
    st.markdown(f"**{fmt_number(filtered_total)}** complaints matching filters")

    st.markdown("---")

    # --- Product Breakdown ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Product Breakdown")
        product_data = run_query(
            f"SELECT product, COUNT(*) AS cnt FROM cfpb_complaints"
            f"{ca_where_sql} GROUP BY product ORDER BY cnt DESC LIMIT 12",
            tuple(ca_params),
        )

        if product_data:
            df_prod = pd.DataFrame(product_data)
            fig_prod = px.pie(
                df_prod,
                values="cnt",
                names="product",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_prod.update_layout(
                **PLOTLY_LAYOUT,
                title="Complaints by Product",
                showlegend=True,
                legend=dict(font=dict(size=10)),
                height=450,
            )
            fig_prod.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_prod, use_container_width=True)
        else:
            st.info("No product data available.")

    with col_right:
        st.subheader("Top Issues")
        issue_data = run_query(
            f"SELECT issue, COUNT(*) AS cnt FROM cfpb_complaints "
            f"WHERE issue IS NOT NULL"
            f"{ca_and_sql} GROUP BY issue ORDER BY cnt DESC LIMIT 12",
            tuple(ca_params),
        )

        if issue_data:
            df_issue = pd.DataFrame(issue_data)
            fig_issue = px.bar(
                df_issue,
                x="cnt",
                y="issue",
                orientation="h",
                labels={"cnt": "Complaints", "issue": ""},
                color_discrete_sequence=[COLOR_ACCENT_2],
            )
            fig_issue.update_layout(
                **PLOTLY_LAYOUT,
                title="Top Issues by Volume",
                yaxis=dict(autorange="reversed"),
                height=450,
            )
            st.plotly_chart(fig_issue, use_container_width=True)
        else:
            st.info("No issue data available.")

    st.markdown("---")

    # --- Company Response Analysis ---
    st.subheader("Company Response Analysis")

    resp_col1, resp_col2 = st.columns(2)

    with resp_col1:
        response_data = run_query(
            f"SELECT company_response, COUNT(*) AS cnt FROM cfpb_complaints "
            f"WHERE company_response IS NOT NULL"
            f"{ca_and_sql} GROUP BY company_response ORDER BY cnt DESC",
            tuple(ca_params),
        )

        if response_data:
            df_resp = pd.DataFrame(response_data)
            fig_resp = px.bar(
                df_resp,
                x="company_response",
                y="cnt",
                labels={"company_response": "Response", "cnt": "Count"},
                color_discrete_sequence=[COLOR_ACCENT_4],
            )
            fig_resp.update_layout(
                **PLOTLY_LAYOUT,
                title="Company Response Distribution",
                xaxis_tickangle=-30,
                height=400,
            )
            st.plotly_chart(fig_resp, use_container_width=True)
        else:
            st.info("No response data available.")

    with resp_col2:
        timely_data = run_query(
            f"SELECT timely_response, COUNT(*) AS cnt FROM cfpb_complaints "
            f"WHERE timely_response IS NOT NULL"
            f"{ca_and_sql} GROUP BY timely_response ORDER BY cnt DESC",
            tuple(ca_params),
        )

        if timely_data:
            df_timely = pd.DataFrame(timely_data)
            fig_timely = px.pie(
                df_timely,
                values="cnt",
                names="timely_response",
                color_discrete_sequence=[COLOR_ACCENT_5, COLOR_DANGER],
            )
            fig_timely.update_layout(
                **PLOTLY_LAYOUT,
                title="Timely Response Rate",
                height=400,
            )
            st.plotly_chart(fig_timely, use_container_width=True)
        else:
            st.info("No timeliness data available.")

    st.markdown("---")

    # --- Submission Channel ---
    st.subheader("Submission Channels")
    channel_data = run_query(
        f"SELECT submitted_via, COUNT(*) AS cnt FROM cfpb_complaints "
        f"WHERE submitted_via IS NOT NULL"
        f"{ca_and_sql} GROUP BY submitted_via ORDER BY cnt DESC",
        tuple(ca_params),
    )

    if channel_data:
        df_channel = pd.DataFrame(channel_data)
        fig_channel = px.bar(
            df_channel,
            x="submitted_via",
            y="cnt",
            labels={"submitted_via": "Channel", "cnt": "Complaints"},
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        fig_channel.update_layout(
            **PLOTLY_LAYOUT,
            title="How Complaints Are Submitted",
            height=350,
        )
        st.plotly_chart(fig_channel, use_container_width=True)

    render_footer()


# ============================================================================
# SECTION 4: ENFORCEMENT TIMELINE
# ============================================================================

elif page == "Enforcement Timeline":
    st.title("Enforcement Timeline")
    st.markdown(
        "Chronological view of CFPB enforcement actions, penalty amounts, "
        "and action types."
    )

    _conn = _get_conn()
    if not _table_exists(_conn, "enforcement_actions"):
        st.warning("No enforcement data loaded. Run the pipeline first: `python -m src.cli pipeline --full`")
        render_footer()
        _conn.close()
        st.stop()
    _conn.close()

    if total_enforcement == 0:
        st.warning("No enforcement actions have been loaded yet.")
        render_footer()
        st.stop()

    # --- KPI Cards ---
    penalties_total = run_scalar(
        "SELECT SUM(penalty_amount) FROM enforcement_actions WHERE penalty_amount > 0"
    ) or 0
    action_type_count = run_scalar(
        "SELECT COUNT(DISTINCT action_type) FROM enforcement_actions"
    ) or 0
    enforcement_date_range = run_query(
        "SELECT MIN(action_date) AS min_d, MAX(action_date) AS max_d "
        "FROM enforcement_actions WHERE action_date IS NOT NULL"
    )

    ek1, ek2, ek3, ek4 = st.columns(4)
    ek1.metric("Total Actions", fmt_number(total_enforcement))
    ek2.metric("Total Penalties", fmt_currency(penalties_total))
    ek3.metric("Action Types", fmt_number(action_type_count))
    if enforcement_date_range and enforcement_date_range[0]["min_d"]:
        ek4.metric(
            "Date Range",
            f"{enforcement_date_range[0]['min_d'][:10]} to {enforcement_date_range[0]['max_d'][:10]}",
        )
    else:
        ek4.metric("Date Range", "N/A")

    st.markdown("---")

    # --- Filters ---
    enf_col1, enf_col2 = st.columns(2)

    with enf_col1:
        action_types = run_query(
            "SELECT DISTINCT action_type FROM enforcement_actions "
            "WHERE action_type IS NOT NULL ORDER BY action_type"
        )
        type_options = ["All Types"] + [r["action_type"] for r in action_types]
        enf_type = st.selectbox("Action Type", type_options, key="enf_type_filter")

    with enf_col2:
        enf_search = st.text_input(
            "Search respondent name",
            placeholder="e.g., Wells Fargo",
            key="enf_search",
        )

    enf_wheres = []
    enf_params: list = []
    if enf_type != "All Types":
        enf_wheres.append("action_type = ?")
        enf_params.append(enf_type)
    if enf_search:
        enf_wheres.append("respondent_name LIKE ?")
        enf_params.append(f"%{enf_search}%")

    enf_where = (" WHERE " + " AND ".join(enf_wheres)) if enf_wheres else ""

    # --- Actions by Type ---
    st.subheader("Actions by Type")
    type_data = run_query(
        "SELECT action_type, COUNT(*) AS cnt FROM enforcement_actions "
        "WHERE action_type IS NOT NULL AND action_type <> '' "
        "GROUP BY action_type ORDER BY cnt DESC"
    )

    if type_data:
        df_type = pd.DataFrame(type_data)
        fig_type = px.bar(
            df_type,
            x="action_type",
            y="cnt",
            labels={"action_type": "Action Type", "cnt": "Count"},
            color_discrete_sequence=[COLOR_DANGER],
        )
        fig_type.update_layout(
            **PLOTLY_LAYOUT,
            title="Enforcement Actions by Type",
            height=350,
        )
        st.plotly_chart(fig_type, use_container_width=True)

    st.markdown("---")

    # --- Timeline ---
    st.subheader("Enforcement Actions Over Time")
    timeline_data = run_query(
        "SELECT substr(action_date, 1, 4) AS year, COUNT(*) AS cnt "
        "FROM enforcement_actions "
        "WHERE action_date IS NOT NULL "
        "GROUP BY year ORDER BY year"
    )

    if timeline_data:
        df_timeline = pd.DataFrame(timeline_data)
        fig_timeline = px.bar(
            df_timeline,
            x="year",
            y="cnt",
            labels={"year": "Year", "cnt": "Actions"},
            color_discrete_sequence=[COLOR_ACCENT_3],
        )
        fig_timeline.update_layout(
            **PLOTLY_LAYOUT,
            title="Enforcement Actions by Year",
            height=350,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("---")

    # --- Full Actions Table ---
    st.subheader("Enforcement Actions Detail")

    actions = run_query(
        f"SELECT respondent_name, action_date, action_type, "
        f"penalty_amount, description, url "
        f"FROM enforcement_actions{enf_where} "
        f"ORDER BY action_date DESC LIMIT 100",
        tuple(enf_params),
    )

    if actions:
        df_actions = pd.DataFrame(actions)
        df_actions["penalty_display"] = df_actions["penalty_amount"].apply(
            lambda x: fmt_currency(x) if x and x > 0 else "N/A"
        )

        display_df = df_actions[
            ["respondent_name", "action_date", "action_type", "penalty_display", "description"]
        ].rename(columns={
            "respondent_name": "Respondent",
            "action_date": "Date",
            "action_type": "Type",
            "penalty_display": "Penalty",
            "description": "Description",
        })
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(display_df, use_container_width=True, height=500)
    else:
        st.info("No enforcement actions match the current filters.")

    render_footer()


# ============================================================================
# SECTION 5: GEOGRAPHIC ANALYSIS
# ============================================================================

elif page == "Geographic Analysis":
    st.title("Geographic Analysis")
    st.markdown(
        "State-level complaint density and geographic distribution of "
        "consumer complaints."
    )

    _conn = _get_conn()
    if not _table_exists(_conn, "cfpb_complaints"):
        st.warning("No complaint data loaded. Run the pipeline first: `python -m src.cli pipeline --full`")
        render_footer()
        _conn.close()
        st.stop()
    _conn.close()

    # --- State-level complaint counts ---
    state_data = run_query(
        "SELECT state, COUNT(*) AS cnt FROM cfpb_complaints "
        "WHERE state IS NOT NULL AND state <> '' "
        "GROUP BY state ORDER BY cnt DESC"
    )

    if not state_data:
        st.warning("No geographic complaint data available.")
        render_footer()
        st.stop()

    df_geo = pd.DataFrame(state_data)

    # Normalize state values to 2-letter abbreviations
    df_geo["state_code"] = df_geo["state"].apply(_normalize_state)
    df_geo["state_name"] = df_geo["state_code"].map(
        lambda x: STATE_ABBREV.get(x, x)
    )

    # Filter to only valid US state codes for the choropleth
    valid_codes = set(STATE_ABBREV.keys())
    df_geo_valid = df_geo[df_geo["state_code"].isin(valid_codes)].copy()

    if df_geo_valid.empty:
        st.warning("No valid US state data available for geographic analysis.")
        render_footer()
        st.stop()

    # --- Choropleth Map ---
    st.subheader("Complaint Density by State")

    fig_map = px.choropleth(
        df_geo_valid,
        locations="state_code",
        locationmode="USA-states",
        color="cnt",
        scope="usa",
        color_continuous_scale="Blues",
        labels={"cnt": "Complaints", "state_code": "State"},
        hover_data={"state_name": True, "cnt": ":,"},
    )
    fig_map.update_layout(
        **PLOTLY_LAYOUT,
        title="Consumer Complaints by State",
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            lakecolor="rgba(0,0,0,0)",
            landcolor=COLOR_SECONDARY_BG,
            showlakes=True,
        ),
        height=550,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    # --- Top / Bottom States ---
    col_top, col_bottom = st.columns(2)

    with col_top:
        st.subheader("Top 15 States by Complaint Volume")
        df_top = df_geo_valid.head(15).copy()
        fig_top = px.bar(
            df_top,
            x="cnt",
            y="state_name",
            orientation="h",
            labels={"cnt": "Complaints", "state_name": ""},
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        fig_top.update_layout(
            **PLOTLY_LAYOUT,
            yaxis=dict(autorange="reversed"),
            height=500,
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col_bottom:
        st.subheader("Bottom 15 States by Complaint Volume")
        df_bottom = df_geo_valid.tail(15).sort_values("cnt", ascending=True).copy()
        fig_bottom = px.bar(
            df_bottom,
            x="cnt",
            y="state_name",
            orientation="h",
            labels={"cnt": "Complaints", "state_name": ""},
            color_discrete_sequence=[COLOR_ACCENT_4],
        )
        fig_bottom.update_layout(
            **PLOTLY_LAYOUT,
            height=500,
        )
        st.plotly_chart(fig_bottom, use_container_width=True)

    st.markdown("---")

    # --- State Comparison Table ---
    st.subheader("Full State Comparison")

    df_table = df_geo_valid[["state_code", "state_name", "cnt"]].copy()
    df_table = df_table.rename(columns={
        "state_code": "Abbreviation",
        "state_name": "State",
        "cnt": "Total Complaints",
    })
    df_table["% of Total"] = (
        df_table["Total Complaints"] / df_table["Total Complaints"].sum() * 100
    ).round(2)
    df_table.index = range(1, len(df_table) + 1)

    st.dataframe(df_table, use_container_width=True, height=400)

    render_footer()


# ============================================================================
# SECTION 6: CRA vs. COMPLAINTS
# ============================================================================

elif page == "CRA vs. Complaints":
    st.title("CRA Ratings vs. Complaint Volume")
    st.markdown(
        "Analysis of the relationship between Community Reinvestment Act (CRA) "
        "ratings and consumer complaint patterns at the institution level."
    )

    _conn = _get_conn()
    if not _table_exists(_conn, "cra_ratings") or not _table_exists(_conn, "financial_institutions"):
        st.warning("No CRA or institution data loaded. Run the pipeline first: `python -m src.cli pipeline --full`")
        render_footer()
        _conn.close()
        st.stop()
    _conn.close()

    # Check if we have CRA data and cross-links
    cra_count = run_scalar("SELECT COUNT(*) FROM cra_ratings") or 0
    cross_link_count = run_scalar("SELECT COUNT(*) FROM institution_complaints") or 0

    if cra_count == 0:
        st.warning(
            "No CRA rating data has been loaded yet. "
            "Run the CRA download pipeline to populate this section."
        )
        st.markdown("---")
        st.markdown(
            "**What this section will show once data is available:**\n"
            "- Scatter plot of institutions by complaint count vs. CRA rating\n"
            "- Identification of outliers (high complaints + Outstanding CRA rating)\n"
            "- Distribution of complaints across CRA rating tiers\n"
            "- Asset size tier filtering"
        )
        render_footer()
        st.stop()

    # --- CRA Rating Distribution ---
    st.subheader("CRA Rating Distribution")
    cra_dist = run_query(
        "SELECT cra_rating, COUNT(DISTINCT cert_number) AS cnt "
        "FROM cra_ratings WHERE cra_rating IS NOT NULL "
        "GROUP BY cra_rating ORDER BY cnt DESC"
    )

    if cra_dist:
        df_cra_dist = pd.DataFrame(cra_dist)

        # Define a rating order for consistent display
        rating_order = ["Outstanding", "Satisfactory", "Needs to Improve", "Substantial Noncompliance"]
        df_cra_dist["sort_order"] = df_cra_dist["cra_rating"].apply(
            lambda x: rating_order.index(x) if x in rating_order else 99
        )
        df_cra_dist = df_cra_dist.sort_values("sort_order")

        fig_cra = px.bar(
            df_cra_dist,
            x="cra_rating",
            y="cnt",
            labels={"cra_rating": "CRA Rating", "cnt": "Institutions"},
            color="cra_rating",
            color_discrete_map={
                "Outstanding": COLOR_ACCENT_5,
                "Satisfactory": COLOR_PRIMARY,
                "Needs to Improve": COLOR_ACCENT_4,
                "Substantial Noncompliance": COLOR_DANGER,
            },
        )
        fig_cra.update_layout(
            **PLOTLY_LAYOUT,
            title="Institutions by CRA Rating",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig_cra, use_container_width=True)

    st.markdown("---")

    # --- Scatter: CRA Rating vs Complaint Count ---
    if cross_link_count > 0:
        st.subheader("CRA Rating vs. Complaint Volume (per Institution)")

        # Get most recent CRA rating per institution
        # Join with complaint counts from institution_complaints
        scatter_data = run_query(
            """
            SELECT
                fi.cert_number,
                fi.institution_name,
                fi.total_assets,
                cr.cra_rating,
                COALESCE(ic_counts.complaint_count, 0) AS complaint_count
            FROM financial_institutions fi
            JOIN (
                SELECT cert_number, cra_rating
                FROM cra_ratings
                WHERE id IN (
                    SELECT MAX(id) FROM cra_ratings GROUP BY cert_number
                )
            ) cr ON fi.cert_number = cr.cert_number
            LEFT JOIN (
                SELECT cert_number, COUNT(*) AS complaint_count
                FROM institution_complaints
                GROUP BY cert_number
            ) ic_counts ON fi.cert_number = ic_counts.cert_number
            WHERE cr.cra_rating IS NOT NULL
            ORDER BY complaint_count DESC
            """
        )

        if scatter_data:
            df_scatter = pd.DataFrame(scatter_data)

            # Assign numeric values to CRA ratings for y-axis
            rating_numeric = {
                "Outstanding": 4,
                "Satisfactory": 3,
                "Needs to Improve": 2,
                "Substantial Noncompliance": 1,
            }
            df_scatter["rating_num"] = df_scatter["cra_rating"].map(rating_numeric)
            df_scatter = df_scatter.dropna(subset=["rating_num"])

            # Asset size tier
            def asset_tier(assets):
                if assets is None or assets <= 0:
                    return "Unknown"
                if assets < 500_000:
                    return "< $500M"
                if assets < 5_000_000:
                    return "$500M - $5B"
                if assets < 50_000_000:
                    return "$5B - $50B"
                return "> $50B"

            df_scatter["asset_tier"] = df_scatter["total_assets"].apply(asset_tier)

            fig_scatter = px.scatter(
                df_scatter,
                x="complaint_count",
                y="rating_num",
                color="cra_rating",
                size="total_assets",
                hover_name="institution_name",
                hover_data={
                    "complaint_count": ":,",
                    "cra_rating": True,
                    "total_assets": ":,.0f",
                    "rating_num": False,
                },
                color_discrete_map={
                    "Outstanding": COLOR_ACCENT_5,
                    "Satisfactory": COLOR_PRIMARY,
                    "Needs to Improve": COLOR_ACCENT_4,
                    "Substantial Noncompliance": COLOR_DANGER,
                },
                labels={
                    "complaint_count": "Complaint Count",
                    "rating_num": "CRA Rating",
                    "cra_rating": "Rating",
                },
            )
            fig_scatter.update_layout(
                **PLOTLY_LAYOUT,
                title="CRA Rating vs. Linked Complaint Volume",
                yaxis=dict(
                    tickvals=[1, 2, 3, 4],
                    ticktext=["Substantial Noncompliance", "Needs to Improve", "Satisfactory", "Outstanding"],
                ),
                height=550,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # --- Outlier Table ---
            st.subheader("Notable Outliers")
            st.markdown(
                "Institutions with high complaint counts relative to their "
                "CRA rating."
            )

            # Show top complaint institutions with Outstanding ratings
            outliers = df_scatter[
                (df_scatter["cra_rating"] == "Outstanding") &
                (df_scatter["complaint_count"] > 0)
            ].sort_values("complaint_count", ascending=False).head(20)

            if not outliers.empty:
                st.markdown("**Outstanding-Rated Institutions with Most Complaints:**")
                display_outliers = outliers[
                    ["institution_name", "complaint_count", "cra_rating", "asset_tier"]
                ].rename(columns={
                    "institution_name": "Institution",
                    "complaint_count": "Complaints",
                    "cra_rating": "CRA Rating",
                    "asset_tier": "Asset Tier",
                })
                display_outliers.index = range(1, len(display_outliers) + 1)
                st.dataframe(display_outliers, use_container_width=True)
            else:
                st.info("No outlier data found with current filters.")
        else:
            st.info("No cross-linked data available for scatter analysis.")
    else:
        st.info(
            "Cross-linked institution-complaint data is not yet available. "
            "Run the entity resolution pipeline to generate cross-links."
        )

    render_footer()


# ============================================================================
# SECTION 7: TRENDS & PATTERNS
# ============================================================================

elif page == "Trends & Patterns":
    st.title("Trends & Patterns")
    st.markdown(
        "Year-over-year complaint trends, monthly patterns, emerging issues, "
        "and response timeliness analysis."
    )

    _conn = _get_conn()
    if not _table_exists(_conn, "cfpb_complaints"):
        st.warning("No complaint data loaded. Run the pipeline first: `python -m src.cli pipeline --full`")
        render_footer()
        _conn.close()
        st.stop()
    _conn.close()

    # --- Year-over-Year Trends by Product ---
    st.subheader("Year-over-Year Trends by Product")

    # Get top 5 products for trend analysis
    top_products = run_query(
        "SELECT product, COUNT(*) AS cnt FROM cfpb_complaints "
        "WHERE product IS NOT NULL "
        "GROUP BY product ORDER BY cnt DESC LIMIT 5"
    )

    if top_products:
        product_names = [r["product"] for r in top_products]
        placeholders = ", ".join(["?" for _ in product_names])

        yoy_data = run_query(
            f"SELECT substr(date_received, 1, 4) AS year, product, COUNT(*) AS cnt "
            f"FROM cfpb_complaints "
            f"WHERE product IN ({placeholders}) AND date_received IS NOT NULL "
            f"GROUP BY year, product ORDER BY year",
            tuple(product_names),
        )

        if yoy_data:
            df_yoy = pd.DataFrame(yoy_data)
            fig_yoy = px.line(
                df_yoy,
                x="year",
                y="cnt",
                color="product",
                markers=True,
                labels={"year": "Year", "cnt": "Complaints", "product": "Product"},
                color_discrete_sequence=[
                    COLOR_PRIMARY, COLOR_ACCENT_1, COLOR_ACCENT_2,
                    COLOR_ACCENT_3, COLOR_ACCENT_4,
                ],
            )
            fig_yoy.update_layout(
                **PLOTLY_LAYOUT,
                title="Top 5 Products: Annual Complaint Volume",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0),
            )
            st.plotly_chart(fig_yoy, use_container_width=True)

    st.markdown("---")

    # --- Monthly Patterns ---
    st.subheader("Monthly Complaint Patterns")

    monthly_data = run_query(
        "SELECT substr(date_received, 6, 2) AS month, COUNT(*) AS cnt "
        "FROM cfpb_complaints "
        "WHERE date_received IS NOT NULL AND length(date_received) >= 7 "
        "GROUP BY month ORDER BY month"
    )

    if monthly_data:
        df_monthly = pd.DataFrame(monthly_data)
        month_names = {
            "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
            "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
            "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
        }
        df_monthly["month_name"] = df_monthly["month"].map(month_names)
        df_monthly = df_monthly.dropna(subset=["month_name"])

        fig_monthly = px.bar(
            df_monthly,
            x="month_name",
            y="cnt",
            labels={"month_name": "Month", "cnt": "Total Complaints"},
            color_discrete_sequence=[COLOR_ACCENT_2],
        )
        fig_monthly.update_layout(
            **PLOTLY_LAYOUT,
            title="Complaints by Month (All Years)",
            height=350,
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    st.markdown("---")

    # --- Day of Week Patterns (derived from date) ---
    st.subheader("Weekly Submission Patterns")

    # SQLite strftime('%w', date) returns 0=Sunday, 6=Saturday
    dow_data = run_query(
        "SELECT CAST(strftime('%w', date_received) AS INTEGER) AS dow, COUNT(*) AS cnt "
        "FROM cfpb_complaints "
        "WHERE date_received IS NOT NULL "
        "GROUP BY dow ORDER BY dow"
    )

    if dow_data:
        df_dow = pd.DataFrame(dow_data)
        day_names = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
                     4: "Thursday", 5: "Friday", 6: "Saturday"}
        df_dow["day_name"] = df_dow["dow"].map(day_names)
        df_dow = df_dow.dropna(subset=["day_name"])

        fig_dow = px.bar(
            df_dow,
            x="day_name",
            y="cnt",
            labels={"day_name": "Day of Week", "cnt": "Total Complaints"},
            color_discrete_sequence=[COLOR_ACCENT_1],
        )
        fig_dow.update_layout(
            **PLOTLY_LAYOUT,
            title="Complaints by Day of Week (All Time)",
            height=350,
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown("---")

    # --- Response Timeliness Trends ---
    st.subheader("Response Timeliness Trends")

    timely_trend = run_query(
        "SELECT substr(date_received, 1, 4) AS year, timely_response, COUNT(*) AS cnt "
        "FROM cfpb_complaints "
        "WHERE timely_response IS NOT NULL AND date_received IS NOT NULL "
        "GROUP BY year, timely_response ORDER BY year"
    )

    if timely_trend:
        df_timely_trend = pd.DataFrame(timely_trend)

        # Calculate timeliness rate per year
        pivot = df_timely_trend.pivot_table(
            index="year", columns="timely_response", values="cnt", fill_value=0
        ).reset_index()

        if "Yes" in pivot.columns:
            total_per_year = pivot.get("Yes", 0) + pivot.get("No", 0)
            # Avoid division by zero
            pivot["timely_pct"] = (
                pivot["Yes"] / total_per_year.replace(0, 1) * 100
            ).round(1)

            fig_timely_trend = px.line(
                pivot,
                x="year",
                y="timely_pct",
                markers=True,
                labels={"year": "Year", "timely_pct": "Timely Response %"},
                color_discrete_sequence=[COLOR_ACCENT_5],
            )
            fig_timely_trend.update_layout(
                **PLOTLY_LAYOUT,
                title="Timely Response Rate Over Time",
                yaxis=dict(range=[80, 105]),
                height=400,
            )
            st.plotly_chart(fig_timely_trend, use_container_width=True)
        else:
            st.info("Insufficient timeliness data for trend analysis.")
    else:
        st.info("No timeliness data available.")

    st.markdown("---")

    # --- Emerging Issues (Fastest Growing Issues) ---
    st.subheader("Emerging Issues (Fastest Growing)")
    st.markdown(
        "Issues with the largest increase in complaint volume comparing the "
        "most recent full year to the year before."
    )

    # Find the two most recent full years
    years_data = run_query(
        "SELECT DISTINCT substr(date_received, 1, 4) AS year "
        "FROM cfpb_complaints "
        "WHERE date_received IS NOT NULL "
        "ORDER BY year DESC LIMIT 5"
    )

    if years_data and len(years_data) >= 3:
        # Skip the current partial year (most recent), use the two before it
        recent_year = years_data[1]["year"]
        prev_year = years_data[2]["year"]

        emerging_data = run_query(
            """
            SELECT
                issue,
                SUM(CASE WHEN substr(date_received, 1, 4) = ? THEN 1 ELSE 0 END) AS recent_cnt,
                SUM(CASE WHEN substr(date_received, 1, 4) = ? THEN 1 ELSE 0 END) AS prev_cnt
            FROM cfpb_complaints
            WHERE issue IS NOT NULL
              AND substr(date_received, 1, 4) IN (?, ?)
            GROUP BY issue
            HAVING prev_cnt > 50
            ORDER BY (recent_cnt - prev_cnt) DESC
            LIMIT 15
            """,
            (recent_year, prev_year, recent_year, prev_year),
        )

        if emerging_data:
            df_emerging = pd.DataFrame(emerging_data)
            df_emerging["change"] = df_emerging["recent_cnt"] - df_emerging["prev_cnt"]
            df_emerging["pct_change"] = (
                (df_emerging["change"] / df_emerging["prev_cnt"].replace(0, 1)) * 100
            ).round(1)

            fig_emerging = px.bar(
                df_emerging,
                x="change",
                y="issue",
                orientation="h",
                labels={"change": f"Change ({prev_year} to {recent_year})", "issue": ""},
                color="change",
                color_continuous_scale=["#D63031", "#FDCB6E", "#00CEC9"],
            )
            fig_emerging.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Fastest Growing Issues: {prev_year} vs. {recent_year}",
                yaxis=dict(autorange="reversed"),
                height=500,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_emerging, use_container_width=True)

            # Table view
            with st.expander("View as table"):
                df_emerging_display = df_emerging[
                    ["issue", "prev_cnt", "recent_cnt", "change", "pct_change"]
                ].rename(columns={
                    "issue": "Issue",
                    "prev_cnt": f"{prev_year} Count",
                    "recent_cnt": f"{recent_year} Count",
                    "change": "Absolute Change",
                    "pct_change": "% Change",
                })
                df_emerging_display.index = range(1, len(df_emerging_display) + 1)
                st.dataframe(df_emerging_display, use_container_width=True)
        else:
            st.info("Insufficient data for emerging issue analysis.")
    else:
        st.info("Need at least 3 years of data for emerging issue detection.")

    st.markdown("---")

    # --- Top Companies Year-over-Year ---
    st.subheader("Top 10 Companies: Complaint Trends")

    top_co_data = run_query(
        "SELECT company FROM cfpb_complaints "
        "GROUP BY company ORDER BY COUNT(*) DESC LIMIT 10"
    )

    if top_co_data:
        co_names = [r["company"] for r in top_co_data]
        co_placeholders = ", ".join(["?" for _ in co_names])

        co_trend = run_query(
            f"SELECT substr(date_received, 1, 4) AS year, company, COUNT(*) AS cnt "
            f"FROM cfpb_complaints "
            f"WHERE company IN ({co_placeholders}) AND date_received IS NOT NULL "
            f"GROUP BY year, company ORDER BY year",
            tuple(co_names),
        )

        if co_trend:
            df_co_trend = pd.DataFrame(co_trend)
            fig_co = px.line(
                df_co_trend,
                x="year",
                y="cnt",
                color="company",
                markers=True,
                labels={"year": "Year", "cnt": "Complaints", "company": "Company"},
            )
            fig_co.update_layout(
                **PLOTLY_LAYOUT,
                title="Top 10 Companies: Annual Complaint Volume",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=-0.4, x=0, font=dict(size=9)),
            )
            st.plotly_chart(fig_co, use_container_width=True)

    render_footer()


# ============================================================================
# SECTION 8: FINANCIAL HEALTH
# ============================================================================

elif page == "Financial Health":
    st.title("Financial Health")
    st.markdown("Institution financial metrics from FDIC Call Reports — assets, capital ratios, profitability.")

    _conn = _get_conn()
    if not _table_exists(_conn, "institution_financials"):
        st.info("No financial data loaded yet. Run: `python -m src.cli scrape --source financials`")
        render_footer()
        _conn.close()
        st.stop()
    _conn.close()

    fin_count = run_scalar("SELECT COUNT(DISTINCT cert_number) FROM institution_financials") or 0

    if fin_count == 0:
        st.info("No financial data loaded yet. Run: `python -m src.cli scrape --source financials`")
        render_footer()
    else:
        # KPI cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Institutions w/ Financials", fmt_number(fin_count))

        total_assets_q = run_scalar(
            "SELECT SUM(total_assets) FROM institution_financials WHERE "
            "(cert_number, report_date) IN (SELECT cert_number, MAX(report_date) FROM institution_financials GROUP BY cert_number)"
        )
        k2.metric("Total Assets Tracked", fmt_currency((total_assets_q or 0) * 1000))

        avg_roa = run_scalar(
            "SELECT AVG(roa) FROM institution_financials WHERE "
            "(cert_number, report_date) IN (SELECT cert_number, MAX(report_date) FROM institution_financials GROUP BY cert_number) "
            "AND roa IS NOT NULL"
        )
        k3.metric("Avg ROA", f"{avg_roa:.2f}%" if avg_roa else "N/A")

        avg_tier1 = run_scalar(
            "SELECT AVG(tier1_capital_ratio) FROM institution_financials WHERE "
            "(cert_number, report_date) IN (SELECT cert_number, MAX(report_date) FROM institution_financials GROUP BY cert_number) "
            "AND tier1_capital_ratio IS NOT NULL"
        )
        k4.metric("Avg Tier 1 Capital", f"{avg_tier1:.1f}%" if avg_tier1 else "N/A")

        st.markdown("---")

        # Top 50 institutions by assets with complaint overlay
        st.subheader("Top 50 Institutions by Assets")

        top_assets = run_query("""
            SELECT fi.institution_name, fi.state, f.total_assets, f.roa, f.roe,
                   f.tier1_capital_ratio, f.noncurrent_loans_pct,
                   COALESCE(ic.complaint_count, 0) as complaints
            FROM institution_financials f
            JOIN financial_institutions fi ON f.cert_number = fi.cert_number
            LEFT JOIN (
                SELECT cert_number, COUNT(*) as complaint_count
                FROM institution_complaints GROUP BY cert_number
            ) ic ON fi.cert_number = ic.cert_number
            WHERE (f.cert_number, f.report_date) IN (
                SELECT cert_number, MAX(report_date) FROM institution_financials GROUP BY cert_number
            )
            ORDER BY f.total_assets DESC LIMIT 50
        """)

        if top_assets:
            df_assets = pd.DataFrame(top_assets)
            df_assets["total_assets_b"] = df_assets["total_assets"].apply(
                lambda x: round(x / 1_000_000, 1) if x else 0
            )

            # Scatter: assets vs complaints
            fig_scatter = px.scatter(
                df_assets,
                x="total_assets_b",
                y="complaints",
                hover_name="institution_name",
                size="total_assets_b",
                color="roa",
                color_continuous_scale="RdYlGn",
                labels={
                    "total_assets_b": "Total Assets ($B)",
                    "complaints": "Complaint Count",
                    "roa": "ROA (%)",
                },
            )
            fig_scatter.update_layout(
                **PLOTLY_LAYOUT,
                title="Total Assets vs. Complaint Volume",
                height=500,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Table view
            st.dataframe(
                df_assets[["institution_name", "state", "total_assets_b", "roa", "roe",
                           "tier1_capital_ratio", "noncurrent_loans_pct", "complaints"]].rename(
                    columns={
                        "institution_name": "Institution",
                        "state": "State",
                        "total_assets_b": "Assets ($B)",
                        "roa": "ROA %",
                        "roe": "ROE %",
                        "tier1_capital_ratio": "Tier 1 Capital %",
                        "noncurrent_loans_pct": "Noncurrent Loans %",
                        "complaints": "Complaints",
                    }
                ),
                use_container_width=True,
                height=500,
            )

        st.markdown("---")

        # Financial trend chart for selected institution
        st.subheader("Financial Trends by Institution")

        search_name = st.text_input("Search institution name", key="fin_search")
        if search_name:
            fin_insts = run_query(
                "SELECT DISTINCT fi.cert_number, fi.institution_name FROM institution_financials f "
                "JOIN financial_institutions fi ON f.cert_number = fi.cert_number "
                "WHERE fi.institution_name LIKE ? LIMIT 20",
                (f"%{search_name}%",),
            )
            if fin_insts:
                selected = st.selectbox(
                    "Select institution",
                    [f["institution_name"] for f in fin_insts],
                    key="fin_select",
                )
                cert = next(f["cert_number"] for f in fin_insts if f["institution_name"] == selected)

                trend_data = run_query(
                    "SELECT report_date, total_assets, roa, roe, tier1_capital_ratio, noncurrent_loans_pct "
                    "FROM institution_financials WHERE cert_number = ? ORDER BY report_date",
                    (cert,),
                )
                if trend_data:
                    df_trend = pd.DataFrame(trend_data)
                    fig_trend = px.line(
                        df_trend,
                        x="report_date",
                        y=["roa", "roe", "tier1_capital_ratio"],
                        markers=True,
                        labels={"report_date": "Quarter", "value": "%", "variable": "Metric"},
                    )
                    fig_trend.update_layout(**PLOTLY_LAYOUT, title=f"Financial Trends: {selected}", height=400)
                    st.plotly_chart(fig_trend, use_container_width=True)

        render_footer()


# ============================================================================
# SECTION 9: MULTI-AGENCY ENFORCEMENT
# ============================================================================

elif page == "Multi-Agency Enforcement":
    st.title("Multi-Agency Enforcement")
    st.markdown("Unified enforcement timeline across CFPB, OCC, Federal Reserve, and FDIC.")

    _conn = _get_conn()
    if not _table_exists(_conn, "regulatory_enforcement"):
        st.info("No multi-agency enforcement data loaded. Run: `python -m src.cli scrape --source all-enriched`")
        render_footer()
        _conn.close()
        st.stop()
    _conn.close()

    reg_count = run_scalar("SELECT COUNT(*) FROM regulatory_enforcement") or 0

    if reg_count == 0:
        st.info(
            "No multi-agency enforcement data loaded. Run: "
            "`python -m src.cli scrape --source all-enriched`"
        )
        render_footer()
    else:
        # KPI cards
        agency_counts = run_query(
            "SELECT agency, COUNT(*) as cnt, SUM(COALESCE(penalty_amount, 0)) as total_penalty "
            "FROM regulatory_enforcement GROUP BY agency ORDER BY cnt DESC"
        )

        total_reg_penalties = run_scalar(
            "SELECT SUM(penalty_amount) FROM regulatory_enforcement WHERE penalty_amount > 0"
        ) or 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Actions", fmt_number(reg_count))
        k2.metric("Agencies", fmt_number(len(agency_counts)))
        k3.metric("Total Penalties", fmt_currency(total_reg_penalties))

        institutions_enforced = run_scalar(
            "SELECT COUNT(DISTINCT cert_number) FROM regulatory_enforcement WHERE cert_number IS NOT NULL"
        ) or 0
        k4.metric("Institutions Affected", fmt_number(institutions_enforced))

        st.markdown("---")

        # Enforcement by agency (stacked bar)
        st.subheader("Enforcement Actions by Agency")
        if agency_counts:
            df_agency = pd.DataFrame(agency_counts)
            fig_agency = px.bar(
                df_agency,
                x="agency",
                y="cnt",
                color="agency",
                labels={"agency": "Agency", "cnt": "Actions"},
                color_discrete_sequence=[COLOR_PRIMARY, COLOR_ACCENT_1, COLOR_ACCENT_2, COLOR_ACCENT_3],
            )
            fig_agency.update_layout(**PLOTLY_LAYOUT, title="Enforcement Actions by Agency", height=400, showlegend=False)
            st.plotly_chart(fig_agency, use_container_width=True)

        st.markdown("---")

        # Timeline
        st.subheader("Enforcement Timeline")

        date_filter = st.slider(
            "Filter by year",
            min_value=2000,
            max_value=2026,
            value=(2015, 2026),
            key="reg_year_filter",
        )

        agency_filter = st.multiselect(
            "Filter by agency",
            options=[a["agency"] for a in agency_counts],
            default=[a["agency"] for a in agency_counts],
            key="reg_agency_filter",
        )

        if agency_filter:
            placeholders = ",".join(["?" for _ in agency_filter])
            timeline_data = run_query(
                f"SELECT agency, action_date, institution_name, action_type, penalty_amount, source_url "
                f"FROM regulatory_enforcement "
                f"WHERE agency IN ({placeholders}) "
                f"AND action_date >= ? AND action_date <= ? "
                f"ORDER BY action_date DESC LIMIT 500",
                tuple(agency_filter) + (f"{date_filter[0]}-01-01", f"{date_filter[1]}-12-31"),
            )

            if timeline_data:
                df_timeline = pd.DataFrame(timeline_data)
                st.dataframe(
                    df_timeline.rename(columns={
                        "agency": "Agency",
                        "action_date": "Date",
                        "institution_name": "Institution",
                        "action_type": "Type",
                        "penalty_amount": "Penalty",
                        "source_url": "Source",
                    }),
                    use_container_width=True,
                    height=500,
                )

        st.markdown("---")

        # Top institutions by enforcement penalties
        st.subheader("Top Institutions by Enforcement Penalties")
        top_penalties = run_query("""
            SELECT institution_name, COUNT(*) as actions,
                   SUM(COALESCE(penalty_amount, 0)) as total_penalty,
                   GROUP_CONCAT(DISTINCT agency) as agencies
            FROM regulatory_enforcement
            WHERE institution_name IS NOT NULL
            GROUP BY institution_name
            ORDER BY total_penalty DESC
            LIMIT 25
        """)

        if top_penalties:
            df_penalties = pd.DataFrame(top_penalties)
            fig_pen = px.bar(
                df_penalties.head(15),
                x="institution_name",
                y="total_penalty",
                color="actions",
                labels={"institution_name": "Institution", "total_penalty": "Total Penalties ($)", "actions": "# Actions"},
                color_continuous_scale="Reds",
            )
            fig_pen.update_layout(**PLOTLY_LAYOUT, title="Top 15 Institutions by Enforcement Penalties", height=500)
            fig_pen.update_xaxes(tickangle=45)
            st.plotly_chart(fig_pen, use_container_width=True)

        render_footer()


# ============================================================================
# SECTION 10: INSTITUTION RISK PROFILES
# ============================================================================

elif page == "Institution Risk Profiles":
    st.title("Institution Risk Profiles")
    st.markdown("Composite risk scoring combining complaints, enforcement, financial health, and lending data.")

    _conn = _get_conn()
    if not _table_exists(_conn, "institution_risk_profiles"):
        st.info("No risk profiles computed yet. Run: `python -m src.cli risk-profiles`")
        render_footer()
        _conn.close()
        st.stop()
    _conn.close()

    risk_count = run_scalar("SELECT COUNT(*) FROM institution_risk_profiles") or 0

    if risk_count == 0:
        st.info("No risk profiles computed yet. Run: `python -m src.cli risk-profiles`")
        render_footer()
    else:
        # KPI cards
        tier_data = run_query(
            "SELECT risk_tier, COUNT(*) as cnt FROM institution_risk_profiles "
            "WHERE risk_tier IS NOT NULL GROUP BY risk_tier"
        )
        tier_map = {t["risk_tier"]: t["cnt"] for t in tier_data}

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Profiles", fmt_number(risk_count))
        k2.metric("CRITICAL Risk", fmt_number(tier_map.get("CRITICAL", 0)))
        k3.metric("HIGH Risk", fmt_number(tier_map.get("HIGH", 0)))
        k4.metric("LOW Risk", fmt_number(tier_map.get("LOW", 0)))

        st.markdown("---")

        # Risk tier distribution pie
        st.subheader("Risk Tier Distribution")
        if tier_data:
            df_tier = pd.DataFrame(tier_data)
            tier_colors = {"LOW": COLOR_ACCENT_5, "MEDIUM": COLOR_ACCENT_4, "HIGH": COLOR_ACCENT_3, "CRITICAL": COLOR_DANGER}
            fig_tier = px.pie(
                df_tier,
                values="cnt",
                names="risk_tier",
                color="risk_tier",
                color_discrete_map=tier_colors,
            )
            fig_tier.update_layout(**PLOTLY_LAYOUT, title="Risk Tier Distribution", height=400)
            st.plotly_chart(fig_tier, use_container_width=True)

        st.markdown("---")

        # Risk detail table
        st.subheader("Institution Risk Details")

        risk_filter = st.multiselect(
            "Filter by risk tier",
            options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH"],
            key="risk_tier_filter",
        )

        asset_tier_filter = st.multiselect(
            "Filter by asset tier",
            options=["gsib", "large", "regional", "community", "unknown"],
            default=["gsib", "large", "regional", "community", "unknown"],
            key="asset_tier_filter",
        )

        if risk_filter and asset_tier_filter:
            r_placeholders = ",".join(["?" for _ in risk_filter])
            a_placeholders = ",".join(["?" for _ in asset_tier_filter])

            risk_details = run_query(
                f"SELECT institution_name, asset_tier, complaint_count, "
                f"complaints_per_billion_assets, enforcement_count, total_penalties, "
                f"financial_health_score, complaint_severity_score, overall_risk_score, risk_tier "
                f"FROM institution_risk_profiles "
                f"WHERE risk_tier IN ({r_placeholders}) AND asset_tier IN ({a_placeholders}) "
                f"ORDER BY overall_risk_score ASC LIMIT 200",
                tuple(risk_filter) + tuple(asset_tier_filter),
            )

            if risk_details:
                df_risk = pd.DataFrame(risk_details)
                st.dataframe(
                    df_risk.rename(columns={
                        "institution_name": "Institution",
                        "asset_tier": "Asset Tier",
                        "complaint_count": "Complaints",
                        "complaints_per_billion_assets": "Complaints/$B",
                        "enforcement_count": "Enforcements",
                        "total_penalties": "Penalties ($)",
                        "financial_health_score": "Financial Score",
                        "complaint_severity_score": "Complaint Score",
                        "overall_risk_score": "Risk Score",
                        "risk_tier": "Risk Tier",
                    }),
                    use_container_width=True,
                    height=500,
                )

        st.markdown("---")

        # Peer comparison scatter
        st.subheader("Peer Comparison: Risk Score vs. Assets")
        peer_data = run_query("""
            SELECT institution_name, total_assets, overall_risk_score, risk_tier,
                   complaint_count, enforcement_count, asset_tier
            FROM institution_risk_profiles
            WHERE total_assets IS NOT NULL AND total_assets > 0
            ORDER BY total_assets DESC LIMIT 500
        """)

        if peer_data:
            df_peer = pd.DataFrame(peer_data)
            df_peer["assets_b"] = df_peer["total_assets"].apply(lambda x: x / 1_000_000 if x else 0)

            fig_peer = px.scatter(
                df_peer,
                x="assets_b",
                y="overall_risk_score",
                color="risk_tier",
                hover_name="institution_name",
                size="complaint_count",
                color_discrete_map={"LOW": COLOR_ACCENT_5, "MEDIUM": COLOR_ACCENT_4, "HIGH": COLOR_ACCENT_3, "CRITICAL": COLOR_DANGER},
                labels={"assets_b": "Total Assets ($B)", "overall_risk_score": "Risk Score (higher=safer)", "risk_tier": "Risk Tier"},
            )
            fig_peer.update_layout(**PLOTLY_LAYOUT, title="Risk Score vs. Total Assets", height=500)
            st.plotly_chart(fig_peer, use_container_width=True)

        render_footer()


# ============================================================================
# SECTION 11: FAIR LENDING ANALYSIS
# ============================================================================

elif page == "Fair Lending Analysis":
    st.title("Fair Lending Analysis")
    st.markdown("HMDA mortgage lending patterns — originations, denial rates, and lending footprint.")

    _conn = _get_conn()
    if not _table_exists(_conn, "hmda_institution_summary"):
        st.info("No HMDA data loaded yet. Run: `python -m src.cli scrape --source hmda`")
        render_footer()
        _conn.close()
        st.stop()
    _conn.close()

    hmda_count = run_scalar("SELECT COUNT(*) FROM hmda_institution_summary") or 0

    if hmda_count == 0:
        st.info("No HMDA data loaded yet. Run: `python -m src.cli scrape --source hmda`")
        render_footer()
    else:
        # KPI cards
        hmda_insts = run_scalar("SELECT COUNT(DISTINCT lei) FROM hmda_institution_summary") or 0
        total_originations = run_scalar(
            "SELECT SUM(total_originations) FROM hmda_institution_summary"
        ) or 0
        avg_denial = run_scalar(
            "SELECT AVG(denial_rate) FROM hmda_institution_summary WHERE denial_rate IS NOT NULL"
        )
        total_volume = run_scalar(
            "SELECT SUM(total_loan_volume) FROM hmda_institution_summary"
        ) or 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("HMDA Institutions", fmt_number(hmda_insts))
        k2.metric("Total Originations", fmt_number(total_originations))
        k3.metric("Avg Denial Rate", f"{avg_denial:.1%}" if avg_denial else "N/A")
        k4.metric("Total Loan Volume", fmt_currency(total_volume))

        st.markdown("---")

        # Denial rate distribution
        st.subheader("Denial Rate Distribution")
        denial_data = run_query("""
            SELECT institution_name, denial_rate, total_applications,
                   total_originations, total_denials, cert_number
            FROM hmda_institution_summary
            WHERE denial_rate IS NOT NULL AND total_applications > 10
            ORDER BY denial_rate DESC
            LIMIT 200
        """)

        if denial_data:
            df_denial = pd.DataFrame(denial_data)

            fig_hist = px.histogram(
                df_denial,
                x="denial_rate",
                nbins=30,
                labels={"denial_rate": "Denial Rate"},
                color_discrete_sequence=[COLOR_PRIMARY],
            )
            fig_hist.update_layout(**PLOTLY_LAYOUT, title="Distribution of Mortgage Denial Rates", height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Top deniers table
            st.subheader("Institutions with Highest Denial Rates")
            st.dataframe(
                df_denial.head(50).rename(columns={
                    "institution_name": "Institution",
                    "denial_rate": "Denial Rate",
                    "total_applications": "Applications",
                    "total_originations": "Originations",
                    "total_denials": "Denials",
                }),
                use_container_width=True,
                height=400,
            )

        st.markdown("---")

        # Year-over-year comparison
        st.subheader("HMDA Trends by Year")
        year_data = run_query("""
            SELECT report_year, SUM(total_applications) as apps,
                   SUM(total_originations) as orig, SUM(total_denials) as denied,
                   AVG(denial_rate) as avg_denial
            FROM hmda_institution_summary
            WHERE total_applications IS NOT NULL
            GROUP BY report_year ORDER BY report_year
        """)

        if year_data:
            df_year = pd.DataFrame(year_data)
            fig_year = px.bar(
                df_year,
                x="report_year",
                y=["orig", "denied"],
                barmode="stack",
                labels={"report_year": "Year", "value": "Count", "variable": "Outcome"},
                color_discrete_sequence=[COLOR_ACCENT_5, COLOR_ACCENT_3],
            )
            fig_year.update_layout(**PLOTLY_LAYOUT, title="Mortgage Originations vs. Denials by Year", height=400)
            st.plotly_chart(fig_year, use_container_width=True)

        render_footer()
