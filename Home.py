# ====================
# IMPORTS
# ====================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import locale
from datetime import date
import io
from logics.data_cleaning import calculate_allocated_budget, calculate_vote_utilisation
from helper_functions.llm import FinanceFriendSession, answer_policy_question
import tempfile
import os
import streamlit.components.v1 as components
import html as _html
from dotenv import load_dotenv

# Page config
st.set_page_config(page_title="Finance Friend", layout="wide")
st.title("💼 Finance Friend")

# ================
# PASSWORD OF APP
# ================
from utility import check_password

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()

# =========================
# BUDGET CAPS PER OFFICER
# =========================
budget_per_officer = {
    "Office Supplies": 25,
    "Training": 2000,
    "Staff Welfare": 80,
    "IT": 100
}

# =========================
# DIRECTORATE SELECTION (SIDEBAR)
# =========================
directorate_options = ["Select a Directorate...", "HR", "Finance", "Policy", "Marketing", "Legal"]

st.sidebar.selectbox(
    "Select your Directorate:",
    directorate_options,
    index=0,
    key="selected_directorate"
)

selected = st.session_state.selected_directorate

# =========================
# ALLOCATED BUDGET TABLE
# =========================
df_dept = pd.read_excel("data/department data.xlsx")
allocated_budget_df = calculate_allocated_budget(df_dept, budget_per_officer)

if selected == "Select a Directorate...":
    filtered_df = pd.DataFrame(columns=allocated_budget_df.columns)
    filtered_df.loc[0] = [""] * len(filtered_df.columns)
else:
    filtered_df = allocated_budget_df[allocated_budget_df["Departments"] == selected]

money_columns = ["Office Supplies", "Training", "Staff Welfare", "IT"]
for col in money_columns:
    if col in filtered_df.columns:
        filtered_df[col] = filtered_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) and x != "" else "")

st.write("Budget Allocated")
st.dataframe(filtered_df)

# =========================
# LAYOUT - TWO COLUMNS
# =========================
left_col, right_col = st.columns([2, 1])

# =========================
# LEFT COLUMN – BAR CHART AND TABLE
# =========================
with left_col:
    st.subheader("Vote Usage")

    utilised_df = calculate_vote_utilisation()

    allocated_melted = allocated_budget_df.melt(
        id_vars=["Departments"],
        value_vars=list(budget_per_officer.keys()),
        var_name="Vote Type",
        value_name="Allocated"
    )

    utilised_melted = utilised_df.melt(
        id_vars=["Directorate"],
        var_name="Vote Type",
        value_name="Utilised"
    ).rename(columns={"Directorate": "Departments"})

    comparison_df = pd.merge(allocated_melted, utilised_melted, on=["Departments", "Vote Type"], how="left").fillna(0)
    comparison_df["Exceed"] = comparison_df["Utilised"] > comparison_df["Allocated"]
    comparison_df["Exceed Amount"] = (comparison_df["Utilised"] - comparison_df["Allocated"]).clip(lower=0)
    comparison_df["Remaining"] = (comparison_df["Allocated"] - comparison_df["Utilised"]).clip(lower=0)
    comparison_df["Total"] = comparison_df["Allocated"].replace(0, pd.NA)
    comparison_df["Utilised %"] = comparison_df["Utilised"] / comparison_df["Total"]
    comparison_df["Remaining %"] = comparison_df["Remaining"] / comparison_df["Total"]

    if selected == "Select a Directorate...":
        st.info("Please select a Directorate to view bar chart.")
    else:
        filtered_df = comparison_df[comparison_df["Departments"] == selected].copy()

        if not filtered_df.empty:
            fig = go.Figure()
            legend_shown = set()

            for _, row in filtered_df.iterrows():
                vote = row["Vote Type"]
                utilised_pct = row["Utilised %"] if pd.notnull(row["Utilised %"]) else 0
                remaining_pct = row["Remaining %"] if pd.notnull(row["Remaining %"]) else 0
                utilised_amt = row["Utilised"]
                remaining_amt = row["Remaining"]
                exceed_amt = row["Exceed Amount"]

                if row["Exceed"]:
                    # Exceeded main red bar (100%)
                    fig.add_trace(go.Bar(
                        name="Exceeded",
                        x=[vote],
                        y=[utilised_pct],
                        marker_color="indianred",
                        showlegend="Exceeded" not in legend_shown,
                        hovertemplate=f"Utilised: {utilised_pct:.0%}<br>Exceeded by: ${exceed_amt:,.0f}<extra></extra>",
                        text=None
                    ))
                    legend_shown.add("Exceeded")

                    # Force label to appear above using annotation
                    fig.add_annotation(
                        x=vote,
                        y=0.95,  # slightly above the 100% bar
                        text=f"Exceeded by: ${exceed_amt:,.0f}",
                        showarrow=False,
                        font=dict(size=13, color="white"),
                        xanchor="center"
                    )
                else:
                    # Utilised bar
                    fig.add_trace(go.Bar(
                        name="Utilised",
                        x=[vote],
                        y=[utilised_pct],
                        text=[f"${utilised_amt:,.0f}"],
                        textposition="inside",
                        insidetextanchor="start",
                        marker_color="#1f77b4",
                        showlegend="Utilised" not in legend_shown,
                        hovertemplate=f"Utilised: {utilised_pct:.0%}<br>${utilised_amt:,.0f}<extra></extra>"
                    ))
                    legend_shown.add("Utilised")

                    # Remaining bar
                    fig.add_trace(go.Bar(
                        name="Remaining",
                        x=[vote],
                        y=[remaining_pct],
                        text=[f"${remaining_amt:,.0f}"],
                        textposition="inside",
                        insidetextanchor="end",
                        marker_color="#d3d3d3",
                        showlegend="Remaining" not in legend_shown,
                        hovertemplate=f"Remaining: {remaining_pct:.0%}<br>${remaining_amt:,.0f}<extra></extra>"
                    ))
                    legend_shown.add("Remaining")

            fig.update_layout(
                barmode='stack',
                height=400,
                title=dict(
                    text=f"Utilisation Breakdown – {selected}",
                    font=dict(family="Segoe UI, Roboto, sans-serif", size=16, color="white")
                ),
                font=dict(
                    family="Segoe UI, Roboto, sans-serif",
                    size=14,
                    color="white"
                ),
                xaxis=dict(
                    tickfont=dict(family="Segoe UI, Roboto, sans-serif", size=13, color="white")
                ),
                yaxis=dict(
                    title=dict(text="% Utilisation of Votes", font=dict(family="Segoe UI, Roboto, sans-serif", size=14, color="white")),
                    tickfont=dict(family="Segoe UI, Roboto, sans-serif", size=13, color="white"),
                    tickformat=".0%",
                    range=[0, 1]
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data found for this directorate.")

    # =========================
    # TRANSACTION TABLE
    # =========================
    if selected == "Select a Directorate...":
        st.info("Please select a Directorate to view transactions.")
    else:
        raw_df = pd.read_excel("data/sample data.xlsx")
        raw_df.columns = raw_df.columns.str.strip()
        raw_df["Amount Used"] = raw_df["Amount Used"].replace('[\$,]', '', regex=True).astype(float)

        filtered_txn = raw_df[raw_df["Directorate"] == selected].copy()

        # Filter dropdown
        col_title, col_filter = st.columns([5, 1])
        with col_title:
            st.markdown("### Transactions")
        with col_filter:
            vote_filter = st.selectbox(
                "Filter by Vote Type",
                options=["All"] + list(filtered_txn["Vote Type"].unique()),
                index=0,
                label_visibility="collapsed"
            )

        if vote_filter != "All":
            filtered_txn = filtered_txn[filtered_txn["Vote Type"] == vote_filter]

        # Add Source column
        filtered_txn["Source"] = "Finance GL"

        # Format Journal Date
        filtered_txn["Journal Date"] = pd.to_datetime(filtered_txn["Journal Date"]).dt.strftime("%d-%m-%Y")

        # Prepare display table
        display_cols = ["Journal Date", "Journal Line Description", "Amount Used"]
        display_df = filtered_txn[display_cols].copy()
        display_df["Amount Used"] = display_df["Amount Used"].apply(lambda x: f"${x:,.2f}")
        display_df["Remarks"] = ""

        # Add total row
        total_amt = filtered_txn["Amount Used"].sum()
        total_row = {
            "Journal Date": "",
            "Journal Line Description": "Total",
            "Amount Used": f"${total_amt:,.2f}",
            "Remarks": ""
        }
        display_df = pd.concat([display_df, pd.DataFrame([total_row])], ignore_index=True)

        # Editable Table
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            disabled=["Journal Date", "Journal Line Description", "Amount Used"],
            column_config={"Remarks": st.column_config.TextColumn("Remarks")}
        )

        # CSV buffer
        csv_buffer = io.StringIO()
        edited_df.to_csv(csv_buffer, index=False)
        today_str = date.today().strftime("%Y-%m-%d")
        file_name = f"{selected}_{today_str}_transactions_with_remarks.csv"

        # Info bar and download button aligned: LEFT + RIGHT
        info_col, download_col = st.columns([3.3, 1])
        with info_col:
            st.markdown("""
                <div style="
                    background-color: #0e1117;
                    border-left: 4px solid #2f81f7;
                    padding: 0.6rem 1rem;
                    font-size: 0.85rem;
                    color: #c9d1d9;
                    border-radius: 6px;
                ">
                📝 Edits made here are temporary. Please download the table for offline editing if required.</div>
            """, unsafe_allow_html=True)
        with download_col:
            st.download_button(
                label="⬇️",
                data=csv_buffer.getvalue(),
                file_name=file_name,
                mime="text/csv",
                help="Download CSV",
                key="download_icon_button"
            )

# =========================
# RIGHT COLUMN – Chatbot (ChatGPT-style)
# =========================
with right_col:
    # optional thin left divider for the whole panel
    st.markdown(
        "<div style='border-left:1px solid #cccccc; padding-left:1rem;'>",
        unsafe_allow_html=True
    )

    st.subheader("Chatbot Assistant")

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = FinanceFriendSession(max_history_tokens=1800)

    def _strip_backticks(text: str) -> str:
        if text.startswith("```") and text.endswith("```"):
            return text.strip("`").strip()
        return text

    bubble_style_user = """
        background-color: #2d2d2d;
        color: #f5f5f5;
        padding: 0.6rem 0.9rem;
        border-radius: 0.6rem;
        margin: 0.3rem 0;
        max-width: 90%;
        align-self: flex-end;
        word-wrap: break-word;
        white-space: pre-wrap;
    """
    bubble_style_bot = """
        background-color: #1e1e1e;
        color: #e0e0e0;
        padding: 0.6rem 0.9rem;
        border-radius: 0.6rem;
        margin: 0.3rem 0;
        max-width: 90%;
        align-self: flex-start;
        word-wrap: break-word;
        white-space: pre-wrap;
    """

    # ---- RENDER CHAT HISTORY (scrollable)
    history = st.session_state.chat_session.build_token_capped_history()
    messages_html = ""
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        # strip code fences and escape HTML to avoid injection
        content = _html.escape(_strip_backticks(content))
        # convert line breaks to <br> for nicer formatting
        content = content.replace("\n", "<br>")
        style = bubble_style_user if role == "user" else bubble_style_bot
        messages_html += f"<div style='{style}'>{content}</div>"

    components.html(
        f"""
        <div id="chat-container"
            style="display:flex; flex-direction:column; height:400px; overflow-y:auto; padding-right:8px; font-family: Arial, Helvetica, sans-serif; font-size: 14px; line-height: 1.4;">
            {messages_html}
        </div>
        <script>
        const el = document.getElementById('chat-container');
        if (el) {{
            el.scrollTop = el.scrollHeight; // auto-scroll to latest
        }}
        </script>
        """,
        height=420,
    )

    # ---- INPUT
    user_q = st.chat_input("Ask about Finance policy…")
    if user_q:
        with st.spinner("Thinking..."):
            _ = answer_policy_question(user_q, session=st.session_state.chat_session)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # close divider wrapper
        
# =========================
# DISCLAIMER (BOTTOM OF PAGE)
# =========================
st.markdown("""
<div style="font-size: 0.95em; color: #b8860b; margin-top: 1.5em;">
<strong>Disclaimer:</strong><br>
1. For complete and official training records, please contact your training coordinator.<br><br>
2. The information presented excludes:
<ul style="margin-top: 0.2em;">
    <li>Outstanding claims or payments that have not yet been processed;</li>
    <li>Courses that officers have registered for but not yet attended;</li>
    <li>Payments pending adjustments, such as those arising from claims filed under incorrect expense types or with missing sub-accounts.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    """
    <div style="font-size: 0.85em; color: grey;">
    <strong>IMPORTANT NOTICE:</strong> This web application is developed as a proof-of-concept prototype.
    The information provided here is <strong>NOT</strong> intended for actual usage and should not be relied upon for making any decisions,
    especially those related to financial, legal, or healthcare matters.<br><br>

    Furthermore, please be aware that the LLM may generate inaccurate or incorrect information.
    You assume full responsibility for how you use any generated output.<br>

    Always consult with qualified professionals for accurate and personalized advice.
    </div>
    """,
    unsafe_allow_html=True
)

