# import streamlit as st
# import pandas as pd
# import os
# from datetime import date
# import calendar

# HOMEWORK_EXCEL_FILE = "homework_data.xlsx"

# def get_sheet_name_from_date(d: date):
#     return d.strftime("%b %Y")

# def get_days_in_month(year, month):
#     return calendar.monthrange(year, month)[1]

# def load_homework_rules():
#     default = [
#         {"class_start": 1, "class_end": 5, "pages": 5},
#         {"class_start": 6, "class_end": 8, "pages": 10},
#         {"class_start": 9, "class_end": 10, "pages": 15},
#         {"class_start": 11, "class_end": 12, "pages": 20}
#     ]
#     if os.path.exists("homework_rules.json"):
#         import json
#         with open("homework_rules.json", "r") as f:
#             return json.load(f)
#     else:
#         return default

# def get_expected_pages(class_str):
#     try:
#         cls = int(class_str)
#     except:
#         return 10
#     rules = load_homework_rules()
#     for rule in rules:
#         if rule["class_start"] <= cls <= rule["class_end"]:
#             return rule["pages"]
#     return 10

# def save_homework_to_excel(entry_df, sheet_name, target_date):
#     year = target_date.year
#     month = target_date.month
#     max_day = get_days_in_month(year, month)
#     day_columns = [str(d) for d in range(1, max_day + 1)]

#     entry_df = entry_df.copy()
#     entry_df["Day"] = entry_df["Date"].apply(lambda d: str(d.day))

#     if os.path.exists(HOMEWORK_EXCEL_FILE):
#         try:
#             with pd.ExcelFile(HOMEWORK_EXCEL_FILE) as xls:
#                 all_sheets = {}
#                 for sheet in xls.sheet_names:
#                     all_sheets[sheet] = pd.read_excel(xls, sheet_name=sheet)
#         except:
#             all_sheets = {}
#     else:
#         all_sheets = {}

#     if sheet_name in all_sheets:
#         existing_wide = all_sheets[sheet_name]
#         id_vars = ["Class", "Section", "Student", "Entered By"]
#         day_cols = [col for col in existing_wide.columns if col in day_columns]
#         if day_cols:
#             existing_long = existing_wide.melt(
#                 id_vars=id_vars,
#                 value_vars=day_cols,
#                 var_name="Day",
#                 value_name="Pages Done"
#             )
#             existing_long = existing_long.dropna(subset=["Pages Done"])
#             existing_long["Date"] = existing_long["Day"].apply(lambda d: date(year, month, int(d)))
#             existing_long = existing_long[
#                 (pd.to_datetime(existing_long["Date"]).dt.month == month) &
#                 (pd.to_datetime(existing_long["Date"]).dt.year == year)
#             ]
#         else:
#             existing_long = pd.DataFrame()
#     else:
#         existing_long = pd.DataFrame()

#     combined_long = pd.concat([existing_long, entry_df], ignore_index=True)
#     combined_long.drop_duplicates(
#         subset=["Date", "Class", "Section", "Student"],
#         keep="last",
#         inplace=True
#     )

#     pivot_df = combined_long.pivot_table(
#         index=["Class", "Section", "Student", "Entered By"],
#         columns="Day",
#         values="Pages Done",
#         aggfunc="first"
#     ).reset_index()

#     for day in day_columns:
#         if day not in pivot_df.columns:
#             pivot_df[day] = None

#     final_cols = ["Class", "Section", "Student", "Entered By"] + day_columns
#     pivot_df = pivot_df[final_cols]

#     with pd.ExcelWriter(HOMEWORK_EXCEL_FILE, engine="openpyxl", mode="w") as writer:
#         for sheet, df in all_sheets.items():
#             if sheet != sheet_name:
#                 df.to_excel(writer, sheet_name=sheet, index=False)
#         pivot_df.to_excel(writer, sheet_name=sheet_name, index=False)

# def load_homework_data():
#     if not os.path.exists(HOMEWORK_EXCEL_FILE):
#         return pd.DataFrame()
#     all_dfs = []
#     with pd.ExcelFile(HOMEWORK_EXCEL_FILE) as xls:
#         for sheet in xls.sheet_names:
#             df = pd.read_excel(xls, sheet_name=sheet)
#             id_vars = ["Class", "Section", "Student", "Entered By"]
#             day_cols = [col for col in df.columns if col.isdigit()]
#             if not day_cols:
#                 continue
#             melted = df.melt(
#                 id_vars=id_vars,
#                 value_vars=day_cols,
#                 var_name="Day",
#                 value_name="Pages Done"
#             )
#             melted = melted.dropna(subset=["Pages Done"])
#             try:
#                 month_name, year_str = sheet.split()
#                 month_num = list(calendar.month_abbr).index(month_name)
#                 year = int(year_str)
#                 melted["Date"] = melted["Day"].apply(
#                     lambda d: date(year, month_num, int(d)) if str(d).isdigit() else None
#                 )
#                 melted = melted.dropna(subset=["Date"])
#                 all_dfs.append(melted)
#             except Exception:
#                 continue
#     return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# # ✅ FINAL CORRECTED FUNCTION — MATCHES YOUR SCREENSHOT EXACTLY
# def show_homework_entry_ui(selected_class, selected_section, student_list):
#     if not selected_class or not selected_section:
#         st.error("❌ Class or Section not selected.")
#         return
#     if not student_list:
#         st.error("❌ No students found.")
#         return

#     st.markdown('<div class="section-header">✏️ Enter Daily Homework Pages</div>', unsafe_allow_html=True)
#     selected_date = st.date_input("📅 Select Date", value=date.today())
#     st.subheader(f"Class {selected_class} {selected_section} • {selected_date.strftime('%A, %d %B %Y')}")

#     # Load existing data for this exact class, section, and date
#     existing_dict = {}
#     existing_remark_dict = {}
#     existing_df = load_homework_data()
#     if not existing_df.empty:
#         existing_df["Date"] = pd.to_datetime(existing_df["Date"]).dt.date
#         filtered = existing_df[
#             (existing_df["Class"] == selected_class) &
#             (existing_df["Section"] == selected_section) &
#             (existing_df["Date"] == selected_date)
#         ]
#         for _, row in filtered.iterrows():
#             student_name = row["Student"]
#             val = row["Pages Done"]
#             if isinstance(val, str) and val.startswith("Remark: "):
#                 existing_dict[student_name] = 0
#                 existing_remark_dict[student_name] = val[8:]
#             else:
#                 existing_dict[student_name] = int(val) if pd.notna(val) else 0
#                 existing_remark_dict[student_name] = ""

#     # Build data with S.No.
#     initial_data = []
#     for idx, student in enumerate(student_list, start=1):
#         pages = existing_dict.get(student, 0)
#         remark = existing_remark_dict.get(student, "")
#         initial_data.append({"S.No.": idx, "Student": student, "Pages Done": pages, "Remark": remark})

#     # Display rows with proper alignment
#     for i, row in enumerate(initial_data):
#         cols = st.columns([0.5, 3, 1, 2.2])  # S.No. | Name | Pages | Remark
#         with cols[0]:
#             st.markdown(f"<b>{row['S.No.']}. </b>", unsafe_allow_html=True)
#         with cols[1]:
#             st.markdown(f"<b>{row['Student']}</b>", unsafe_allow_html=True)
#         with cols[2]:
#             new_pages = st.number_input(
#                 "",
#                 min_value=0,
#                 max_value=100,
#                 value=row["Pages Done"],
#                 step=1,
#                 key=f"pages_{i}"
#             )
#             initial_data[i]["Pages Done"] = new_pages
#         with cols[3]:
#             new_remark = st.text_input(
#                 "",
#                 value=row["Remark"],
#                 placeholder="Absent / Medical / Other",
#                 key=f"remark_{i}"
#             )
#             initial_data[i]["Remark"] = new_remark

#     if st.button("✅ Submit Homework Data"):
#         df = pd.DataFrame(initial_data)
#         df["Date"] = selected_date
#         df["Class"] = selected_class
#         df["Section"] = selected_section
#         df["Entered By"] = st.session_state.get("username", "guruji")

#         def process(row):
#             p = row["Pages Done"]
#             r = row["Remark"]
#             if p == 0 and r.strip():
#                 return f"Remark: {r.strip()}"
#             return p if p > 0 else 0

#         df["Pages Done"] = df.apply(process, axis=1)
#         sheet_name = get_sheet_name_from_date(selected_date)
#         save_homework_to_excel(df, sheet_name, selected_date)
#         st.success("✅ Homework data saved successfully!")
#         st.rerun()

# # Other helper functions (for reports)
# def show_monthly_status_view(selected_class, selected_section, student_list):
#     st.markdown('<div class="section-header">📅 Monthly Homework Status</div>', unsafe_allow_html=True)
#     month_names = ["January", "February", "March", "April", "May", "June",
#                    "July", "August", "September", "October", "November", "December"]
#     selected_month_name = st.selectbox("📅 Select Month", month_names, index=date.today().month - 1)
#     selected_year = st.number_input("🗓️ Year", min_value=2020, max_value=2030, value=date.today().year)

#     df = load_homework_data()
#     if df.empty:
#         st.info("No homework entries yet.")
#         return

#     df["Date"] = pd.to_datetime(df["Date"])
#     month = month_names.index(selected_month_name) + 1
#     monthly_df = df[
#         (df["Class"] == selected_class) &
#         (df["Section"] == selected_section) &
#         (df["Date"].dt.year == selected_year) &
#         (df["Date"].dt.month == month)
#     ]

#     if monthly_df.empty:
#         st.warning("No data for this month.")
#         return

#     year = selected_year
#     max_day = calendar.monthrange(year, month)[1]
#     filled_dates = set(monthly_df["Date"].dt.date)

#     st.subheader(f"📅 {selected_month_name} {selected_year}")
#     cal = calendar.monthcalendar(year, month)
#     day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
#     cols = st.columns(7)
#     for i, day in enumerate(day_names):
#         cols[i].markdown(f"<div style='text-align: center; font-weight: bold;'>{day}</div>", unsafe_allow_html=True)

#     for week in cal:
#         cols = st.columns(7)
#         for i, day_num in enumerate(week):
#             if day_num == 0:
#                 cols[i].markdown("", unsafe_allow_html=True)
#             else:
#                 current_date = date(year, month, day_num)
#                 color = "#4CAF50" if current_date in filled_dates else "#F44336"
#                 text_color = "white"
#                 cols[i].markdown(
#                     f"<div style='background: {color}; color: {text_color}; border-radius: 8px; padding: 8px; text-align: center; font-weight: bold;'>{day_num}</div>",
#                     unsafe_allow_html=True
#                 )

#     st.markdown("---")
#     st.caption("🟢 = Homework Filled | 🔴 = Not Filled")

# def show_homework_daily_report(selected_class, selected_section, student_list):
#     st.markdown('<div class="section-header">📅 Daily Homework Report</div>', unsafe_allow_html=True)
#     selected_date = st.date_input("📅 Select Date", value=date.today())
#     df = load_homework_data()
#     if df.empty:
#         st.info("No homework entries yet.")
#         return

#     daily_df = df[
#         (df["Class"] == selected_class) &
#         (df["Section"] == selected_section) &
#         (df["Date"] == pd.to_datetime(selected_date).date())
#     ]

#     if daily_df.empty:
#         st.warning(f"No data for {selected_date.strftime('%d %B %Y')}.")
#         return

#     full_list = pd.DataFrame(student_list, columns=["Student"])
#     merged = full_list.merge(daily_df[["Student", "Pages Done"]], on="Student", how="left")
#     merged["Pages Done Display"] = merged["Pages Done"].apply(
#         lambda x: x if isinstance(x, str) else int(x) if pd.notna(x) else 0
#     )
#     expected = get_expected_pages(selected_class)
#     merged["Expected"] = expected
#     merged["Status"] = merged["Pages Done"].apply(
#         lambda x: "✅ Done" if (isinstance(x, (int, float)) and x >= expected) else "⚠️ Incomplete"
#     )

#     st.subheader(f"📊 Homework Status for {selected_date.strftime('%d %B %Y')}")
#     st.dataframe(
#         merged[["Student", "Pages Done Display", "Expected", "Status"]],
#         use_container_width=True,
#         hide_index=True
#     )

#     completed = len(merged[(merged["Pages Done"].apply(lambda x: isinstance(x, (int, float))) & (merged["Pages Done"] >= expected))])
#     total = len(merged)
#     st.metric("Students Completed", f"{completed}/{total}", f"{(completed/total)*100:.1f}% on track")

#     if os.path.exists(HOMEWORK_EXCEL_FILE):
#         with open(HOMEWORK_EXCEL_FILE, "rb") as f:
#             st.download_button(
#                 "📥 Download Full Homework Data (Excel)",
#                 f,
#                 file_name="homework_data.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )

# def show_homework_monthly_report(selected_class, selected_section, student_list):
#     st.markdown('<div class="section-header">📊 Monthly Homework Report</div>', unsafe_allow_html=True)
#     month_names = ["January", "February", "March", "April", "May", "June",
#                    "July", "August", "September", "October", "November", "December"]
#     selected_month_name = st.selectbox("📅 Select Month", month_names, index=date.today().month - 1)
#     selected_month = month_names.index(selected_month_name) + 1
#     selected_year = st.number_input("🗓️ Year", min_value=2020, max_value=2030, value=date.today().year)

#     df = load_homework_data()
#     if df.empty:
#         st.info("No homework entries yet.")
#         return

#     df["Date"] = pd.to_datetime(df["Date"])
#     monthly_df = df[
#         (df["Class"] == selected_class) &
#         (df["Section"] == selected_section) &
#         (df["Date"].dt.year == selected_year) &
#         (df["Date"].dt.month == selected_month)
#     ]

#     if monthly_df.empty:
#         st.warning("No data for this month.")
#         return

#     monthly_df["Numeric Pages"] = monthly_df["Pages Done"].apply(
#         lambda x: x if isinstance(x, (int, float)) else 0
#     )

#     student_avg = monthly_df.groupby("Student")["Numeric Pages"].agg(["mean", "count"]).reset_index()
#     student_avg.columns = ["Student", "Avg Pages/Day", "Days Submitted"]
#     student_avg["Avg Pages/Day"] = student_avg["Avg Pages/Day"].round(1)
#     expected = get_expected_pages(selected_class)
#     student_avg["Expected"] = expected
#     student_avg["% of Expected"] = (student_avg["Avg Pages/Day"] / expected * 100).round(1)

#     st.subheader("📈 Student Performance (Monthly)")
#     st.dataframe(student_avg, use_container_width=True, hide_index=True)
#     class_avg = student_avg["Avg Pages/Day"].mean()
#     st.metric("Class Average (Pages/Day)", f"{class_avg:.1f}", f"Expected: {expected}")

#     st.subheader("💡 Insights & Recommendations")
#     below_60 = student_avg[student_avg["% of Expected"] < 60]
#     below_30 = student_avg[student_avg["% of Expected"] < 30]
#     if len(below_30) > 0:
#         st.error(f"⚠️ {len(below_30)} students are doing less than 30% of expected homework.")
#     elif len(below_60) > len(student_list) * 0.4:
#         st.warning("⚠️ Over 40% of the class is below 60% expected homework.")
#     else:
#         st.success("✅ Most students are meeting homework expectations.")

#     if class_avg < expected * 0.7:
#         st.markdown("🔹 **Suggestion**: Assign lighter but more frequent homework.")
#     if class_avg > expected * 1.2:
#         st.markdown("🔹 **Suggestion**: Homework load may be excessive.")

#     import plotly.express as px
#     fig = px.bar(
#         student_avg,
#         x="Student",
#         y="Avg Pages/Day",
#         color="% of Expected",
#         color_continuous_scale=["red", "yellow", "green"],
#         text="Avg Pages/Day"
#     )
#     fig.add_hline(y=expected, line_dash="dot", line_color="blue", annotation_text="Expected")
#     fig.update_layout(xaxis_tickangle=-45)
#     st.plotly_chart(fig, use_container_width=True)

#     if os.path.exists(HOMEWORK_EXCEL_FILE):
#         with open(HOMEWORK_EXCEL_FILE, "rb") as f:
#             st.download_button(
#                 "📥 Download Full Homework Data (Excel)",
#                 f,
#                 file_name="homework_data.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )









import streamlit as st
import pandas as pd
import os
from datetime import date
import calendar

HOMEWORK_EXCEL_FILE = "homework_data.xlsx"

def get_sheet_name_from_date(d: date):
    return d.strftime("%b %Y")

def get_days_in_month(year, month):
    return calendar.monthrange(year, month)[1]

def load_homework_rules():
    default = [
        {"class_start": 1, "class_end": 5, "pages": 5},
        {"class_start": 6, "class_end": 8, "pages": 10},
        {"class_start": 9, "class_end": 10, "pages": 15},
        {"class_start": 11, "class_end": 12, "pages": 20}
    ]
    if os.path.exists("homework_rules.json"):
        import json
        with open("homework_rules.json", "r") as f:
            return json.load(f)
    else:
        return default

def get_expected_pages(class_str):
    try:
        cls = int(class_str)
    except:
        return 10
    rules = load_homework_rules()
    for rule in rules:
        if rule["class_start"] <= cls <= rule["class_end"]:
            return rule["pages"]
    return 10

def save_homework_to_excel(entry_df, sheet_name, target_date):
    year = target_date.year
    month = target_date.month
    max_day = get_days_in_month(year, month)
    day_columns = [str(d) for d in range(1, max_day + 1)]

    entry_df = entry_df.copy()
    entry_df["Day"] = entry_df["Date"].apply(lambda d: str(d.day))

    if os.path.exists(HOMEWORK_EXCEL_FILE):
        try:
            with pd.ExcelFile(HOMEWORK_EXCEL_FILE) as xls:
                all_sheets = {}
                for sheet in xls.sheet_names:
                    all_sheets[sheet] = pd.read_excel(xls, sheet_name=sheet)
        except:
            all_sheets = {}
    else:
        all_sheets = {}

    if sheet_name in all_sheets:
        existing_wide = all_sheets[sheet_name]
        id_vars = ["Class", "Section", "Student", "Entered By"]
        day_cols = [col for col in existing_wide.columns if col in day_columns]
        if day_cols:
            existing_long = existing_wide.melt(
                id_vars=id_vars,
                value_vars=day_cols,
                var_name="Day",
                value_name="Pages Done"
            )
            existing_long = existing_long.dropna(subset=["Pages Done"])
            existing_long["Date"] = existing_long["Day"].apply(lambda d: date(year, month, int(d)))
            existing_long = existing_long[
                (pd.to_datetime(existing_long["Date"]).dt.month == month) &
                (pd.to_datetime(existing_long["Date"]).dt.year == year)
            ]
        else:
            existing_long = pd.DataFrame()
    else:
        existing_long = pd.DataFrame()

    combined_long = pd.concat([existing_long, entry_df], ignore_index=True)
    combined_long.drop_duplicates(
        subset=["Date", "Class", "Section", "Student"],
        keep="last",
        inplace=True
    )

    pivot_df = combined_long.pivot_table(
        index=["Class", "Section", "Student", "Entered By"],
        columns="Day",
        values="Pages Done",
        aggfunc="first"
    ).reset_index()

    for day in day_columns:
        if day not in pivot_df.columns:
            pivot_df[day] = None

    final_cols = ["Class", "Section", "Student", "Entered By"] + day_columns
    pivot_df = pivot_df[final_cols]

    with pd.ExcelWriter(HOMEWORK_EXCEL_FILE, engine="openpyxl", mode="w") as writer:
        for sheet, df in all_sheets.items():
            if sheet != sheet_name:
                df.to_excel(writer, sheet_name=sheet, index=False)
        pivot_df.to_excel(writer, sheet_name=sheet_name, index=False)

def load_homework_data():
    if not os.path.exists(HOMEWORK_EXCEL_FILE):
        return pd.DataFrame()
    all_dfs = []
    with pd.ExcelFile(HOMEWORK_EXCEL_FILE) as xls:
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            id_vars = ["Class", "Section", "Student", "Entered By"]
            day_cols = [col for col in df.columns if col.isdigit()]
            if not day_cols:
                continue
            melted = df.melt(
                id_vars=id_vars,
                value_vars=day_cols,
                var_name="Day",
                value_name="Pages Done"
            )
            melted = melted.dropna(subset=["Pages Done"])
            try:
                month_name, year_str = sheet.split()
                month_num = list(calendar.month_abbr).index(month_name)
                year = int(year_str)
                melted["Date"] = melted["Day"].apply(
                    lambda d: date(year, month_num, int(d)) if str(d).isdigit() else None
                )
                melted = melted.dropna(subset=["Date"])
                all_dfs.append(melted)
            except Exception:
                continue
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def show_homework_entry_ui(selected_class, selected_section, student_list):
    if not selected_class or not selected_section:
        st.error("❌ Class or Section not selected.")
        return
    if not student_list:
        st.error("❌ No students found.")
        return

    st.markdown('<div class="section-header">✏️ Enter Daily Homework Pages</div>', unsafe_allow_html=True)
    selected_date = st.date_input("📅 Select Date", value=date.today())
    st.subheader(f"Class {selected_class} {selected_section} • {selected_date.strftime('%A, %d %B %Y')}")

    existing_df = load_homework_data()
    if not existing_df.empty:
        existing_df["Date"] = pd.to_datetime(existing_df["Date"]).dt.date
        existing_today = existing_df[
            (existing_df["Class"] == selected_class) &
            (existing_df["Section"] == selected_section) &
            (existing_df["Date"] == selected_date)
        ]
        existing_dict = dict(zip(existing_today["Student"], existing_today["Pages Done"]))
    else:
        existing_dict = {}

    initial_data = []
    for student in student_list:
        pages_done = int(existing_dict.get(student, 0))
        # ✅ Wrap student name in **bold** HTML
        initial_data.append({"Student": f"**{student}**", "Pages Done": pages_done})

    if len(initial_data) == 0:
        st.info("ℹ️ No students to enter homework for.")
        return

    # ✅ Tab navigation works by default; student name is bold
    edited_df = st.data_editor(
        pd.DataFrame(initial_data),
        column_config={
            "Student": st.column_config.TextColumn("Student", disabled=True),
            "Pages Done": st.column_config.NumberColumn(
                "Pages Completed",
                min_value=0,
                max_value=100,
                step=1,
                format="%d"
            )
        },
        hide_index=True,
        use_container_width=True,
        key="homework_editor"
    )

    if st.button("✅ Submit Homework Data"):
        # ✅ Strip bold markers before saving
        clean_df = edited_df.copy()
        clean_df["Student"] = clean_df["Student"].str.replace("**", "", regex=False)
        clean_df["Date"] = selected_date
        clean_df["Class"] = selected_class
        clean_df["Section"] = selected_section
        clean_df["Entered By"] = st.session_state["username"]
        sheet_name = get_sheet_name_from_date(selected_date)
        save_homework_to_excel(clean_df, sheet_name, selected_date)
        st.success("✅ Homework data saved successfully to Excel!")
        st.rerun()

# 🔥 CALENDAR VIEW
def show_monthly_status_view(selected_class, selected_section, student_list):
    st.markdown('<div class="section-header">📅 Monthly Homework Status</div>', unsafe_allow_html=True)
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    selected_month_name = st.selectbox("📅 Select Month", month_names, index=date.today().month - 1)
    selected_year = st.number_input("🗓️ Year", min_value=2020, max_value=2030, value=date.today().year)

    df = load_homework_data()
    if df.empty:
        st.info("No homework entries yet.")
        return

    df["Date"] = pd.to_datetime(df["Date"])
    month = month_names.index(selected_month_name) + 1
    monthly_df = df[
        (df["Class"] == selected_class) &
        (df["Section"] == selected_section) &
        (df["Date"].dt.year == selected_year) &
        (df["Date"].dt.month == month)
    ]

    if monthly_df.empty:
        st.warning("No data for this month.")
        return

    year = selected_year
    max_day = calendar.monthrange(year, month)[1]
    filled_dates = set(monthly_df["Date"].dt.date)

    st.subheader(f"📅 {selected_month_name} {selected_year}")
    cal = calendar.monthcalendar(year, month)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    cols = st.columns(7)
    for i, day in enumerate(day_names):
        cols[i].markdown(f"<div style='text-align: center; font-weight: bold;'>{day}</div>", unsafe_allow_html=True)

    for week in cal:
        cols = st.columns(7)
        for i, day_num in enumerate(week):
            if day_num == 0:
                cols[i].markdown("", unsafe_allow_html=True)
            else:
                current_date = date(year, month, day_num)
                color = "#4CAF50" if current_date in filled_dates else "#F44336"
                text_color = "white"
                cols[i].markdown(
                    f"<div style='background: {color}; color: {text_color}; border-radius: 8px; "
                    f"padding: 8px; text-align: center; font-weight: bold;'>{day_num}</div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.caption("🟢 = Homework Filled | 🔴 = Not Filled")

# Daily & Monthly Reports
def show_homework_daily_report(selected_class, selected_section, student_list):
    st.markdown('<div class="section-header">📅 Daily Homework Report</div>', unsafe_allow_html=True)
    selected_date = st.date_input("📅 Select Date", value=date.today())
    df = load_homework_data()
    if df.empty:
        st.info("No homework entries yet.")
        return

    daily_df = df[
        (df["Class"] == selected_class) &
        (df["Section"] == selected_section) &
        (df["Date"] == pd.to_datetime(selected_date).date())
    ]

    if daily_df.empty:
        st.warning(f"No data for {selected_date.strftime('%d %B %Y')}.")
        return

    full_list = pd.DataFrame(student_list, columns=["Student"])
    merged = full_list.merge(daily_df[["Student", "Pages Done"]], on="Student", how="left")
    merged["Pages Done"] = merged["Pages Done"].fillna(0).astype(int)
    expected = get_expected_pages(selected_class)
    merged["Expected"] = expected
    merged["Status"] = merged["Pages Done"].apply(lambda x: "✅ Done" if x >= expected else "⚠️ Incomplete")

    st.subheader(f"📊 Homework Status for {selected_date.strftime('%d %B %Y')}")
    st.dataframe(
        merged[["Student", "Pages Done", "Expected", "Status"]],
        use_container_width=True,
        hide_index=True
    )

    completed = len(merged[merged["Pages Done"] >= expected])
    total = len(merged)
    st.metric("Students Completed", f"{completed}/{total}", f"{(completed/total)*100:.1f}% on track")

    if os.path.exists(HOMEWORK_EXCEL_FILE):
        with open(HOMEWORK_EXCEL_FILE, "rb") as f:
            st.download_button(
                "📥 Download Full Homework Data (Excel)",
                f,
                file_name="homework_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def show_homework_monthly_report(selected_class, selected_section, student_list):
    st.markdown('<div class="section-header">📊 Monthly Homework Report</div>', unsafe_allow_html=True)
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    selected_month_name = st.selectbox("📅 Select Month", month_names, index=date.today().month - 1)
    selected_month = month_names.index(selected_month_name) + 1
    selected_year = st.number_input("🗓️ Year", min_value=2020, max_value=2030, value=date.today().year)

    df = load_homework_data()
    if df.empty:
        st.info("No homework entries yet.")
        return

    df["Date"] = pd.to_datetime(df["Date"])
    monthly_df = df[
        (df["Class"] == selected_class) &
        (df["Section"] == selected_section) &
        (df["Date"].dt.year == selected_year) &
        (df["Date"].dt.month == selected_month)
    ]

    if monthly_df.empty:
        st.warning("No data for this month.")
        return

    student_avg = monthly_df.groupby("Student")["Pages Done"].agg(["mean", "count"]).reset_index()
    student_avg.columns = ["Student", "Avg Pages/Day", "Days Submitted"]
    student_avg["Avg Pages/Day"] = student_avg["Avg Pages/Day"].round(1)
    expected = get_expected_pages(selected_class)
    student_avg["Expected"] = expected
    student_avg["% of Expected"] = (student_avg["Avg Pages/Day"] / expected * 100).round(1)

    st.subheader("📈 Student Performance (Monthly)")
    st.dataframe(student_avg, use_container_width=True, hide_index=True)
    class_avg = student_avg["Avg Pages/Day"].mean()
    st.metric("Class Average (Pages/Day)", f"{class_avg:.1f}", f"Expected: {expected}")

    st.subheader("💡 Insights & Recommendations")
    below_60 = student_avg[student_avg["% of Expected"] < 60]
    below_30 = student_avg[student_avg["% of Expected"] < 30]
    if len(below_30) > 0:
        st.error(f"⚠️ {len(below_30)} students are doing less than 30% of expected homework.")
    elif len(below_60) > len(student_list) * 0.4:
        st.warning("⚠️ Over 40% of the class is below 60% expected homework.")
    else:
        st.success("✅ Most students are meeting homework expectations.")

    if class_avg < expected * 0.7:
        st.markdown("🔹 **Suggestion**: Assign lighter but more frequent homework.")
    if class_avg > expected * 1.2:
        st.markdown("🔹 **Suggestion**: Homework load may be excessive.")

    import plotly.express as px
    fig = px.bar(
        student_avg,
        x="Student",
        y="Avg Pages/Day",
        color="% of Expected",
        color_continuous_scale=["red", "yellow", "green"],
        text="Avg Pages/Day"
    )
    fig.add_hline(y=expected, line_dash="dot", line_color="blue", annotation_text="Expected")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    if os.path.exists(HOMEWORK_EXCEL_FILE):
        with open(HOMEWORK_EXCEL_FILE, "rb") as f:
            st.download_button(
                "📥 Download Full Homework Data (Excel)",
                f,
                file_name="homework_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )