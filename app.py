# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import numpy as np

# # # Set page config
# # st.set_page_config(
# #     page_title="📊 Student Performance Analysis Dashboard",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # Custom CSS for styling
# # st.markdown(
# #     """
# #     <style>
# #     .main-header {
# #         font-size: 2.5rem;
# #         color: #2c3e50;
# #         text-align: center;
# #         margin-bottom: 1rem;
# #     }
# #     .sub-header {
# #         font-size: 1.5rem;
# #         color: #34495e;
# #         text-align: center;
# #         margin-bottom: 0.5rem;
# #     }
# #     .info-box {
# #         background-color: #f8f9fa;
# #         padding: 1rem;
# #         border-radius: 0.5rem;
# #         margin-top: 1rem;
# #     }
# #     </style>
# #     """,
# #     unsafe_allow_html=True
# # )

# # # Define subjects (as per your Excel)
# # SUBJECTS = ["ENGLISH", "SANSKRIT/URDU", "MATHEMATICS", "SCIENCE AND TECHNOLOGY", "HINDI", "SOCIAL SCIENCE"]

# # def extract_marks_for_student(row):
# #     """Extract marks from row in your exact Excel format with multi-level headers."""
# #     data = {
# #         'Periodic1': {},
# #         'MidTerm': {}
# #     }

# #     # Helper function to safely get value by column tuple
# #     def safe_get_value(col_tuple):
# #         try:
# #             return row[col_tuple]
# #         except (KeyError, TypeError):
# #             return None

# #     # Extract Periodic Test 1 marks (out of 50)
# #     for subj in SUBJECTS:
# #         total_col = (subj, 'Total(50)')
# #         mark = safe_get_value(total_col)
# #         if mark is not None and pd.notna(mark):
# #             try:
# #                 data['Periodic1'][subj] = float(mark)
# #             except ValueError:
# #                 data['Periodic1'][subj] = 0.0

# #     # Extract Mid Term marks (out of 100)
# #     for subj in SUBJECTS:
# #         total_col = (subj, 'Total(100)')
# #         mark = safe_get_value(total_col)
# #         if mark is not None and pd.notna(mark):
# #             try:
# #                 data['MidTerm'][subj] = float(mark)
# #             except ValueError:
# #                 data['MidTerm'][subj] = 0.0

# #     return data

# # def plot_student_analysis(student_name, marks_dict):
# #     """Generate plots and insights for a single student."""
# #     # Create DataFrame for plotting
# #     records = []
# #     for exam, marks in marks_dict.items():
# #         for subj, mark in marks.items():
# #             # Convert to percentage (Periodic: out of 50, MidTerm: out of 100)
# #             mark = float(mark) if pd.notna(mark) else 0.0
# #             percentage = mark if exam == 'MidTerm' else (mark / 50) * 100
# #             records.append({
# #                 'Exam': exam,
# #                 'Subject': subj,
# #                 'Marks': mark,
# #                 'Percentage': percentage
# #             })

# #     df = pd.DataFrame(records)
# #     exams = df['Exam'].unique()

# #     if df.empty:
# #         st.warning(f"No valid marks data found for {student_name}.")
# #         return

# #     # === PLOT 1: Bar Chart (Raw Marks) ===
# #     fig1, ax1 = plt.subplots(figsize=(10, 5))
# #     width = 0.35
# #     indices = np.arange(len(SUBJECTS))

# #     for i, exam in enumerate(exams):
# #         exam_data = df[df['Exam'] == exam].set_index('Subject').reindex(SUBJECTS)['Marks']
# #         ax1.bar(indices + i * width, exam_data, width, label=exam)

# #     ax1.set_title(f"📈 Raw Marks: {student_name}", fontsize=16)
# #     ax1.set_xlabel("Subjects")
# #     ax1.set_ylabel("Marks Obtained")
# #     ax1.set_xticks(indices + width / 2)
# #     ax1.set_xticklabels(SUBJECTS, rotation=30)
# #     ax1.legend()
# #     ax1.grid(axis='y', linestyle='--', alpha=0.7)
# #     st.pyplot(fig1)
# #     plt.close(fig1)

# #     # === PLOT 2: Percentage Comparison ===
# #     fig2, ax2 = plt.subplots(figsize=(10, 5))
# #     for exam in exams:
# #         pct = df[df['Exam'] == exam].set_index('Subject').reindex(SUBJECTS)['Percentage']
# #         ax2.plot(SUBJECTS, pct, marker='o', label=exam, linewidth=2)

# #     ax2.set_title(f"📊 Percentage Comparison: {student_name}", fontsize=16)
# #     ax2.set_ylabel("Percentage (%)")
# #     ax2.set_ylim(0, 100)
# #     ax2.set_xticklabels(SUBJECTS, rotation=30)
# #     ax2.grid(True, linestyle='--', alpha=0.6)
# #     ax2.legend()
# #     st.pyplot(fig2)
# #     plt.close(fig2)

# #     # === INSIGHTS ===
# #     st.markdown('<div class="info-box">', unsafe_allow_html=True)
# #     st.subheader("🔍 Key Insights")
# #     col1, col2 = st.columns(2)

# #     with col1:
# #         # Exam-wise average percentage
# #         exam_avg = df.groupby('Exam')['Percentage'].mean().sort_values(ascending=False)
# #         st.write("**Exam-wise Avg %**")
# #         for exam, avg in exam_avg.items():
# #             emoji = "🟢" if avg >= 80 else "🟡" if avg >= 60 else "🔴"
# #             st.markdown(f"- {exam}: **{avg:.1f}%** {emoji}")

# #     with col2:
# #         # Subject-wise improvement
# #         if 'Periodic1' in exams and 'MidTerm' in exams:
# #             p1_pct = df[df['Exam']=='Periodic1']['Percentage'].mean()
# #             mt_pct = df[df['Exam']=='MidTerm']['Percentage'].mean()
# #             diff = mt_pct - p1_pct
# #             trend = "📈 Improved" if diff > 0 else "📉 Declined"
# #             st.write("**Improvement**")
# #             st.markdown(f"{trend} by {diff:+.1f}% on average")

# #     # Best & weakest subjects
# #     subject_avg = df.groupby('Subject')['Percentage'].mean().sort_values(ascending=False)
# #     best = subject_avg.idxmax()
# #     worst = subject_avg.idxmin()
# #     st.info(f"✅ Best Subject: **{best}** ({subject_avg[best]:.1f}%) | ❗ Weakest: **{worst}** ({subject_avg[worst]:.1f}%)")
# #     st.markdown('</div>', unsafe_allow_html=True)

# # # === CLASS LEVEL ANALYSIS ===
# # def plot_class_analysis(df):
# #     """Generate class-level insights."""
# #     st.header("📈 Class-Level Analysis")

# #     # Get all students' marks
# #     class_records = []
# #     for _, row in df.iterrows():
# #         marks = extract_marks_for_student(row)
# #         for exam, subj_marks in marks.items():
# #             for subj, mark in subj_marks.items():
# #                 mark = float(mark) if pd.notna(mark) else 0.0
# #                 percentage = mark if exam == 'MidTerm' else (mark / 50) * 100
# #                 class_records.append({
# #                     'Exam': exam,
# #                     'Subject': subj,
# #                     'Percentage': percentage
# #                 })

# #     class_df = pd.DataFrame(class_records)

# #     if class_df.empty:
# #         st.warning("No valid marks data found for the class.")
# #         return

# #     # === PLOT 1: Subject-wise Average Percentage ===
# #     fig1, ax1 = plt.subplots(figsize=(10, 5))
# #     subject_avg = class_df.groupby(['Subject', 'Exam'])['Percentage'].mean().unstack()
# #     if not subject_avg.empty:
# #         subject_avg.plot(kind='bar', ax=ax1, color=['skyblue', 'lightgreen'])
# #         ax1.set_title("Average Percentage by Subject (Periodic vs Mid Term)")
# #         ax1.set_ylabel("Average Percentage (%)")
# #         ax1.set_ylim(0, 100)
# #         ax1.grid(axis='y', linestyle='--', alpha=0.7)
# #         ax1.legend(title='Exam')
# #         st.pyplot(fig1)
# #         plt.close(fig1)
# #     else:
# #         st.warning("No data to plot for Subject-wise Average Percentage.")

# #     # === PLOT 2: Subject-wise Performance Distribution ===
# #     fig2, ax2 = plt.subplots(figsize=(10, 5))
# #     for subj in SUBJECTS:
# #         subj_data = class_df[class_df['Subject'] == subj]
# #         if not subj_data.empty:
# #             ax2.hist(subj_data['Percentage'], bins=10, alpha=0.5, label=subj)

# #     ax2.set_title("Performance Distribution by Subject")
# #     ax2.set_xlabel("Percentage (%)")
# #     ax2.set_ylabel("Number of Students")
# #     ax2.set_xlim(0, 100)
# #     ax2.legend()
# #     ax2.grid(True, linestyle='--', alpha=0.6)
# #     st.pyplot(fig2)
# #     plt.close(fig2)

# #     # === INSIGHTS ===
# #     st.markdown('<div class="info-box">', unsafe_allow_html=True)
# #     st.subheader("🔍 Class Insights")

# #     # Worst performing subject
# #     subject_avg_all = class_df.groupby('Subject')['Percentage'].mean().sort_values()
# #     if not subject_avg_all.empty:
# #         worst_subject = subject_avg_all.idxmin()
# #         st.warning(f"❗ **Worst Performing Subject**: {worst_subject} (Avg: {subject_avg_all[worst_subject]:.1f}%)")

# #     # Most improved subject
# #     if 'Periodic1' in class_df['Exam'].unique() and 'MidTerm' in class_df['Exam'].unique():
# #         p1_avg = class_df[class_df['Exam'] == 'Periodic1'].groupby('Subject')['Percentage'].mean()
# #         mt_avg = class_df[class_df['Exam'] == 'MidTerm'].groupby('Subject')['Percentage'].mean()
# #         if not p1_avg.empty and not mt_avg.empty:
# #             improvement = mt_avg - p1_avg
# #             if not improvement.empty:
# #                 most_improved = improvement.idxmax()
# #                 st.success(f"📈 **Most Improved Subject**: {most_improved} (+{improvement[most_improved]:.1f}%)")

# #     # Grade distribution (only for Mid Term as an example)
# #     st.write("**Grade Distribution (Mid Term)**")
# #     mid_term_grades = []
# #     for subj in SUBJECTS:
# #         grade_col = (subj, 'Grade')  # This is now a tuple!
# #         if grade_col in df.columns:
# #             # Filter out NaN grades
# #             valid_grades = df[grade_col].dropna()
# #             mid_term_grades.extend(valid_grades.tolist())

# #     if mid_term_grades:
# #         grade_counts = pd.Series(mid_term_grades).value_counts().sort_index()
# #         if not grade_counts.empty:
# #             st.bar_chart(grade_counts)
# #         else:
# #             st.info("No grade data available for Mid Term.")
# #     else:
# #         st.info("No grade columns found in the uploaded file.")
# #     st.markdown('</div>', unsafe_allow_html=True)

# # # === STREAMLIT UI ===
# # st.markdown('<h1 class="main-header">🎓 Student Performance Analysis Dashboard</h1>', unsafe_allow_html=True)
# # st.markdown('<p class="sub-header">Upload your Excel file with student marks to get instant visual insights!</p>', unsafe_allow_html=True)

# # uploaded_file = st.file_uploader("📤 Upload Excel File (.xlsx)", type=["xlsx"])

# # if uploaded_file is not None:
# #     try:
# #         # Read Excel with multi-level headers
# #         # Skip first row (school name), use row 1 and 2 as headers
# #         df = pd.read_excel(uploaded_file, header=[1, 2])  # Use second and third row as headers
# #         st.success(f"✅ Loaded {len(df)} students!")

# #         # Display first few rows for debugging
# #         st.write("✅ First few rows of your Excel:")
# #         st.dataframe(df.head())

# #         # --- FIX: Find the Student Name column ---
# #         # The first column (index 0) has ("Student Name", "") as its multi-level header
# #         # So we can directly use df.columns[0]
# #         student_col = df.columns[0]

# #         # Verify that the first level header is "Student Name"
# #         if student_col[0] != "Student Name":
# #             # Fallback: search for any column where first level contains "Student Name"
# #             student_col = None
# #             for col in df.columns:
# #                 if "Student" in str(col[0]) and "Name" in str(col[0]):
# #                     student_col = col
# #                     break

# #         if student_col is None:
# #             st.error("❌ Column containing 'Student Name' not found. Available columns:")
# #             st.write([col for col in df.columns])
# #             st.stop()

# #         st.success(f"✅ Found student column: '{student_col}'")

# #         # Student selector
# #         student_list = df[student_col].dropna().tolist()
# #         if len(student_list) == 0:
# #             st.error("❌ No valid student names found.")
# #             st.stop()

# #         selected_student = st.selectbox("👤 Select Student", student_list)

# #         # Get student row
# #         student_row = df[df[student_col] == selected_student].iloc[0]
# #         st.session_state.student_row = student_row

# #         marks_data = extract_marks_for_student(student_row)

# #         if marks_data:
# #             st.header(f"📊 Analysis for: **{selected_student}**")
# #             plot_student_analysis(selected_student, marks_data)
# #         else:
# #             st.warning("⚠️ No valid exam data found for this student.")

# #         # Class level analysis
# #         plot_class_analysis(df)

# #     except Exception as e:
# #         st.error(f"❌ Error: {e}")
# # else:
# #     st.info("👆 Please upload an Excel file in the format: `Student Name`, `ENGLISH Total(50)`, `ENGLISH Total(100)`, etc.")

# # st.markdown("---")
# # st.caption("💡 This dashboard works with your exact format: Periodic Test (out of 50), Mid Term (out of 100)")





# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from io import BytesIO
# import re
# import plotly.express as px

# # ======================
# # Page Config
# # ======================
# st.set_page_config(
#     page_title="📊 Student Performance Dashboard – Macro Vision Academy",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="🎓"
# )

# # ======================
# # Custom CSS
# # ======================
# st.markdown(
#     """
#     <style>
#     .main-header { font-size: 2.5rem; color: #1a3d6d; text-align: center; margin-bottom: 0.5rem; }
#     .section-header { font-size: 1.8rem; color: #2c3e50; margin-top: 2rem; }
#     .insight-box { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
#     .branding { text-align: center; color: #4a5568; font-size: 1.1rem; margin-bottom: 1.5rem; }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # ======================
# # Helper Functions
# # ======================

# def find_student_column(columns):
#     for col in columns:
#         if 'student' in str(col[0]).lower() and 'name' in str(col[0]).lower():
#             return col
#     return None

# def extract_subjects_from_columns(columns):
#     subjects = set()
#     for col in columns:
#         subj, metric = col[0], col[1]
#         if subj != "Student Name" and isinstance(subj, str) and metric in ["Total(50)", "Total(100)"]:
#             subjects.add(subj)
#     return sorted(list(subjects))

# def safe_float(x):
#     try:
#         return float(x) if pd.notna(x) else np.nan
#     except:
#         return np.nan

# def assign_grade(percentage):
#     """Assign grade based on percentage"""
#     if pd.isna(percentage):
#         return None
#     if percentage >= 91:
#         return 'A1'
#     elif percentage >= 81:
#         return 'A2'
#     elif percentage >= 71:
#         return 'B1'
#     elif percentage >= 61:
#         return 'B2'
#     elif percentage >= 51:
#         return 'C1'
#     elif percentage >= 41:
#         return 'C2'
#     elif percentage >= 33:
#         return 'D'
#     else:
#         return 'E'

# def preprocess_data(df, subjects):
#     student_col = find_student_column(df.columns)
#     if student_col is None:
#         raise ValueError("Student Name column not found")

#     records = []
#     for _, row in df.iterrows():
#         student_name = row[student_col]
#         for subj in subjects:
#             pt_val = safe_float(row.get((subj, 'Total(50)'), None))
#             if not np.isnan(pt_val):
#                 records.append({
#                     'Student': student_name,
#                     'Subject': subj,
#                     'Exam': 'Periodic1',
#                     'RawMark': pt_val,
#                     'Percentage': (pt_val / 50) * 100
#                 })
#             mt_val = safe_float(row.get((subj, 'Total(100)'), None))
#             if not np.isnan(mt_val):
#                 records.append({
#                     'Student': student_name,
#                     'Subject': subj,
#                     'Exam': 'MidTerm',
#                     'RawMark': mt_val,
#                     'Percentage': mt_val
#                 })
#     return pd.DataFrame(records)

# def parse_sheet_name(sheet_name):
#     parts = str(sheet_name).strip().split(maxsplit=1)
#     if len(parts) == 1:
#         return parts[0], ""
#     else:
#         return parts[0], parts[1]

# # ======================
# # Sidebar Controls
# # ======================

# # ✅ LOGO IN SIDEBAR (LEFT SLIDER)
# try:
#     st.sidebar.image("logo.png", use_container_width=True)
# except:
#     st.sidebar.image("https://via.placeholder.com/150x50?text=Macro+Vision", use_container_width=True)

# st.sidebar.title(" Dashboard Controls")

# uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

# # Centered logo in main area ONLY when no file or before selection
# if uploaded_file is None:
#     col1, col2, col3 = st.columns([1, 1, 1])
#     with col2:
#         try:
#             st.image("logo.png", width=300)
#         except:
#             st.image("https://via.placeholder.com/200x60?text=MVA", width=200)
#     st.markdown('<h1 class="main-header"> Student Performance Dashboard</h1>', unsafe_allow_html=True)
#     st.markdown('<h3><p style="text-align:center; font-weight:bold;">Macro Vision Academy, Burhanpur</h3></p>', unsafe_allow_html=True)
#     st.info("📤 Please upload a Result Excel file in the sidebar to begin.")
#     st.stop()

# # Read sheets
# try:
#     xls = pd.ExcelFile(uploaded_file)
#     all_sheet_names = xls.sheet_names
# except Exception as e:
#     st.error(f"Error reading Excel file: {e}")
#     st.stop()

# sheet_info = []
# for name in all_sheet_names:
#     cls, sec = parse_sheet_name(name)
#     sheet_info.append({'SheetName': name, 'Class': cls, 'Section': sec})

# sheet_df = pd.DataFrame(sheet_info)

# classes = sorted(sheet_df['Class'].unique())

# # ✅ CRITICAL FIX: Add empty option for Class
# selected_class = st.sidebar.selectbox(
#     "🏫 Select Class",
#     [""] + classes,
#     format_func=lambda x: "Select Class..." if x == "" else x
# )

# selected_section = ""
# if selected_class:
#     sections_in_class = sheet_df[sheet_df['Class'] == selected_class]['Section'].dropna().astype(str).tolist()
#     sections_in_class = [s for s in sections_in_class if s.strip() != ""]
#     if not sections_in_class:
#         st.sidebar.warning(f"No sections found for class {selected_class}")
#     else:
#         # ✅ CRITICAL FIX: Add empty option for Section
#         selected_section = st.sidebar.selectbox(
#             "🏫 Select Section",
#             [""] + sections_in_class,
#             format_func=lambda x: "Select Section..." if x == "" else x
#         )

# # ✅ STOP if class or section not selected
# if not selected_class or not selected_section:
    
#     col1, col2, col3 = st.columns([1, 1, 1])
#     with col2:
#         try:
#             st.image("logo.png", width=300)
#         except:
#             st.image("https://via.placeholder.com/200x60?text=MVA", width=200)
#     st.markdown('<h1 class="main-header"> Student Performance Dashboard</h1>', unsafe_allow_html=True)
#     st.markdown('<h3><p style="text-align:center; font-weight:bold;">Macro Vision Academy, Burhanpur</h3></p>', unsafe_allow_html=True)
#     st.info("✅ Please select a **Class** and **Section** from the sidebar to begin analysis.")
#     st.stop()

# # Find matching sheet
# matching_row = sheet_df[
#     (sheet_df['Class'] == selected_class) &
#     (sheet_df['Section'] == selected_section)
# ]
# if matching_row.empty:
#     st.error("Could not find matching sheet.")
#     st.stop()

# selected_sheet = matching_row.iloc[0]['SheetName']

# # Now allow comparison mode (since class & section are selected)
# comparison_mode = st.sidebar.radio(
#     "📊 Comparison Mode",
#     ["PT-1 vs Mid-Term", "PT-1 only", "Mid-Term only"]
# )

# # ======================
# # Load Data
# # ======================
# try:
#     raw_df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, header=None)
#     header_row = None
#     for i in range(min(5, len(raw_df))):
#         if any("Total(50)" in str(c) or "Total(100)" in str(c) for c in raw_df.iloc[i]):
#             header_row = i - 1
#             break
#     if header_row is None:
#         st.error("Header detection failed in selected sheet.")
#         st.stop()

#     df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, header=[header_row, header_row+1])
#     SUBJECTS = extract_subjects_from_columns(df.columns)
#     if not SUBJECTS:
#         st.error("No subjects found in the selected sheet.")
#         st.stop()

#     long_df = preprocess_data(df, SUBJECTS)
#     student_col = find_student_column(df.columns)
#     if student_col is None:
#         st.error("Student Name column not found.")
#         st.stop()
#     student_list = df[student_col].dropna().astype(str).tolist()
#     student_list = [s for s in student_list if s != 'nan' and s.strip() != ""]

#     if comparison_mode == "PT-1 only":
#         filtered_df = long_df[long_df['Exam'] == 'Periodic1']
#     elif comparison_mode == "Mid-Term only":
#         filtered_df = long_df[long_df['Exam'] == 'MidTerm']
#     else:
#         filtered_df = long_df

#     if comparison_mode == "PT-1 vs Mid-Term":
#         selected_subjects = st.sidebar.multiselect("📚 Select Subjects", SUBJECTS, default=SUBJECTS)
#         filtered_df = filtered_df[filtered_df['Subject'].isin(selected_subjects)]
#         available_subjects = selected_subjects
#     else:
#         selected_subject = st.sidebar.selectbox("📚 Select Subject", SUBJECTS)
#         filtered_df = filtered_df[filtered_df['Subject'] == selected_subject]
#         available_subjects = [selected_subject]

#     if not student_list:
#         st.warning("No students found.")
#         st.stop()
#     selected_student = st.sidebar.selectbox("👤 Select Student", student_list)

# except Exception as e:
#     st.error(f"Error loading data from sheet '{selected_sheet}': {e}")
#     st.stop()

# # ======================
# # Unified Header with Centered Logo (Always Visible)
# # ======================

# # Centered layout for logo + text + exam dates
# col1, col2, col3 = st.columns([1, 1, 1])  # Wider center column for logo

# with col2:
#     # Centered Logo (medium size)
#     try:
#         st.image("logo.png", width=300)  # Adjust width as needed (e.g., 200–250)
#     except:
#         st.image("https://via.placeholder.com/220x60?text=Macro+Vision", width=300)

# # Title and Subtitle (centered below logo)
# st.markdown(
#     """
#     <div style="text-align: center; margin-top: 10px;">
#         <h1 style="color: black; margin: 0;">Student Performance Dashboard</h1>
#         <h3 style="color: black; margin: 8px 0 4px;">Macro Vision Academy, Burhanpur</h3>
#     </div>
#     """,
#     unsafe_allow_html=True
# )
# # ===========================================================================
# # 1️⃣ Class Overview
# # ===========================================================================
# st.markdown('<div class="section-header">1️⃣ Class Overview</div>', unsafe_allow_html=True)

# class_avg = filtered_df['Percentage'].mean()
# class_max = filtered_df['Percentage'].max()
# class_min = filtered_df['Percentage'].min()

# col1, col2, col3 = st.columns(3)
# col1.metric("📊 Average Score", f"{class_avg:.1f}%")
# col2.metric("🔝 Highest Score", f"{class_max:.1f}%")
# col3.metric("🔻 Lowest Score", f"{class_min:.1f}%")

# # Grade Distribution
# st.subheader("🏅 Grade Distribution")

# VALID_GRADES = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D', 'E']

# filtered_df['Grade'] = filtered_df['Percentage'].apply(assign_grade)
# grade_data = filtered_df['Grade'].dropna().tolist()
# grade_data = [g for g in grade_data if g in VALID_GRADES]

# if grade_data:
#     grades_series = pd.Series(grade_data)
#     grade_counts = grades_series.value_counts().reindex(VALID_GRADES, fill_value=0)
#     grade_counts = grade_counts[grade_counts > 0]

#     if not grade_counts.empty:
#         pie_data = pd.DataFrame({
#             'Grade': grade_counts.index,
#             'Count': grade_counts.values
#         })

#         colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

#         fig = px.pie(
#             pie_data,
#             names='Grade',
#             values='Count',
#             color='Grade',
#             color_discrete_sequence=colors,
#             title="",
#             hole=0.3
#         )

#         fig.update_traces(
#             textposition='inside',
#             textinfo='percent+label',
#             textfont_size=30,
#             textfont_color='white',
#             hovertemplate="<b>%{label}</b><br>Students: %{value}<extra></extra>",
#             marker=dict(line=dict(color='#FFFFFF', width=1))
#         )

#         fig.update_layout(
#             showlegend=False,
#             margin=dict(t=0, b=0, l=0, r=0),
#             font=dict(size=12),
#             width=800,
#             height=700,
#             transition={'duration': 800, 'easing': 'cubic-in-out'},
#             hoverlabel=dict(
#                 bgcolor="white",
#                 font_size=20,
#                 font_family="Arial",
#                 font_color="black"
#             )
#         )

#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info(f"No valid grades found. Valid grades: {', '.join(VALID_GRADES)}")
# else:
#     st.info("No grade data available.")

# st.markdown("<br>", unsafe_allow_html=True)

# # Top 5 Performers
# st.subheader("🏆 Top 5 Performers")
# if comparison_mode == "PT-1 vs Mid-Term":
#     student_avg = long_df.groupby('Student')['Percentage'].mean().sort_values(ascending=False).head(5)
# else:
#     student_avg = filtered_df.groupby('Student')['Percentage'].mean().sort_values(ascending=False).head(5)
# for i, (student, avg) in enumerate(student_avg.items(), 1):
#     st.write(f"{i}. **{student}** – {avg:.1f}%")

# st.markdown("<br>", unsafe_allow_html=True)

# # Subject Ranking
# st.subheader("📚 Subject Ranking (by Class Avg %)")
# subject_ranking = filtered_df.groupby('Subject')['Percentage'].mean().sort_values(ascending=False)

# if not subject_ranking.empty:
#     rank_df = subject_ranking.reset_index()
#     rank_df.columns = ['Subject', 'Average %']

#     color_sequence = px.colors.qualitative.Bold[:len(rank_df)]

#     fig = px.bar(
#         rank_df,
#         x='Subject',
#         y='Average %',
#         color='Subject',
#         color_discrete_sequence=color_sequence,
#         text='Average %'
#     )

#     fig.update_traces(
#         texttemplate='%{text:.1f}%',
#         textposition='outside',
#         textfont_size=20,
#         textfont_color='black',
#         hovertemplate="<b>%{x}</b><br>Average: <b>%{y:.1f}%</b><extra></extra>"
#     )

#     fig.update_layout(
#         showlegend=False,
#         xaxis_title="Subjects",
#         yaxis_title="Average Percentage (%)",
#         yaxis=dict(range=[0, 100]),
#         font=dict(size=20),
#         hoverlabel=dict(
#             bgcolor="white",
#             font_size=14,
#             font_family="Arial",
#             font_color="black"
#         ),
#         margin=dict(t=20, b=40, l=40, r=20)
#     )

#     st.plotly_chart(fig, use_container_width=True)
# else:
#     st.info("No data available for subject ranking.")

# st.markdown("<br>", unsafe_allow_html=True)

# # ===========================================================================
# # 2️⃣ Class Subject Comparison (PT-1 vs Mid-Term)
# # ===========================================================================
# if comparison_mode == "PT-1 vs Mid-Term":
#     st.markdown('<div class="section-header">2️⃣ Class: Subject Comparison (PT-1 vs Mid-Term)</div>', unsafe_allow_html=True)

#     pt_class = long_df[long_df['Exam'] == 'Periodic1'].groupby('Subject')['Percentage'].mean()
#     mt_class = long_df[long_df['Exam'] == 'MidTerm'].groupby('Subject')['Percentage'].mean()
#     all_subjects = sorted(set(pt_class.index) | set(mt_class.index))
#     pt_vals = [pt_class.get(s, 0) for s in all_subjects]
#     mt_vals = [mt_class.get(s, 0) for s in all_subjects]

#     fig, ax = plt.subplots(figsize=(10, 5))
#     indices = np.arange(len(all_subjects))
#     width = 0.35
#     bars1 = ax.bar(indices - width/2, pt_vals, width, label='Periodic1', color='#1f77b4')
#     bars2 = ax.bar(indices + width/2, mt_vals, width, label='MidTerm', color='#ff7f0e')

#     for bar in bars1:
#         height = bar.get_height()
#         ax.annotate(f'{height:.1f}%',
#                     xy=(bar.get_x() + bar.get_width() / 1, height),
#                     xytext=(0, 3), textcoords="offset points",
#                     ha='center', va='bottom', fontsize=9)
#     for bar in bars2:
#         height = bar.get_height()
#         ax.annotate(f'{height:.1f}%',
#                     xy=(bar.get_x() + bar.get_width() / 1, height),
#                     xytext=(0, 3), textcoords="offset points",
#                     ha='center', va='bottom', fontsize=9)

#     ax.set_xlabel("Subjects")
#     ax.set_ylabel("Average Percentage (%)")
#     ax.set_title("Class Average: Periodic1 vs MidTerm")
#     ax.set_xticks(indices)
#     ax.set_xticklabels(all_subjects, rotation=30, ha='right')
#     ax.set_ylim(0, 100)
#     ax.legend()
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     st.pyplot(fig)
#     plt.close(fig)

#     st.subheader("Insights")
#     for subj in all_subjects:
#         pt = pt_class.get(subj, 0)
#         mt = mt_class.get(subj, 0)
#         diff = mt - pt
#         if diff >= 0:
#             st.markdown(f"- **{subj}**: Improved by **{diff:.1f}%**")
#         else:
#             st.markdown(f"- **{subj}**: Declined by **{abs(diff):.1f}%**")

#     st.markdown("<br>", unsafe_allow_html=True)

# # ===========================================================================
# # 4️⃣ Student Explorer
# # ===========================================================================
# st.markdown('<div class="section-header">4️⃣ Student Performance Explorer</div>', unsafe_allow_html=True)

# student_data = long_df[long_df['Student'] == selected_student]
# if student_data.empty:
#     st.warning(f"No data for {selected_student}")
# else:
# # ======================
# # STUDENT SUBJECT COMPARISON BAR CHART (PT-1 vs MidTerm) — FIXED LABEL OVERLAP
# # ======================
#     # ======================
# # STUDENT SUBJECT COMPARISON BAR CHART — FINAL FIX (NO OVERLAP)
# # ======================
# # ======================
# # STUDENT SUBJECT COMPARISON BAR CHART — FINAL FIX (NO OVERLAP)
# # ======================
#     if comparison_mode == "PT-1 vs Mid-Term":
#         st.subheader(f"📊 {selected_student}: Subject Comparison (PT-1 vs Mid-Term)")

#         # Get student's scores
#         pt_scores = student_data[student_data['Exam']=='Periodic1'].set_index('Subject')['Percentage']
#         mt_scores = student_data[student_data['Exam']=='MidTerm'].set_index('Subject')['Percentage']
#         all_subjects_s = sorted(set(pt_scores.index) | set(mt_scores.index))

#         # Prepare data
#         pt_vals_s = [pt_scores.get(s, 0) for s in all_subjects_s]
#         mt_vals_s = [mt_scores.get(s, 0) for s in all_subjects_s]

#         # Set up positions and width
#         indices = np.arange(len(all_subjects_s))
#         width = 0.25  # Narrower bars

#         fig, ax = plt.subplots(figsize=(14, 7))  # 👈 Increased height for more space

#         # Plot bars
#         bars1 = ax.bar(indices - width/2, pt_vals_s, width, label='Periodic1', color='#1f77b4', edgecolor='white', linewidth=1.5)
#         bars2 = ax.bar(indices + width/2, mt_vals_s, width, label='MidTerm', color='#ff7f0e', edgecolor='white', linewidth=1.5)

#         # Add labels on top of bars — with clip_on=False to allow outside plot area
#         for bar in bars1:
#             height = bar.get_height()
#             ax.annotate(f'{height:.1f}%',
#                         xy=(bar.get_x() + bar.get_width() / 2, height),
#                         xytext=(0, 12),  # Push label 12 points above bar
#                         textcoords="offset points",
#                         ha='center', va='bottom',
#                         fontsize=10, fontweight='bold', color='black',
#                         clip_on=False)  # 👈 CRITICAL: Allow label to appear outside plot area

#         for bar in bars2:
#             height = bar.get_height()
#             ax.annotate(f'{height:.1f}%',
#                         xy=(bar.get_x() + bar.get_width() / 2, height),
#                         xytext=(0, 12),  # Same offset
#                         textcoords="offset points",
#                         ha='center', va='bottom',
#                         fontsize=10, fontweight='bold', color='black',
#                         clip_on=False)  # 👈 CRITICAL

#         # Customize axes
#         ax.set_xlabel("Subjects", fontsize=12, fontweight='bold')
#         ax.set_ylabel("Percentage (%)", fontsize=12, fontweight='bold')
#         ax.set_title(f"{selected_student}: Periodic1 vs MidTerm", fontsize=14, fontweight='bold')
#         ax.set_xticks(indices)
#         ax.set_xticklabels(all_subjects_s, rotation=45, ha='right', fontsize=11, fontweight='medium')
#         ax.set_ylim(0, 100)  # Keep max at 100% — bars don't go beyond
#         ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1), fontsize=12)
#         ax.grid(axis='y', linestyle='--', alpha=0.6)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#         # Tight layout to avoid clipping
#         plt.tight_layout()

#         st.pyplot(fig)
#         plt.close(fig)

#     if comparison_mode == "PT-1 vs Mid-Term":
#         pt_row = student_data[student_data['Exam']=='Periodic1'][['Subject','RawMark']].set_index('Subject')['RawMark']
#         mt_row = student_data[student_data['Exam']=='MidTerm'][['Subject','RawMark']].set_index('Subject')['RawMark']
#         comp_df = pd.DataFrame({
#             'Periodic1 (50)': pt_row,
#             'MidTerm (100)': mt_row,
#             'PT1 %': [f"{v/50*100:.1f}%" for v in pt_row],
#             'MidTerm %': [f"{v:.1f}%" for v in mt_row]
#         }).reindex(all_subjects_s)
#         st.dataframe(comp_df)

#     if comparison_mode == "PT-1 vs Mid-Term":
#         st.subheader(f"📊 {selected_student}: Subject Comparison (PT-1 vs Mid-Term)")
#         all_subjects_s = sorted(set(pt_scores.index) | set(mt_scores.index))
#         pt_vals_s = [pt_scores.get(s, 0) for s in all_subjects_s]
#         mt_vals_s = [mt_scores.get(s, 0) for s in all_subjects_s]
#         indices2 = np.arange(len(all_subjects_s))
#         width = 0.35

#         fig2, ax2 = plt.subplots(figsize=(10, 5))
#         bars1 = ax2.bar(indices2 - width/2, pt_vals_s, width, label='Periodic1', color='#1f77b4')
#         bars2 = ax2.bar(indices2 + width/2, mt_vals_s, width, label='MidTerm', color='#ff7f0e')

#         for bar in bars1:
#             height = bar.get_height()
#             ax2.annotate(f'{height:.1f}%',
#                          xy=(bar.get_x() + bar.get_width() / 2, height),
#                          xytext=(0, 3), textcoords="offset points",
#                          ha='center', va='bottom', fontsize=9)
#         for bar in bars2:
#             height = bar.get_height()
#             ax2.annotate(f'{height:.1f}%',
#                          xy=(bar.get_x() + bar.get_width() / 2, height),
#                          xytext=(0, 3), textcoords="offset points",
#                          ha='center', va='bottom', fontsize=9)

#         ax2.set_xlabel("Subjects")
#         ax2.set_ylabel("Percentage (%)")
#         ax2.set_title(f"{selected_student}: Periodic1 vs MidTerm")
#         ax2.set_xticks(indices2)
#         ax2.set_xticklabels(all_subjects_s, rotation=30, ha='right')
#         ax2.set_ylim(0, 100)
#         ax2.legend()
#         ax2.grid(axis='y', linestyle='--', alpha=0.7)
#         st.pyplot(fig2)
#         plt.close(fig2)

#         st.subheader("Insights")
#         for subj in all_subjects_s:
#             pt = pt_scores.get(subj, 0)
#             mt = mt_scores.get(subj, 0)
#             diff = mt - pt
#             if diff >= 0:
#                 st.markdown(f"- **{subj}**: Improved by **{diff:.1f}%**")
#             else:
#                 st.markdown(f"- **{subj}**: Declined by **{abs(diff):.1f}%**")

#     overall_avg = student_data['Percentage'].mean()
#     best_subj = student_data.loc[student_data['Percentage'].idxmax()]['Subject']
#     worst_subj = student_data.loc[student_data['Percentage'].idxmin()]['Subject']
#     insights = [
#         f"✅ **Strongest Subject**: {best_subj}",
#         f"❗ **Needs Improvement**: {worst_subj}"
#     ]
#     if comparison_mode == "PT-1 vs Mid-Term":
#         pt_avg = student_data[student_data['Exam']=='Periodic1']['Percentage'].mean()
#         mt_avg = student_data[student_data['Exam']=='MidTerm']['Percentage'].mean()
#         if not np.isnan(pt_avg) and not np.isnan(mt_avg):
#             diff = mt_avg - pt_avg
#             insights.append(f"📈 **Overall Improvement**: {diff:+.1f}%")
#     st.markdown('<div class="insight-box">' + "<br>".join(insights) + '</div>', unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)

# # ===========================================================================
# # 5️⃣ Correlation Heatmap
# # ===========================================================================
# if comparison_mode == "PT-1 vs Mid-Term":
#     st.markdown('<div class="section-header">5️⃣ Subject Correlation</div>', unsafe_allow_html=True)
#     pivot = long_df[long_df['Exam']=='MidTerm'].pivot(index='Student', columns='Subject', values='Percentage')
#     corr = pivot.corr()
#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
#     ax.set_title("Subject Correlation Matrix", fontsize=14)
#     st.pyplot(fig)
#     plt.close(fig)

#     corr_pairs = []
#     for i in range(len(corr.columns)):
#         for j in range(i+1, len(corr.columns)):
#             sub1, sub2 = corr.columns[i], corr.columns[j]
#             val = corr.iloc[i, j]
#             if abs(val) > 0.6:
#                 direction = "positively" if val > 0 else "negatively"
#                 corr_pairs.append(f"**{sub1}** and **{sub2}** are {direction} correlated (r={val:.2f})")
#     if corr_pairs:
#         st.markdown('<div class="insight-box">' + "<br>".join(corr_pairs) + '</div>', unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)

# # ===========================================================================
# # 6️⃣ Downloads
# # ===========================================================================
# st.markdown('<div class="section-header">6️⃣ Export Reports</div>', unsafe_allow_html=True)

# class_summary = long_df.groupby(['Student', 'Exam'])['Percentage'].mean().unstack(fill_value=0)
# csv = class_summary.to_csv().encode('utf-8')
# st.download_button("📥 Download Class Summary (CSV)", csv, "class_summary.csv", "text/csv")

# student_report = student_data[['Exam', 'Subject', 'RawMark', 'Percentage']]
# csv2 = student_report.to_csv(index=False).encode('utf-8')
# st.download_button("📥 Download Student Report (CSV)", csv2, f"{selected_student}_report.csv", "text/csv")

# st.markdown("<br>", unsafe_allow_html=True)

# # ===========================================================================
# # 7️⃣ Automated Insights Engine
# # ===========================================================================
# st.markdown('<div class="section-header">7️⃣ Automated Insights Engine</div>', unsafe_allow_html=True)

# insights_list = []
# top3 = filtered_df.groupby('Subject')['Percentage'].mean().nlargest(3)
# insights_list.append(f"✅ **Top Performing Subjects**: {', '.join(top3.index[:3])}")

# weakest = filtered_df.groupby('Subject')['Percentage'].mean().idxmin()
# insights_list.append(f"❗ **Weakest Subject**: {weakest} — Work needed here!")

# if comparison_mode == "PT-1 vs Mid-Term":
#     pt_per_student = long_df[long_df['Exam']=='Periodic1'].groupby('Student')['Percentage'].mean()
#     mt_per_student = long_df[long_df['Exam']=='MidTerm'].groupby('Student')['Percentage'].mean()
#     improvement = (mt_per_student - pt_per_student).dropna().sort_values(ascending=False)
#     if not improvement.empty:
#         most_improved = improvement.head(3).index.tolist()
#         insights_list.append(f"🚀 **Most Improved Students**: {', '.join(most_improved)}")
#         declined = improvement[improvement < 0].sort_values().head(3).index.tolist()
#         if declined:
#             insights_list.append(f"⚠️ **Performance Declined**: {', '.join(declined)}")

#     pt_class_avg = long_df[long_df['Exam']=='Periodic1']['Percentage'].mean()
#     mt_class_avg = long_df[long_df['Exam']=='MidTerm']['Percentage'].mean()
#     growth = mt_class_avg - pt_class_avg
#     insights_list.append(f"📈 **Class Academic Growth**: {growth:+.1f}%")

#     subj_improvement = {}
#     for subj in available_subjects:
#         subj_data = long_df[long_df['Subject'] == subj]
#         pt_avg = subj_data[subj_data['Exam']=='Periodic1']['Percentage'].mean()
#         mt_avg = subj_data[subj_data['Exam']=='MidTerm']['Percentage'].mean()
#         subj_improvement[subj] = mt_avg - pt_avg

#     best_imp = max(subj_improvement, key=subj_improvement.get)
#     worst_imp = min(subj_improvement, key=subj_improvement.get)
#     insights_list.append(f"🎯 **Biggest Improvement**: {best_imp} (+{subj_improvement[best_imp]:.1f}%)")
#     insights_list.append(f"⚠️ **Biggest Decline**: {worst_imp} ({subj_improvement[worst_imp]:+.1f}%)")

# for insight in insights_list:
#     st.markdown(f"- {insight}")

# st.markdown("---")
# st.caption("💡 Grade chart shows computed grades (A1–E) from percentages. Bar charts show exact % values.")






















# ======================
# FINAL FILE — DYNAMIC EXAM SUPPORT + ALL ORIGINAL FEATURES INTACT
# ======================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from io import BytesIO
import json
import os
import bcrypt
import requests
from tempfile import NamedTemporaryFile
from datetime import datetime, date
import calendar

# ======================
# GOOGLE SHEET CONFIG
# ======================
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQu-LpLCot1a5xibAAGfcnfSsqwtGFLJmraZUGWT_mutvswBjIgoDo7D4fsm-4lNW_KBxw1cyTWLO3l/pub?output=xlsx"

# ======================
# CONFIG & PATHS
# ======================
USERS_FILE = "users.json"
SESSION_FILE_PREFIX = "session_"
SESSION_TIMEOUT_HOURS = 24
ADMIN_USERNAME = "admin"
ADMIN_DEFAULT_PASSWORD = "MVA@123"
HOMEWORK_EXCEL_FILE = "homework_data.xlsx"
HOMEWORK_EXCEL_VALID = HOMEWORK_EXCEL_FILE 

# ======================
# EXAM MAPPING — USER-FRIENDLY NAMES
# ======================
EXAM_MAPPING = {
    50: "PT 1",
    40: "PT 2",
    100: "Mid Term",
    200: "Annual Exam"
}

def get_exam_name_from_total(total_str):
    """Convert 'Total(50)' → 'PT 1', 'Total(100)' → 'Mid Term', etc."""
    if not isinstance(total_str, str):
        return None
    if "total" in total_str.lower():
        try:
            marks = int(total_str.split("(")[1].split(")")[0])
            return EXAM_MAPPING.get(marks, f"Exam({marks})")
        except (IndexError, ValueError):
            return None
    return None

# ======================
# UNIVERSAL HELPER: Student Name Normalization
# ======================
def normalize_student_name(name):
    """Convert student name to lowercase and strip whitespace for consistent matching."""
    if pd.isna(name) or str(name).strip() == "":
        return ""
    return str(name).strip().lower()

# ======================
# SESSION MANAGEMENT
# ======================
def create_session_file(username, role):
    session_data = {
        "username": username,
        "role": role,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    with open(f"{SESSION_FILE_PREFIX}{username}.json", 'w') as f:
        json.dump(session_data, f)

def load_active_session():
    for file in os.listdir("."):
        if file.startswith(SESSION_FILE_PREFIX) and file.endswith(".json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                session_time = pd.Timestamp(data["timestamp"])
                if (pd.Timestamp.now() - session_time).total_seconds() < SESSION_TIMEOUT_HOURS * 3600:
                    return data["username"], data["role"]
                else:
                    os.remove(file)
            except Exception:
                try:
                    os.remove(file)
                except:
                    pass
    return None, None

def clear_session(username):
    try:
        os.remove(f"{SESSION_FILE_PREFIX}{username}.json")
    except:
        pass

# ======================
# User System
# ======================
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def load_users():
    if not os.path.exists(USERS_FILE):
        users = {
            ADMIN_USERNAME: {
                "password": hash_password(ADMIN_DEFAULT_PASSWORD),
                "role": "admin",
                "audit_log": []
            }
        }
        save_users(users)
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# ======================
# HOMEWORK HELPER FUNCTIONS (With Normalization)
# ======================
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
    entry_df["Student"] = entry_df["Student"].apply(normalize_student_name)
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
            try:
                month_name, year_str = sheet_name.split()
                month_num = list(calendar.month_abbr).index(month_name)
                year_val = int(year_str)
                existing_long["Date"] = existing_long["Day"].apply(
                    lambda d: date(year_val, month_num, int(d)) if str(d).isdigit() else None
                )
                existing_long = existing_long.dropna(subset=["Date"])
                existing_long["Student"] = existing_long["Student"].apply(normalize_student_name)
            except:
                existing_long = pd.DataFrame()
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
                parts = sheet.split()
                if len(parts) < 2:
                    continue
                month_name = parts[0]
                year_str = parts[1]
                month_num = list(calendar.month_abbr).index(month_name)
                year = int(year_str)
                melted["Day"] = pd.to_numeric(melted["Day"], errors='coerce')
                melted["Date"] = melted["Day"].apply(
                    lambda d: date(year, month_num, int(d)) if pd.notna(d) and d > 0 and d <= 31 else None
                )
                melted = melted.dropna(subset=["Date"])
                melted["Student"] = melted["Student"].apply(normalize_student_name)
                all_dfs.append(melted)
            except Exception as e:
                continue
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# ======================
# HOMEWORK UI FUNCTIONS (Unchanged)
# ======================
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
    existing_dict = {}
    existing_remark_dict = {}
    existing_df = load_homework_data()
    if not existing_df.empty:
        existing_df["Date"] = pd.to_datetime(existing_df["Date"], errors='coerce').dt.date
        target_date = selected_date
        filtered = existing_df[
            (existing_df["Class"].astype(str) == str(selected_class)) &
            (existing_df["Section"].astype(str) == str(selected_section)) &
            (existing_df["Date"] == target_date)
        ]
        for _, row in filtered.iterrows():
            student_name = normalize_student_name(str(row["Student"]))
            val = row["Pages Done"]
            if isinstance(val, str) and val.startswith("Remark: "):
                existing_dict[student_name] = 0
                existing_remark_dict[student_name] = val[8:].strip()
            else:
                try:
                    num_val = float(val)
                    if pd.notna(num_val):
                        existing_dict[student_name] = int(num_val)
                    else:
                        existing_dict[student_name] = 0
                except (ValueError, TypeError):
                    existing_dict[student_name] = 0
                existing_remark_dict[student_name] = ""

    initial_data = []
    for idx, student in enumerate(student_list, start=1):
        student_clean = normalize_student_name(str(student))
        pages = existing_dict.get(student_clean, 0)
        remark = existing_remark_dict.get(student_clean, "")
        initial_data.append({"S.No.": idx, "Student": student_clean, "Pages Done": pages, "Remark": remark})

    if not initial_data:
        st.info("ℹ️ No students to enter homework for.")
        return

    header_cols = st.columns([0.5, 3.0, 1.0, 2.2], gap="small")
    with header_cols[0]:
        st.markdown("**S.No.**")
    with header_cols[1]:
        st.markdown("**Student Name**")
    with header_cols[2]:
        st.markdown("**Homework Pages**")
    with header_cols[3]:
        st.markdown("**Remark**")

    for i, row in enumerate(initial_data):
        cols = st.columns([0.5, 3.0, 1.0, 2.2], gap="small")
        with cols[0]:
            st.markdown(f"<div style='display: flex; align-items: center; height: 100%;'><b>{row['S.No.']}. </b></div>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"<div style='display: flex; align-items: center; height: 100%;'><b>{row['Student']}</b></div>", unsafe_allow_html=True)
        with cols[2]:
            pages_val = int(row["Pages Done"]) if pd.notna(row["Pages Done"]) else 0
            pages_val = max(0, min(100, pages_val))
            new_pages = st.number_input(
                "",
                min_value=0,
                max_value=100,
                value=pages_val,
                step=1,
                key=f"pages_{i}_{selected_date.isoformat()}",
                label_visibility="collapsed"
            )
            initial_data[i]["Pages Done"] = new_pages
        with cols[3]:
            new_remark = st.text_input(
                "",
                value=row["Remark"],
                placeholder="Absent / Medical / Other...",
                key=f"remark_{i}_{selected_date.isoformat()}",
                label_visibility="collapsed"
            )
            initial_data[i]["Remark"] = new_remark

    if st.button("✅ Submit Homework Data"):
        df = pd.DataFrame(initial_data)
        df["Date"] = selected_date
        df["Class"] = str(selected_class)
        df["Section"] = str(selected_section)
        df["Entered By"] = st.session_state.get("username", "guruji")

        def process(row):
            p = row["Pages Done"]
            r = row["Remark"]
            if p == 0 and r.strip():
                return f"Remark: {r.strip()}"
            return int(p) if p > 0 else 0

        df["Pages Done"] = df.apply(process, axis=1)
        df["Student"] = df["Student"].apply(normalize_student_name)
        sheet_name = get_sheet_name_from_date(selected_date)
        save_homework_to_excel(df, sheet_name, selected_date)
        st.success("✅ Homework data saved successfully!")
        st.rerun()

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
    full_list["Student_norm"] = full_list["Student"].apply(normalize_student_name)
    daily_df_norm = daily_df.copy()
    daily_df_norm["Student_norm"] = daily_df_norm["Student"].apply(normalize_student_name)
    merged = full_list.merge(daily_df_norm[["Student_norm", "Pages Done"]], on="Student_norm", how="left")
    merged["Pages Done Display"] = merged["Pages Done"].apply(
        lambda x: x if isinstance(x, str) else int(x) if pd.notna(x) else 0
    )
    expected = get_expected_pages(selected_class)
    merged["Expected"] = expected
    merged["Pages Numeric"] = pd.to_numeric(merged["Pages Done"], errors='coerce').fillna(0)
    merged["Status"] = merged["Pages Numeric"].apply(
        lambda x: "✅ Done" if x >= expected else "⚠️ Incomplete"
    )
    st.subheader(f"📊 Homework Status for {selected_date.strftime('%d %B %Y')}")
    st.dataframe(
        merged[["Student", "Pages Done Display", "Expected", "Status"]],
        use_container_width=True,
        hide_index=True
    )
    completed = (merged["Pages Numeric"] >= expected).sum()
    total = len(merged)
    st.metric("Students Completed", f"{completed}/{total}", f"{(completed/total)*100:.1f}% on track")
    if os.path.exists(HOMEWORK_EXCEL_VALID):
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
    monthly_df["Numeric Pages"] = monthly_df["Pages Done"].apply(
        lambda x: x if isinstance(x, (int, float)) else 0
    )
    monthly_df["Student_norm"] = monthly_df["Student"].apply(normalize_student_name)
    student_avg = monthly_df.groupby("Student_norm")["Numeric Pages"].agg(["mean", "count"]).reset_index()
    student_avg.columns = ["Student_norm", "Avg Pages/Day", "Days Submitted"]
    student_norm_to_orig = {normalize_student_name(s): s for s in student_list}
    student_avg["Student"] = student_avg["Student_norm"].map(student_norm_to_orig)
    student_avg = student_avg.dropna(subset=["Student"])
    student_avg["Avg Pages/Day"] = student_avg["Avg Pages/Day"].round(1)
    expected = get_expected_pages(selected_class)
    student_avg["Expected"] = expected
    student_avg["% of Expected"] = (student_avg["Avg Pages/Day"] / expected * 100).round(1)
    st.subheader("📈 Student Performance (Monthly)")
    st.dataframe(student_avg[["Student", "Avg Pages/Day", "Days Submitted", "Expected", "% of Expected"]], use_container_width=True, hide_index=True)
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

# ======================
# Google Sheet Loader
# ======================
def load_data_from_google_sheet(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        xls = pd.ExcelFile(tmp_path)
        os.unlink(tmp_path)
        return xls
    except Exception as e:
        st.error(f"❌ Not able to Load Data From Google Sheet: {e}")
        st.stop()

# ======================
# PDF GENERATION FUNCTIONS (Dynamic Exam Support)
# ======================
def generate_cbse_report_pdf(student_name, class_sec, df_student, buffer):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
    except ImportError:
        return
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,
        spaceAfter=8,
        textColor=colors.HexColor("#1a3d6d")
    )
    subhead_style = ParagraphStyle('SubHead', parent=styles['Heading2'], alignment=1, fontSize=12, spaceAfter=15)
    normal_center = ParagraphStyle('NormalCenter', parent=styles['Normal'], alignment=1)
    elements.append(Paragraph("Macro Vision Academy, Burhanpur", title_style))
    elements.append(Paragraph("Academic Performance Report", subhead_style))
    elements.append(Paragraph(f"Student: <b>{student_name}</b>", normal_center))
    elements.append(Paragraph(f"Class: <b>{class_sec}</b>", normal_center))
    elements.append(Spacer(1, 15))
    
    # Dynamically get unique exams
    exam_names = sorted(df_student['Exam'].unique())
    data = [["Subject"] + exam_names + ["Grade", "Remarks"]]
    
    subjects = sorted(df_student['Subject'].unique())
    for subj in subjects:
        subj_data = df_student[df_student['Subject'] == subj]
        row = [subj]
        for exam in exam_names:
            exam_row = subj_data[subj_data['Exam'] == exam]
            mark = int(exam_row['RawMark'].iloc[0]) if not exam_row.empty else '--'
            row.append(str(mark))
        pct = subj_data['Percentage'].mean()
        grade = assign_grade(pct) if pd.notna(pct) else 'N/A'
        if grade in ['A1', 'A2']:
            remarks = "Excellent"
        elif grade in ['B1', 'B2']:
            remarks = "Good"
        elif grade in ['C1', 'C2', 'D']:
            remarks = "Needs Improvement"
        elif grade == 'E':
            remarks = "Urgent Attention"
        else:
            remarks = "N/A"
        row.extend([grade, remarks])
        data.append(row)
    
    overall_pct = df_student['Percentage'].mean()
    final_grade = assign_grade(overall_pct)
    data.append(["", *[""] * len(exam_names), "", ""])
    data.append(["**Overall Performance**", *[""] * (len(exam_names)-1), f"**{round(overall_pct, 1)}%**", f"**{final_grade}**", ""])
    
    col_widths = [1.4*inch] + [0.9*inch]*len(exam_names) + [0.8*inch, 1.4*inch]
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('GRID', (0,0), (-1,-1), 0.7, colors.black),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<i>Note: Grades as per CBSE grading system.</i>", ParagraphStyle('Note', fontSize=8, alignment=1, textColor=colors.gray)))
    doc.build(elements)

def generate_class_report_pdf(class_name, section, long_df, buffer):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
    except ImportError:
        return
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,
        spaceAfter=8,
        textColor=colors.HexColor("#1a3d6d")
    )
    normal_center = ParagraphStyle('NormalCenter', parent=styles['Normal'], alignment=1)
    elements.append(Paragraph("Macro Vision Academy, Burhanpur", title_style))
    elements.append(Paragraph("Class Performance Summary Report", styles['Heading2']))
    elements.append(Paragraph(f"Class: <b>{class_name} {section}</b>", normal_center))
    elements.append(Spacer(1, 15))
    student_avg_series = long_df.groupby('Student')['Percentage'].mean()
    class_avg = student_avg_series.mean()
    top5 = student_avg_series.nlargest(5)
    bottom5 = student_avg_series.nsmallest(5)
    failed_students = student_avg_series[student_avg_series < 33].index.tolist()
    summary_data = [
        ["Metric", "Value"],
        ["Class Average (%)", f"{class_avg:.1f}"],
        ["Top Performer", top5.index[0] if not top5.empty else "N/A"],
        ["Students Below 33%", str(len(failed_students))],
    ]
    summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2c3e50")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,1), (-1,-1), 10),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("<b>Top 5 Performers</b>", styles['Heading3']))
    top_data = [["Rank", "Student", "Avg %"]]
    for i, (name, avg) in enumerate(top5.items(), 1):
        top_data.append([str(i), name, f"{avg:.1f}"])
    top_table = Table(top_data, colWidths=[0.5*inch, 1.8*inch, 0.8*inch])
    top_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('FONTSIZE', (0,0), (-1,-1), 9)]))
    elements.append(top_table)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>Students Needing Support (Bottom 5)</b>", styles['Heading3']))
    bottom_data = [["Rank", "Student", "Avg %"]]
    for i, (name, avg) in enumerate(bottom5.items(), 1):
        bottom_data.append([str(i), name, f"{avg:.1f}"])
    bottom_table = Table(bottom_data, colWidths=[0.5*inch, 1.8*inch, 0.8*inch])
    bottom_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('FONTSIZE', (0,0), (-1,-1), 9)]))
    elements.append(bottom_table)
    elements.append(Spacer(1, 10))
    if failed_students:
        elements.append(Paragraph(f"<b>Students Below 33% ({len(failed_students)}):</b>", styles['Heading3']))
        failed_text = ", ".join(failed_students)
        elements.append(Paragraph(failed_text, styles['Normal']))
        elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>Recommendations</b>", styles['Heading3']))
    recs = []
    if class_avg < 60:
        recs.append("• Conduct remedial classes for core subjects.")
    if len(failed_students) > 5:
        recs.append("• Initiate individualized learning plans for at-risk students.")
    if not recs:
        recs.append("• Maintain current strategy; recognize top performers.")
    for rec in recs:
        elements.append(Paragraph(rec, styles['Normal']))
    doc.build(elements)

# ======================
# LOGIN PAGE
# ======================
def show_login_page():
    st.set_page_config(page_title="Macro Vision Academy - Login", layout="centered")
    st.markdown(
        """
        <style>
        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }
        .login-header {
            text-align: center;
            margin-bottom: 1.5rem;
            color: #1a3d6d;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        try:
            st.image("logo.png", width=300)
        except:
            st.image("https://via.placeholder.com/220x60?text=Macro+Vision", width=220)
    st.markdown('<h1 class="main-header"> Student Performance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h3 align=center class="branding">Macro Vision Academy, Burhanpur</h3>', unsafe_allow_html=True)
    with st.form("login_form"):
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["Admin", "Incharge", "Guruji"])
        submitted = st.form_submit_button("Login")
        if submitted:
            users = load_users()
            if role == "Admin":
                user_key = "admin"
                if user_key not in users:
                    st.error("❌ Admin account not found.")
                    return
                user = users[user_key]
                if user["role"] != "admin":
                    st.error("❌ Invalid admin role.")
                    return
            else:
                if username not in users:
                    st.error("❌ User not found.")
                    return
                user = users[username]
                expected_role = "incharge" if role == "Incharge" else "guruji"
                if user["role"] != expected_role:
                    st.error(f"❌ '{username}' is not an {role}. Please select correct role.")
                    return
                user_key = username
            if verify_password(password, user["password"]):
                st.session_state["logged_in"] = True
                st.session_state["username"] = user_key
                st.session_state["role"] = user["role"]
                create_session_file(user_key, user["role"])
                st.rerun()
            else:
                st.error("❌ Invalid password")
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("© 2025 Macro Vision Academy, Burhanpur")

# ======================
# Admin/Incharge Panels
# ======================
def show_admin_panel():
    if st.button("← Back", key="back_from_admin"):
        st.session_state.pop("show_admin", None)
        st.rerun()
    st.markdown('<div class="section-header">🔐 Admin User Management</div>', unsafe_allow_html=True)
    users = load_users()
    incharge_users = {k: v for k, v in users.items() if v["role"] == "incharge"}
    if "add_user" not in st.session_state:
        st.session_state["add_user"] = ""
    if "add_pass" not in st.session_state:
        st.session_state["add_pass"] = ""
    st.subheader("➕ Add New Incharge")
    with st.form("add_incharge"):
        new_user = st.text_input("New Username", value=st.session_state["add_user"], key="add_user_input")
        new_pass = st.text_input("New Password", type="password", value=st.session_state["add_pass"], key="add_pass_input")
        try:
            xls = load_data_from_google_sheet(GOOGLE_SHEET_URL)
            all_sheet_names = xls.sheet_names
            sheet_info = [{'SheetName': name, 'Class': parse_sheet_name(name)[0], 'Section': parse_sheet_name(name)[1]} for name in all_sheet_names]
            sheet_df = pd.DataFrame(sheet_info)
            all_classes = sorted(sheet_df['Class'].unique())
            assigned_classes = st.multiselect("Assign Classes", all_classes)
            assigned_sections = {}
            for cls in assigned_classes:
                secs = sheet_df[sheet_df['Class'] == cls]['Section'].dropna().astype(str).unique().tolist()
                secs = [s for s in secs if s.strip() != ""]
                assigned_sections[cls] = st.multiselect(f"Sections for {cls}", secs)
        except:
            assigned_classes = []
            assigned_sections = {}
        add_btn = st.form_submit_button("Add Incharge")
        if add_btn:
            if new_user and new_pass:
                if new_user in users:
                    st.error("User already exists")
                else:
                    users[new_user] = {
                        "password": hash_password(new_pass),
                        "plain_password": new_pass,
                        "role": "incharge",
                        "audit_log": [],
                        "assigned_classes": assigned_classes,
                        "assigned_sections": assigned_sections
                    }
                    save_users(users)
                    st.success(f"Incharge '{new_user}' added successfully!")
                    st.session_state["add_user"] = ""
                    st.session_state["add_pass"] = ""
                    st.rerun()
            else:
                st.warning("Please fill both fields")
    st.subheader("✏️ Manage Incharges")
    for user, data in incharge_users.items():
        with st.expander(f"Incharge: {user}"):
            new_user_name = st.text_input(f"Username", value=user, key=f"user_{user}")
            new_password = st.text_input(f"Password", type="password", key=f"pass_{user}")
            try:
                xls = load_data_from_google_sheet(GOOGLE_SHEET_URL)
                all_sheet_names = xls.sheet_names
                sheet_info = [{'SheetName': name, 'Class': parse_sheet_name(name)[0], 'Section': parse_sheet_name(name)[1]} for name in all_sheet_names]
                sheet_df = pd.DataFrame(sheet_info)
                all_classes = sorted(sheet_df['Class'].unique())
                curr_assigned = data.get("assigned_classes", [])
                curr_sections = data.get("assigned_sections", {})
                edited_classes = st.multiselect("Assigned Classes", all_classes, default=curr_assigned, key=f"edit_cls_{user}")
                edited_sections = {}
                for cls in edited_classes:
                    secs = sheet_df[sheet_df['Class'] == cls]['Section'].dropna().astype(str).unique().tolist()
                    secs = [s for s in secs if s.strip() != ""]
                    default_secs = curr_sections.get(cls, [])
                    edited_sections[cls] = st.multiselect(f"Sections for {cls}", secs, default=default_secs, key=f"edit_sec_{user}_{cls}")
            except:
                edited_classes = []
                edited_sections = {}
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update", key=f"update_{user}"):
                    if new_user_name != user:
                        users[new_user_name] = users.pop(user)
                    if new_password:
                        users[new_user_name]["password"] = hash_password(new_password)
                        users[new_user_name]["plain_password"] = new_password
                    users[new_user_name]["assigned_classes"] = edited_classes
                    users[new_user_name]["assigned_sections"] = edited_sections
                    save_users(users)
                    st.rerun()
            with col2:
                if st.button("Delete", key=f"delete_{user}"):
                    del users[user]
                    save_users(users)
                    st.success(f"Incharge '{user}' deleted")
                    st.rerun()
    st.subheader("📝 Audit Log")
    all_logs = []
    for u, d in users.items():
        if "audit_log" in d:
            for log in d["audit_log"]:
                all_logs.append(f"{log} (by {u})")
    if all_logs:
        st.text_area("Recent Changes", "\n".join(all_logs[-10:]), height=200)
    else:
        st.info("No changes recorded yet.")
    st.subheader("🔐 Incharge Credentials (Visible to Admin Only)")
    if incharge_users:
        cred_data = []
        for user, data in incharge_users.items():
            plain_pass = data.get("plain_password", "⚠️ Not available")
            cred_data.append({"Username": user, "Password": plain_pass})
        cred_df = pd.DataFrame(cred_data)
        st.dataframe(cred_df, use_container_width=True, hide_index=True)
    else:
        st.info("No incharges created yet.")

def show_incharge_panel():
    if st.button("← Back", key="back_from_incharge"):
        st.session_state.pop("show_incharge", None)
        st.rerun()
    st.markdown('<div class="section-header">📚 Guruji Management</div>', unsafe_allow_html=True)
    users = load_users()
    guruji_users = {k: v for k, v in users.items() if v["role"] == "guruji"}
    current_user_data = users.get(st.session_state.get("username"), {})
    assigned_classes = current_user_data.get("assigned_classes", [])
    if not assigned_classes:
        st.warning("You have no classes assigned. Contact Admin.")
        return
    sheet_info = []
    try:
        xls = load_data_from_google_sheet(GOOGLE_SHEET_URL)
        all_sheet_names = xls.sheet_names
        sheet_info = [{'SheetName': name, 'Class': parse_sheet_name(name)[0], 'Section': parse_sheet_name(name)[1]} for name in all_sheet_names]
        sheet_df = pd.DataFrame(sheet_info)
    except:
        pass
    if "add_guruji_user" not in st.session_state:
        st.session_state["add_guruji_user"] = ""
    if "add_guruji_pass" not in st.session_state:
        st.session_state["add_guruji_pass"] = ""
    st.subheader("➕ Add New Guruji")
    with st.form("add_guruji"):
        new_user = st.text_input("New Username", value=st.session_state["add_guruji_user"], key="add_guruji_user_input")
        new_pass = st.text_input("New Password", type="password", value=st.session_state["add_guruji_pass"], key="add_guruji_pass_input")
        guruji_class = st.selectbox("Assign Class", assigned_classes)
        guruji_section = ""
        if guruji_class:
            secs = sheet_df[sheet_df['Class'] == guruji_class]['Section'].dropna().astype(str).unique().tolist()
            secs = [s for s in secs if s.strip() != ""]
            guruji_section = st.selectbox("Assign Section", secs)
        add_btn = st.form_submit_button("Add Guruji")
        if add_btn:
            if new_user and new_pass and guruji_class and guruji_section:
                if new_user in users:
                    st.error("Guruji already exists")
                else:
                    users[new_user] = {
                        "password": hash_password(new_pass),
                        "plain_password": new_pass,
                        "role": "guruji",
                        "audit_log": [],
                        "assigned_class": guruji_class,
                        "assigned_section": guruji_section
                    }
                    save_users(users)
                    st.success(f"Guruji '{new_user}' added successfully!")
                    st.session_state["add_guruji_user"] = ""
                    st.session_state["add_guruji_pass"] = ""
                    st.rerun()
            else:
                st.warning("Please fill all fields")
    st.subheader("✏️ Manage Guruji")
    for user, data in guruji_users.items():
        with st.expander(f"Guruji: {user}"):
            new_user_name = st.text_input(f"Username", value=user, key=f"guruji_user_{user}")
            new_password = st.text_input(f"Password", type="password", key=f"guruji_pass_{user}")
            g_cls = data.get("assigned_class", "N/A")
            g_sec = data.get("assigned_section", "N/A")
            st.write(f"Assigned: Class {g_cls} {g_sec}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update", key=f"update_guruji_{user}"):
                    if new_user_name != user:
                        users[new_user_name] = users.pop(user)
                    if new_password:
                        users[new_user_name]["password"] = hash_password(new_password)
                        users[new_user_name]["plain_password"] = new_password
                    save_users(users)
                    st.rerun()
            with col2:
                if st.button("Delete", key=f"delete_guruji_{user}"):
                    del users[user]
                    save_users(users)
                    st.success(f"Guruji '{user}' deleted")
                    st.rerun()
    st.subheader("🔐 Guruji Credentials (Visible to Incharge Only)")
    if guruji_users:
        cred_data = []
        for user, data in guruji_users.items():
            plain_pass = data.get("plain_password", "⚠️ Not available")
            cred_data.append({"Username": user, "Password": plain_pass})
        cred_df = pd.DataFrame(cred_data)
        st.dataframe(cred_df, use_container_width=True, hide_index=True)
    else:
        st.info("No gurujis created yet.")
    st.markdown('<div class="section-header">✏️ Set Homework Page Rules (Class-wise)</div>', unsafe_allow_html=True)
    default_rules = [
        {"class_start": 1, "class_end": 5, "pages": 5},
        {"class_start": 6, "class_end": 8, "pages": 10},
        {"class_start": 9, "class_end": 10, "pages": 15},
        {"class_start": 11, "class_end": 12, "pages": 20}
    ]
    rules_file = "homework_rules.json"
    if os.path.exists(rules_file):
        with open(rules_file, "r") as f:
            rules = json.load(f)
    else:
        rules = default_rules
    updated_rules = []
    for i, rule in enumerate(rules):
        st.markdown(f"**Rule {i+1}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.number_input("Class From", min_value=1, max_value=12, value=rule["class_start"], key=f"start_{i}")
        with col2:
            end = st.number_input("Class To", min_value=1, max_value=12, value=rule["class_end"], key=f"end_{i}")
        with col3:
            pages = st.number_input("Pages", min_value=1, max_value=50, value=rule["pages"], key=f"pages_{i}")
        updated_rules.append({"class_start": start, "class_end": end, "pages": pages})
    if st.button("💾 Save Homework Rules"):
        with open(rules_file, "w") as f:
            json.dump(updated_rules, f, indent=4)
        st.success("✅ Homework rules updated successfully!")
        st.rerun()

# ======================
# Change Password
# ======================
def show_change_password():
    if st.button("← Back", key="back_from_change_pass"):
        st.session_state.pop("change_password", None)
        st.rerun()
    st.markdown('<div class="section-header">🔑 Change Password</div>', unsafe_allow_html=True)
    users = load_users()
    current_user = st.session_state["username"]
    if "curr_pass" not in st.session_state:
        st.session_state["curr_pass"] = ""
    if "new_pass1" not in st.session_state:
        st.session_state["new_pass1"] = ""
    if "new_pass2" not in st.session_state:
        st.session_state["new_pass2"] = ""
    with st.form("change_pass"):
        current_pass = st.text_input("Current Password", type="password", value=st.session_state["curr_pass"], key="curr_pass_input")
        new_pass = st.text_input("New Password", type="password", value=st.session_state["new_pass1"], key="new_pass1_input")
        confirm_pass = st.text_input("Confirm New Password", type="password", value=st.session_state["new_pass2"], key="new_pass2_input")
        submit = st.form_submit_button("Update Password")
        if submit:
            if not verify_password(current_pass, users[current_user]["password"]):
                st.error("Current password is incorrect")
            elif new_pass != confirm_pass:
                st.error("New passwords do not match")
            else:
                users[current_user]["password"] = hash_password(new_pass)
                if users[current_user]["role"] in ["incharge", "guruji"]:
                    users[current_user]["plain_password"] = new_pass
                log_msg = f"Password changed on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
                if "audit_log" not in users[current_user]:
                    users[current_user]["audit_log"] = []
                users[current_user]["audit_log"].append(log_msg)
                save_users(users)
                st.session_state["curr_pass"] = ""
                st.session_state["new_pass1"] = ""
                st.session_state["new_pass2"] = ""
                st.session_state.pop("change_password", None)
                st.rerun()

# ======================
# DYNAMIC EXAM HELPER FUNCTIONS
# ======================
def find_student_column(columns):
    for col in columns:
        col_str = str(col[0]).strip().lower()
        if col_str in ["student name", "name", "student"]:
            return col
        if "student" in col_str or "name" in col_str:
            return col
    return None

def extract_exams_and_subjects_from_columns(columns):
    exams_set = set()
    subject_exams = {}
    for col in columns:
        if isinstance(col, tuple) and len(col) >= 2:
            subj = str(col[0])
            metric = str(col[1])
            if subj == "Student Name" or subj == "Name":
                continue
            exam_name = get_exam_name_from_total(metric)
            if exam_name:
                exams_set.add(exam_name)
                if subj not in subject_exams:
                    subject_exams[subj] = set()
                subject_exams[subj].add(exam_name)
    exam_list = sorted(exams_set)
    for subj in subject_exams:
        subject_exams[subj] = sorted(subject_exams[subj])
    return exam_list, subject_exams

def safe_float(x):
    try:
        return float(x) if pd.notna(x) else np.nan
    except:
        return np.nan

def assign_grade(percentage):
    if pd.isna(percentage):
        return None
    if percentage >= 91: return 'A1'
    elif percentage >= 81: return 'A2'
    elif percentage >= 71: return 'B1'
    elif percentage >= 61: return 'B2'
    elif percentage >= 51: return 'C1'
    elif percentage >= 41: return 'C2'
    elif percentage >= 33: return 'D'
    else: return 'E'

def preprocess_data_dynamic(df, exam_list, subject_exams):
    student_col = find_student_column(df.columns)
    if student_col is None:
        raise ValueError("Student Name column not found")
    records = []
    for _, row in df.iterrows():
        student_name_raw = row[student_col]
        student_name = normalize_student_name(student_name_raw)
        for subj, exams in subject_exams.items():
            for exam in exams:
                marks_val = None
                percentage = None
                for marks, name in EXAM_MAPPING.items():
                    if name == exam:
                        col_key = (subj, f"Total({marks})")
                        if col_key in df.columns:
                            val = safe_float(row[col_key])
                            if not np.isnan(val):
                                marks_val = val
                                percentage = (val / marks) * 100
                                break
                if marks_val is None:
                    for col in df.columns:
                        if isinstance(col, tuple) and col[0] == subj:
                            exam_candidate = get_exam_name_from_total(col[1])
                            if exam_candidate == exam:
                                val = safe_float(row[col])
                                if not np.isnan(val):
                                    try:
                                        max_marks = int(col[1].split("(")[1].split(")")[0])
                                        marks_val = val
                                        percentage = (val / max_marks) * 100
                                    except:
                                        continue
                if marks_val is not None:
                    records.append({
                        'Student': student_name,
                        'Subject': subj,
                        'Exam': exam,
                        'RawMark': marks_val,
                        'Percentage': percentage
                    })
    return pd.DataFrame(records)

def parse_sheet_name(sheet_name):
    parts = str(sheet_name).strip().split(maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    else:
        return parts[0], parts[1]

# ======================
# MAIN DASHBOARD — FULLY DYNAMIC
# ======================
def show_main_dashboard():
    st.set_page_config(
        page_title="📊 Student Performance Dashboard – Macro Vision Academy",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🎓"
    )
    st.markdown(
        """
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
        body, [data-testid="stAppViewContainer"] {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(to bottom, #f9fbfd, #ffffff);
        }
        .main-header { font-size: clamp(1.5rem, 5vw, 2.5rem); text-align: center; margin: 0.5rem 0; font-weight: 700; color: #1a3d6d; }
        .branding { text-align: center; color: #4a5568; font-size: clamp(0.9rem, 2.8vw, 1.1rem); margin-bottom: 0.3rem; }
        .section-header { font-size: clamp(1.2rem, 4vw, 1.8rem); margin: 2rem 0 1rem; padding-bottom: 0.4rem; border-bottom: 2px solid #cbd5e0; color: #2c3e50; }
        .insight-box { background: #f8fafc; padding: 1rem; border-radius: 10px; margin: 1.2rem 0; box-shadow: 0 2px 6px rgba(0,0,0,0.05); border-left: 4px solid #3182ce; }
        .cbse-report { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin: 1.5rem 0; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True
    )
    try:
        st.sidebar.image("logo.png", use_container_width=True)
    except:
        st.sidebar.image("https://via.placeholder.com/150x50?text=Macro+Vision", use_container_width=True)
    
    if "cached_xls" not in st.session_state:
        try:
            with st.spinner("🔄 Data Loading..........."):
                st.session_state["cached_xls"] = load_data_from_google_sheet(GOOGLE_SHEET_URL)
        except Exception as e:
            st.error(f"Not able to Load Data From Google Sheet {e}")
            st.stop()
    xls = st.session_state["cached_xls"]
    current_role = st.session_state.get("role")
    current_user = st.session_state.get("username")
    users = load_users()
    user_data = users.get(current_user, {})
    
    if current_role == "admin":
        if st.sidebar.button("🔐 Admin Panel"):
            st.session_state["show_admin"] = True
        if st.sidebar.button("🔑 Change My Password"):
            st.session_state["change_password"] = True
    elif current_role == "incharge":
        if st.sidebar.button("📚 Guruji Panel"):
            st.session_state["show_incharge"] = True
        if st.sidebar.button("🔑 Change Password"):
            st.session_state["change_password"] = True
    elif current_role == "guruji":
        if st.sidebar.button("🔑 Change Password"):
            st.session_state["change_password"] = True
    
    if st.session_state.get("show_admin"):
        show_admin_panel()
        return
    if st.session_state.get("show_incharge"):
        show_incharge_panel()
        return
    if st.session_state.get("change_password"):
        show_change_password()
        return
    
    if st.session_state.get("logged_in"):
        if st.sidebar.button("🚪 Logout"):
            clear_session(st.session_state["username"])
            st.session_state.clear()
            st.rerun()
    
    if st.session_state.get("logged_in"):
        if st.sidebar.button("🔄 Refresh Data"):
            st.session_state.pop("selected_class", None)
            st.session_state.pop("selected_section", None)
            st.rerun()
    
    # ======================
    # 🔒 FILTER SHEETS BY ROLE & ASSIGNMENT
    # ======================
    all_sheet_names = xls.sheet_names
    sheet_info = [{'SheetName': name, 'Class': parse_sheet_name(name)[0], 'Section': parse_sheet_name(name)[1]} for name in all_sheet_names]
    sheet_df = pd.DataFrame(sheet_info)
    
    if current_role == "admin":
        allowed_sheet_info = sheet_info
    elif current_role == "incharge":
        allowed_classes = user_data.get("assigned_classes", [])
        allowed_sections_map = user_data.get("assigned_sections", {})
        allowed_sheet_info = []
        for s in sheet_info:
            if s['Class'] in allowed_classes:
                allowed_secs = allowed_sections_map.get(s['Class'], [])
                if not allowed_secs or s['Section'] in allowed_secs:
                    allowed_sheet_info.append(s)
    elif current_role == "guruji":
        g_class = user_data.get("assigned_class")
        g_section = user_data.get("assigned_section")
        if g_class and g_section:
            allowed_sheet_info = [s for s in sheet_info if s['Class'] == g_class and s['Section'] == g_section]
        else:
            allowed_sheet_info = []
    else:
        allowed_sheet_info = []
    
    if not allowed_sheet_info:
        st.error("❌ You have no access to any class/section. Contact Admin.")
        st.stop()
    
    classes = sorted(set(s['Class'] for s in allowed_sheet_info))
    
    def on_class_change():
        st.session_state["selected_section"] = ""
        st.session_state["analysis_type"] = "None"
        st.session_state["homework_action"] = "None"
    
    selected_class = st.sidebar.selectbox(
        "🏫 Select Class",
        [""] + classes,
        index=([""] + classes).index(st.session_state.get("selected_class", "")) if st.session_state.get("selected_class") in classes else 0,
        format_func=lambda x: "Select Class..." if x == "" else x,
        key="selected_class",
        on_change=on_class_change
    )
    
    selected_section = ""
    if selected_class:
        sections = [s['Section'] for s in allowed_sheet_info if s['Class'] == selected_class]
        sections = [s for s in sections if s and s.strip() != ""]
        if sections:
            def on_section_change():
                st.session_state["analysis_type"] = "None"
                st.session_state["homework_action"] = "None"
            selected_section = st.sidebar.selectbox(
                "🏫 Select Section",
                [""] + sections,
                index=([""] + sections).index(st.session_state.get("selected_section", "")) if st.session_state.get("selected_section") in sections else 0,
                format_func=lambda x: "Select Section..." if x == "" else x,
                key="selected_section",
                on_change=on_section_change
            )
    
    if not selected_class or not selected_section:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            try:
                st.image("logo.png", width=300)
            except:
                st.image("https://via.placeholder.com/220x60?text=Macro+Vision", width=220)
        st.markdown('<h1 class="main-header"> Student Performance Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<h3 class="branding">Macro Vision Academy, Burhanpur</h3>', unsafe_allow_html=True)
        st.info("✅ Please select **Class** and **Section** to proceed.")
        st.stop()
    
    matching_row = [s for s in allowed_sheet_info if s['Class'] == selected_class and s['Section'] == selected_section]
    if not matching_row:
        st.error("Sheet not found.")
        st.stop()
    selected_sheet = matching_row[0]['SheetName']
    
    try:
        raw_df = pd.read_excel(xls, sheet_name=selected_sheet, header=None)
        header_row = None
        for i in range(min(5, len(raw_df))):
            if any("Total(" in str(c) for c in raw_df.iloc[i]):
                header_row = i - 1
                break
        if header_row is None:
            st.error("❌ Header detection failed.")
            st.stop()
        df = pd.read_excel(xls, sheet_name=selected_sheet, header=[header_row, header_row+1])
        
        # 🔥 DYNAMIC EXAM DETECTION
        exam_list, subject_exams = extract_exams_and_subjects_from_columns(df.columns)
        if not exam_list:
            st.error("❌ No exams detected (look for 'Total(X)' columns).")
            st.stop()
        long_df = preprocess_data_dynamic(df, exam_list, subject_exams)
        
        student_col = find_student_column(df.columns)
        if student_col is None:
            st.error("❌ 'Student Name' column not found.")
            st.stop()
        student_list_raw = df[student_col].dropna().astype(str).tolist()
        student_list_raw = [s.strip() for s in student_list_raw if s.strip() not in ['nan', 'None', '', 'Student Name', 'Name']]
        student_list = [normalize_student_name(s) for s in student_list_raw]
        if not student_list:
            st.error("⚠️ No valid student names found.")
            st.stop()
        norm_to_orig = {normalize_student_name(s): s for s in student_list_raw}
    except Exception as e:
        st.error(f"Data loading error: {e}")
        st.stop()
    
    # Integrate Homework Data
    hw_df = load_homework_data()
    if not hw_df.empty:
        expected_pages = get_expected_pages(selected_class)
        hw_df["Numeric Pages"] = hw_df["Pages Done"].apply(lambda x: x if isinstance(x, (int, float)) else 0)
        hw_avg_series = hw_df.groupby("Student")["Numeric Pages"].mean()
        hw_avg_series = (hw_avg_series / expected_pages * 100).clip(0, 100)
        student_homework_pct = hw_avg_series.reindex(student_list, fill_value=0).to_dict()
    else:
        student_homework_pct = {s: 0 for s in student_list}
    
    long_df["Homework %"] = long_df["Student"].map(student_homework_pct)
    
    if "homework_action" not in st.session_state:
        st.session_state["homework_action"] = "None"
    if "analysis_type" not in st.session_state:
        st.session_state["analysis_type"] = "None"
    
    if current_role in ["admin", "incharge"]:
        st.sidebar.markdown("---")
        st.sidebar.markdown("📚 **Homework Tracker**")
        new_hw = st.sidebar.radio(
            "",
            ["None", "View Daily Report", "View Monthly Report"],
            index=["None", "View Daily Report", "View Monthly Report"].index(st.session_state["homework_action"]),
            key="homework_key"
        )
        if new_hw != st.session_state["homework_action"]:
            st.session_state["homework_action"] = new_hw
            st.session_state["analysis_type"] = "None"
            st.rerun()
        st.sidebar.markdown("---")
        st.sidebar.markdown("🔍 **Analysis Type**")
        new_an = st.sidebar.radio(
            "",
            ["None", "Class Analysis", "Student Analysis"],
            index=["None", "Class Analysis", "Student Analysis"].index(st.session_state["analysis_type"]),
            key="analysis_key"
        )
        if new_an != st.session_state["analysis_type"]:
            st.session_state["analysis_type"] = new_an
            st.session_state["homework_action"] = "None"
            st.rerun()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        try:
            st.image("logo.png", width=300)
        except:
            st.image("https://via.placeholder.com/220x60?text=Macro+Vision", width=220)
    st.markdown('<h1 class="main-header"> Student Performance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="branding">Macro Vision Academy, Burhanpur</h3>', unsafe_allow_html=True)
    
    if current_role == "guruji":
        st.info("✅ Guruji: Enter daily homework for students.")
        show_homework_entry_ui(selected_class, selected_section, [norm_to_orig.get(n, n) for n in student_list])
        st.markdown("---")
        show_monthly_status_view(selected_class, selected_section, [norm_to_orig.get(n, n) for n in student_list])
        return
    
    if st.session_state["homework_action"] == "View Daily Report":
        show_homework_daily_report(selected_class, selected_section, [norm_to_orig.get(n, n) for n in student_list])
    elif st.session_state["homework_action"] == "View Monthly Report":
        show_homework_monthly_report(selected_class, selected_section, [norm_to_orig.get(n, n) for n in student_list])
    elif st.session_state["analysis_type"] == "Class Analysis":
        st.markdown('<div class="section-header"> Class Overview</div>', unsafe_allow_html=True)
        
        # 🔥 DYNAMIC EXAM VIEW SELECTION
        exam_views = ["None"]
        for exam in exam_list:
            exam_views.append(f"{exam} Only")
        for i in range(len(exam_list)):
            for j in range(i+1, len(exam_list)):
                exam_views.append(f"{exam_list[i]} vs {exam_list[j]}")
        
        current_view = st.session_state.get("selected_exam_view", "None")
        if current_view not in exam_views:
            current_view = "None"
        
        selected_view = st.radio(
            "📊 Select Exam View",
            exam_views,
            index=exam_views.index(current_view),
            key="exam_view_radio"
        )
        
        if selected_view != current_view:
            st.session_state["selected_exam_view"] = selected_view
            st.rerun()
        
        if selected_view == "None":
            st.info("✅ Please select an exam view.")
            st.stop()
        
        # Parse selected_view
        if " vs " in selected_view:
            exam_a, exam_b = selected_view.replace(" Only", "").split(" vs ")
            filtered_df = long_df[long_df['Exam'].isin([exam_a, exam_b])].copy()
            is_comparison = True
        else:
            exam_name = selected_view.replace(" Only", "")
            filtered_df = long_df[long_df['Exam'] == exam_name].copy()
            is_comparison = False
            exam_a = exam_b = exam_name
        
        filtered_df['Grade'] = filtered_df['Percentage'].apply(assign_grade)
        student_academic_avg = filtered_df.groupby('Student')['Percentage'].mean()
        class_avg = student_academic_avg.mean()
        homework_class_avg = np.mean([student_homework_pct.get(s, 0) for s in student_academic_avg.index])
        class_max = student_academic_avg.max()
        class_min = student_academic_avg.min()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Avg Academic %", f"{class_avg:.1f}%")
        col2.metric("📚 Avg Homework %", f"{homework_class_avg:.1f}%")
        col3.metric("🔝 Highest %", f"{class_max:.1f}%")
        col4.metric("🔻 Lowest %", f"{class_min:.1f}%")
        
        st.markdown('<div class="section-header">🧠 Homework Impact Analysis</div>', unsafe_allow_html=True)
        impact_df = pd.DataFrame({
            'Student': student_academic_avg.index,
            'Academic %': student_academic_avg.values,
            'Homework %': [student_homework_pct.get(s, 0) for s in student_academic_avg.index]
        })
        impact_df = impact_df.dropna(subset=['Academic %'])
        if not impact_df.empty:
            corr = impact_df['Homework %'].corr(impact_df['Academic %'])
            if not pd.isna(corr):
                st.markdown(f"**Correlation between Homework % and Academic %**: **{corr:.2f}**")
                if corr > 0.5:
                    st.success("✅ Strong positive correlation: Homework completion strongly linked to better scores.")
                elif corr > 0.2:
                    st.info("ℹ️ Moderate positive trend.")
                else:
                    st.warning("⚠️ Weak or no correlation observed.")
            fig_scatter = px.scatter(
                impact_df,
                x='Homework %',
                y='Academic %',
                hover_data={'Student': True, 'Academic %': ':.1f', 'Homework %': ':.1f'},
                title='Student-wise: Homework Completion vs Academic Performance',
                labels={'Homework %': 'Homework Completion (%)', 'Academic %': 'Academic Performance (%)'},
                color_discrete_sequence=['#2e8b57']
            )
            fig_scatter.update_traces(
                marker=dict(size=10, line=dict(width=1.5, color='DarkSlateGrey')),
                hovertemplate="<b>%{customdata[0]}</b><br>Academic: %{y:.1f}%<br>Homework: %{x:.1f}%<extra></extra>"
            )
            fig_scatter.update_layout(
                xaxis=dict(range=[0, 100], title='Homework Completion (%)'),
                yaxis=dict(range=[0, 100], title='Academic Performance (%)'),
                showlegend=False,
                plot_bgcolor='white',
                hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No student data available for correlation analysis.")
        
        st.subheader("🏅 Grade Distribution")
        VALID_GRADES = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D', 'E']
        grade_data = [g for g in filtered_df['Grade'].dropna() if g in VALID_GRADES]
        if grade_data:
            grades_series = pd.Series(grade_data)
            grade_counts = grades_series.value_counts().reindex(VALID_GRADES, fill_value=0)
            grade_counts = grade_counts[grade_counts > 0]
            if not grade_counts.empty:
                pie_data = pd.DataFrame({'Grade': grade_counts.index, 'Count': grade_counts.values})
                colors = px.colors.sequential.Blues[::-1][:len(pie_data)]
                fig = px.pie(pie_data, names='Grade', values='Count', color='Grade', color_discrete_sequence=colors, hole=0.3)
                fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14, textfont_color='white')
                fig.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid grades to display.")
        else:
            st.info("No grade data available.")
        
        st.subheader("🏆 Top 5 & 🚨 Bottom 5 Performers")
        student_avg_filtered = filtered_df.groupby('Student')['Percentage'].mean().sort_values(ascending=False)
        top5 = student_avg_filtered.head(5)
        bottom5 = student_avg_filtered.tail(5)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top 5:**")
            for i, (student, avg) in enumerate(top5.items(), 1):
                hw_pct = student_homework_pct.get(student, 0)
                orig_name = norm_to_orig.get(student, student)
                st.write(f"{i}. **{orig_name}** – {avg:.1f}% (HW: {hw_pct:.1f}%)")
        with col2:
            st.write("**Bottom 5:**")
            for i, (student, avg) in enumerate(bottom5.items(), 1):
                hw_pct = student_homework_pct.get(student, 0)
                orig_name = norm_to_orig.get(student, student)
                st.write(f"{i}. **{orig_name}** – {avg:.1f}% (HW: {hw_pct:.1f}%)")
        
        st.subheader("📚 Subject Wise (Class Avg %)")
        subject_ranking = filtered_df.groupby('Subject')['Percentage'].mean().sort_values(ascending=False)
        if not subject_ranking.empty:
            rank_df = subject_ranking.reset_index()
            rank_df.columns = ['Subject', 'Average %']
            fig = px.bar(rank_df, x='Subject', y='Average %', color='Subject', color_discrete_sequence=px.colors.qualitative.Bold, text='Average %')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', textfont_size=14)
            fig.update_layout(showlegend=False, xaxis_title="Subjects", yaxis_title="Average Percentage (%)", yaxis=dict(range=[0, 100]), margin=dict(t=30, b=80, l=50, r=20))
            st.plotly_chart(fig, use_container_width=True)
        
        if is_comparison:
            st.markdown(f'<div class="section-header"> Class: Subject Comparison ({exam_a} vs {exam_b})</div>', unsafe_allow_html=True)
            pt_class = long_df[long_df['Exam'] == exam_a].groupby('Subject')['Percentage'].mean()
            mt_class = long_df[long_df['Exam'] == exam_b].groupby('Subject')['Percentage'].mean()
            all_subjects = sorted(set(pt_class.index) | set(mt_class.index))
            pt_vals = [pt_class.get(s, 0) for s in all_subjects]
            mt_vals = [mt_class.get(s, 0) for s in all_subjects]
            fig, ax = plt.subplots(figsize=(12, 6))
            indices = np.arange(len(all_subjects))
            width = 0.35
            bars1 = ax.bar(indices - width/2, pt_vals, width, label=exam_a, color='#2e8b57')
            bars2 = ax.bar(indices + width/2, mt_vals, width, label=exam_b, color='#cd5c5c')
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='darkgreen')
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='darkred')
            ax.set_xlabel("Subjects", fontweight='bold')
            ax.set_ylabel("Average Percentage (%)", fontweight='bold')
            ax.set_title(f"Class Average: {exam_a} vs {exam_b}", fontweight='bold')
            ax.set_xticks(indices)
            ax.set_xticklabels(all_subjects, rotation=30, ha='right')
            ax.set_ylim(0, 100)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        
        st.markdown('<div class="section-header">🧠 Advanced Academic Diagnostics</div>', unsafe_allow_html=True)
        advanced_insights = []
        if is_comparison:
            pt_subj_avg = long_df[long_df['Exam'] == exam_a].groupby('Subject')['Percentage'].mean()
            mt_subj_avg = long_df[long_df['Exam'] == exam_b].groupby('Subject')['Percentage'].mean()
            volatility = ((mt_subj_avg - pt_subj_avg).abs()).sort_values(ascending=False)
            if not volatility.empty:
                most_volatile = volatility.index[0]
                vol_value = volatility.iloc[0]
                if vol_value >= 8:
                    advanced_insights.append(f"⚠️ **High Volatility in {most_volatile}**: Score changed by **{vol_value:.1f}%** — Review teaching consistency or assessment difficulty.")
                elif vol_value >= 4:
                    advanced_insights.append(f"ℹ️ **Moderate Shift in {most_volatile}**: Changed by **{vol_value:.1f}%** — Monitor next cycle.")
            pt_per_student = long_df[long_df['Exam'] == exam_a].groupby('Student')['Percentage'].mean()
            mt_per_student = long_df[long_df['Exam'] == exam_b].groupby('Student')['Percentage'].mean()
            delta_per_student = (mt_per_student - pt_per_student).dropna()
            improving = delta_per_student[delta_per_student > 3].index.tolist()
            declining = delta_per_student[delta_per_student < -3].index.tolist()
            if len(improving) > len(student_list) * 0.3:
                advanced_insights.append("✅ **Positive Momentum**: Over 30% students show significant improvement — teaching strategy is effective.")
            if len(declining) > len(student_list) * 0.2:
                advanced_insights.append("❗ **At-Risk Group**: Over 20% students declined significantly — schedule parent meetings & remedial plan.")
        homework_series = pd.Series([student_homework_pct.get(s, 0) for s in student_academic_avg.index], index=student_academic_avg.index)
        academic_series = student_academic_avg
        gap_df = pd.DataFrame({'Homework %': homework_series, 'Academic %': academic_series})
        gap_df['Gap'] = gap_df['Academic %'] - gap_df['Homework %']
        large_gap_students = gap_df[gap_df['Gap'] > 30].index.tolist()
        reverse_gap_students = gap_df[gap_df['Gap'] < -30].index.tolist()
        if large_gap_students:
            advanced_insights.append(f"🔍 **Inconsistent Learners**: {len(large_gap_students)} students score well despite low homework — may indicate external coaching or assessment issues.")
        if reverse_gap_students:
            advanced_insights.append(f"📚 **Hardworking but Struggling**: {len(reverse_gap_students)} students complete homework but score low — suggest concept gaps or exam anxiety.")
        std_dev = student_academic_avg.std()
        if std_dev > 15:
            advanced_insights.append("🧩 **High Performance Spread**: Std. Dev. >15% — wide gap between toppers & laggards. Consider differentiated instruction.")
        elif std_dev < 8:
            advanced_insights.append("🎯 **Uniform Performance**: Std. Dev. <8% — consistent understanding. Ready for advanced topics.")
        at_risk = student_academic_avg[student_academic_avg < 40].index.tolist()
        if len(at_risk) > len(student_list) * 0.25:
            advanced_insights.append("🚨 **Early Warning**: >25% students below 40% — initiate ILP (Individual Learning Plan) immediately.")
        if not advanced_insights:
            advanced_insights.append("✅ **Healthy Academic Profile**: No critical risks detected. Continue current pedagogy.")
        for insight in advanced_insights:
            st.markdown(f"- {insight}")
        
        st.subheader("💡 Deep Insights & Recommendations")
        insights_list = []
        if is_comparison:
            pt_class_avg = long_df[long_df['Exam'] == exam_a]['Percentage'].mean()
            mt_class_avg = long_df[long_df['Exam'] == exam_b]['Percentage'].mean()
            growth = mt_class_avg - pt_class_avg
            if growth >= 2:
                insights_list.append(f"✅ **Positive Momentum**: Class shows significant improvement (>2%) from {exam_a} to {exam_b}.")
            elif growth <= -2:
                insights_list.append(f"⚠️ **Academic Concern**: Class average dropped by more than 2% from {exam_a} to {exam_b}. Intervention needed.")
            pt_class = long_df[long_df['Exam'] == exam_a].groupby('Subject')['Percentage'].mean()
            mt_class = long_df[long_df['Exam'] == exam_b].groupby('Subject')['Percentage'].mean()
            for subj in sorted(set(pt_class.index) | set(mt_class.index)):
                pt = pt_class.get(subj, 0)
                mt = mt_class.get(subj, 0)
                diff = mt - pt
                if diff >= 5:
                    insights_list.append(f"🟢 **{subj}**: Strong improvement (**+{diff:.1f}%**). Recognize subject teacher!")
                elif diff <= -5:
                    insights_list.append(f"🔴 **{subj}**: Significant decline (**{diff:.1f}%**). Diagnostic test recommended.")
        else:
            if not subject_ranking.empty:
                top_subj = subject_ranking.index[0]
                bottom_subj = subject_ranking.index[-1]
                insights_list.append(f"🟢 **Top Subject**: {top_subj} ({subject_ranking.iloc[0]:.1f}%)")
                insights_list.append(f"🔴 **Weakest Subject**: {bottom_subj} ({subject_ranking.iloc[-1]:.1f}%)")
            homework_avg = homework_class_avg
            if homework_avg < 50:
                insights_list.append(f"⚠️ **Low Homework Engagement**: Class average homework completion is only {homework_avg:.1f}%. Recommend parent notice.")
            elif homework_avg > 85:
                insights_list.append(f"✅ **Excellent Homework Compliance**: {homework_avg:.1f}% — likely contributing to academic success.")
        student_overall_avg = long_df.groupby('Student')['Percentage'].mean()
        failed_students = student_overall_avg[student_overall_avg < 33].index.tolist()
        if failed_students:
            failed_names = ", ".join([norm_to_orig.get(s, s) for s in failed_students[:5]])
            more = f" (and {len(failed_students)-5} more)" if len(failed_students) > 5 else ""
            insights_list.append(f"❗ **{len(failed_students)} student(s)** scored below 33% overall: {failed_names}{more}. Immediate remedial plan required.")
        else:
            insights_list.append("✅ **No student below 33%** overall. Strong foundational understanding across class.")
        grade_counts_filtered = filtered_df['Grade'].value_counts().reindex(VALID_GRADES, fill_value=0)
        total_students_filtered = len(filtered_df['Student'].unique())
        if grade_counts_filtered['A1'] + grade_counts_filtered['A2'] >= 0.3 * total_students_filtered:
            insights_list.append("🌟 **Top Performers**: Over 30% of class scored A1/A2. Consider enrichment activities.")
        for insight in insights_list:
            st.markdown(f"- {insight}")
        
        if current_role in ["admin", "incharge"]:
            st.markdown('<div class="section-header">🎯 Actionable Suggestions Based on Performance</div>', unsafe_allow_html=True)
            exam_key = selected_view.replace(" ", "_").replace("-", "").replace("vs", "vs")
            suggestions_file = f"suggestions_{selected_class}_{selected_section}_{exam_key}.json"
            if os.path.exists(suggestions_file):
                with open(suggestions_file, 'r') as f:
                    suggestions_status = json.load(f)
            else:
                suggestions_status = {}
            suggestions = []
            if not is_comparison:
                weakest_subj = subject_ranking.index[-1] if not subject_ranking.empty else "N/A"
                suggestions.extend([
                    f"🔹 **Early Intervention**: {exam_name} assesses foundational concepts. Organize remedial sessions for students below 60%.",
                    f"🔹 **Concept Reinforcement**: Use activity-based learning for weak subjects like {weakest_subj}.",
                    f"🔹 **Parent Engagement**: Share {exam_name} results with parents to align home-school support.",
                    f"🔹 **Baseline for Growth**: Use this data to set personalized goals for next exam improvement."
                ])
            else:
                growth = long_df[long_df['Exam'] == exam_b]['Percentage'].mean() - long_df[long_df['Exam'] == exam_a]['Percentage'].mean()
                if growth > 0:
                    suggestions.extend([
                        "🔹 **Celebrate Growth**: Acknowledge teachers/students driving improvement.",
                        "🔹 **Scale Best Practices**: Identify strategies from improving subjects and replicate.",
                        "🔹 **Bridge Gaps**: For declining subjects, conduct root-cause analysis (content, pedagogy, assessment)."
                    ])
                else:
                    suggestions.extend([
                        "🔹 **Diagnostic Review**: Analyze question papers & answer scripts to identify learning gaps.",
                        "🔹 **Differentiated Instruction**: Group students by performance for targeted teaching.",
                        "🔹 **Motivational Workshops**: Address confidence issues in consistently low performers."
                    ])
                suggestions.append("🔹 **Holistic Tracking**: Maintain student-wise progress dashboards for continuous monitoring.")
            if homework_class_avg < 60:
                suggestions.append("🔹 **Homework Improvement Plan**: Introduce daily homework tracking (as per Guruji system) and share completion % with parents.")
            for i, sug in enumerate(suggestions):
                key = f"suggestion_{i}"
                if key not in suggestions_status:
                    suggestions_status[key] = "Pending"
                status = suggestions_status[key]
                status_color = "green" if status == "Done" else "orange"
                col1, col2 = st.columns([9, 1])
                with col1:
                    st.markdown(f"- {sug} &nbsp; <span style='color:{status_color}; font-weight:bold;'>[{status}]</span>", unsafe_allow_html=True)
                with col2:
                    if current_role == "incharge" and status == "Pending":
                        if st.button("✅", key=f"done_{selected_class}_{selected_section}_{exam_key}_{i}"):
                            suggestions_status[key] = "Done"
                            with open(suggestions_file, 'w') as f:
                                json.dump(suggestions_status, f, indent=4)
                            st.rerun()
                    else:
                        st.empty()
        
        st.markdown('<div class="section-header"> Export Reports</div>', unsafe_allow_html=True)
        class_summary = long_df.groupby(['Student', 'Exam'])['Percentage'].mean().unstack(fill_value=0)
        output_excel = BytesIO()
        with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
            class_summary.to_excel(writer, sheet_name='Class_Summary')
            long_df.to_excel(writer, sheet_name='Raw_Data')
        st.download_button("📥 Download Class Report (Excel)", output_excel.getvalue(), f"Class_{selected_class}_{selected_section}_Report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        try:
            pdf_buffer_class = BytesIO()
            generate_class_report_pdf(selected_class, selected_section, long_df, pdf_buffer_class)
            st.download_button(
                label="🖨️ Download Complete Class Report (PDF)",
                data=pdf_buffer_class.getvalue(),
                file_name=f"Class_{selected_class}_{selected_section}_Full_Report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")
    
    elif st.session_state["analysis_type"] == "Student Analysis":
        selected_student = st.sidebar.selectbox(
            "👤 Select Student",
            student_list,
            index=0,
            key=f"student_select_{selected_class}_{selected_section}"
        )
        selected_orig = norm_to_orig.get(selected_student, selected_student)
        st.session_state["selected_student"] = selected_student
        st.markdown(f'<div class="section-header"> Student Performance Explorer: {selected_orig}</div>', unsafe_allow_html=True)
        student_data = long_df[long_df['Student'] == selected_student]
        if student_data.empty:
            st.warning(f"No data for {selected_orig}")
            st.stop()
        homework_pct = student_homework_pct.get(selected_student, 0)
        class_sec_label = f"{selected_class} {selected_section}".strip()
        st.markdown('<div class="cbse-report">', unsafe_allow_html=True)
        st.subheader("📄 Exam - Performance Summary")
        cbse_data = []
        subjects = sorted(student_data['Subject'].unique())
        for subj in subjects:
            subj_df = student_data[student_data['Subject'] == subj]
            exam_marks = {}
            for exam in exam_list:
                exam_row = subj_df[subj_df['Exam'] == exam]
                mark = int(exam_row['RawMark'].iloc[0]) if not exam_row.empty else "--"
                exam_marks[exam] = mark
            pct = subj_df['Percentage'].mean()
            grade = assign_grade(pct) if pd.notna(pct) else 'N/A'
            row = [subj]
            for exam in exam_list:
                row.append(exam_marks[exam])
            row.extend([grade])
            cbse_data.append(row)
        headers = ["Subject"] + exam_list + ["Grade"]
        cbse_df = pd.DataFrame(cbse_data, columns=headers)
        st.dataframe(cbse_df, use_container_width=True)
        overall_pct = student_data['Percentage'].mean()
        final_grade = assign_grade(overall_pct)
        st.markdown(f"**Overall Percentage**: {overall_pct:.1f}% &nbsp; | &nbsp; **Final Grade**: **{final_grade}**")
        st.markdown(f"**Homework Completion %**: **{homework_pct:.1f}%**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if is_comparison:
            st.subheader(f"📊 {selected_orig}: Subject Comparison ({exam_a} vs {exam_b})")
            pt_scores = student_data[student_data['Exam'] == exam_a].set_index('Subject')['Percentage']
            mt_scores = student_data[student_data['Exam'] == exam_b].set_index('Subject')['Percentage']
            all_subjects_s = sorted(set(pt_scores.index) | set(mt_scores.index))
            pt_vals_s = [pt_scores.get(s, 0) for s in all_subjects_s]
            mt_vals_s = [mt_scores.get(s, 0) for s in all_subjects_s]
            fig, ax = plt.subplots(figsize=(12, 6))
            indices = np.arange(len(all_subjects_s))
            width = 0.35
            bars1 = ax.bar(indices - width/2, pt_vals_s, width, label=exam_a, color='#2e8b57')
            bars2 = ax.bar(indices + width/2, mt_vals_s, width, label=exam_b, color='#cd5c5c')
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='darkgreen')
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9, color='darkred')
            ax.set_xlabel("Subjects", fontweight='bold')
            ax.set_ylabel("Percentage (%)", fontweight='bold')
            ax.set_title(f"{selected_orig}: {exam_a} vs {exam_b}", fontweight='bold')
            ax.set_xticks(indices)
            ax.set_xticklabels(all_subjects_s, rotation=30, ha='right')
            ax.set_ylim(0, 100)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.subheader("💡 Subject-wise Insights")
            for subj in all_subjects_s:
                pt = pt_scores.get(subj, 0)
                mt = mt_scores.get(subj, 0)
                diff = mt - pt
                if diff >= 5:
                    st.markdown(f"- **{subj}**: <span style='color:green;'>Significant improvement (**+{diff:.1f}%**)</span>", unsafe_allow_html=True)
                elif diff <= -5:
                    st.markdown(f"- **{subj}**: <span style='color:red;'>Major decline (**{abs(diff):.1f}%**) — Needs attention</span>", unsafe_allow_html=True)
                elif diff > 0:
                    st.markdown(f"- **{subj}**: <span style='color:green;'>Slight improvement (+{diff:.1f}%)</span>", unsafe_allow_html=True)
                elif diff < 0:
                    st.markdown(f"- **{subj}**: <span style='color:red;'>Slight decline ({abs(diff):.1f}%)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"- **{subj}**: No change", unsafe_allow_html=True)
        
        st.subheader("🧠 Homework Impact Analysis")
        if homework_pct < 50:
            st.warning(f"❗ **Critical**: {selected_orig}'s homework completion (**{homework_pct:.1f}%**) is very low. This is strongly correlated with lower academic performance.")
        elif homework_pct < 70:
            st.info(f"ℹ️ **Improvement Area**: Increasing homework completion above 70% could significantly boost scores.")
        else:
            st.success(f"✅ **Excellent**: Consistent homework (**{homework_pct:.1f}%**) is likely a key factor in strong performance.")
        
        st.subheader("🧠 Holistic Performance Analysis")
        insights = []
        subject_avg = student_data.groupby('Subject')['Percentage'].mean().reset_index()
        if not subject_avg.empty:
            best_row = subject_avg.loc[subject_avg['Percentage'].idxmax()]
            worst_row = subject_avg.loc[subject_avg['Percentage'].idxmin()]
            insights.append(f"✅ **Academic Strength**: {best_row['Subject']} ({best_row['Percentage']:.1f}%)")
            insights.append(f"❗ **Area of Concern**: {worst_row['Subject']} ({worst_row['Percentage']:.1f}%)")
        if is_comparison:
            pt_avg = student_data[student_data['Exam'] == exam_a]['Percentage'].mean()
            mt_avg = student_data[student_data['Exam'] == exam_b]['Percentage'].mean()
            diff = mt_avg - pt_avg
            if diff >= 3:
                insights.append("📈 **Positive Trend**: Consistent improvement across exams.")
            elif diff <= -3:
                insights.append("📉 **Warning**: Performance dropped significantly. Recommend parent-teacher meeting.")
            else:
                insights.append("↔️ **Stable Performance**: Scores are consistent. Maintain focus.")
        final_grade = assign_grade(student_data['Percentage'].mean())
        grade_remark = ""
        if final_grade in ['A1', 'A2']:
            grade_remark = "Outstanding performance! Keep up the excellence."
        elif final_grade in ['B1', 'B2']:
            grade_remark = "Good work. With focused effort, you can reach A grades."
        elif final_grade in ['C1', 'C2']:
            grade_remark = "Needs more practice. Regular revision will help improve."
        elif final_grade == 'D':
            grade_remark = "Below average. Immediate remedial support is advised."
        elif final_grade == 'E':
            grade_remark = "Critical. Urgent intervention and academic counseling required."
        if grade_remark:
            insights.append(f"🔖 **Overall Remark**: {grade_remark}")
        st.markdown('<div class="insight-box">' + "<br>".join(insights) + '</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header"> Export Reports</div>', unsafe_allow_html=True)
        student_report = student_data[['Exam', 'Subject', 'RawMark', 'Percentage']].copy()
        student_report['Grade'] = student_report['Percentage'].apply(assign_grade)
        output2 = BytesIO()
        with pd.ExcelWriter(output2, engine='xlsxwriter') as writer:
            student_report.to_excel(writer, index=False, sheet_name='Student_Report')
        st.download_button("📥 Download Student Report (Excel)", output2.getvalue(), f"{selected_orig}_Report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        try:
            pdf_buffer = BytesIO()
            generate_cbse_report_pdf(selected_orig, class_sec_label, student_data, pdf_buffer)
            st.download_button(
                label="🖨️ Download CBSE-Style Report (PDF)",
                data=pdf_buffer.getvalue(),
                file_name=f"{selected_orig}_CBSE_Report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")
    else:
        st.info("✅ Please select **Homework Tracker** or **Analysis Type** from the sidebar.")
    
    st.markdown("---")
    st.caption("💡 Grade chart shows computed grades (A1–E) from percentages. Bar charts show exact % values.")

# ======================
# MAIN
# ======================
def main():
    if "logged_in" not in st.session_state:
        username, role = load_active_session()
        if username and role:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = role
        else:
            show_login_page()
            return
    show_main_dashboard()

if __name__ == "__main__":
    main()