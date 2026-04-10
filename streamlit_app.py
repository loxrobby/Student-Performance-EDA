from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import scipy.stats as stats
import streamlit as st


DATA_PATH_DEFAULT = Path(__file__).with_name("Students_Grading_Dataset.csv")
GRADE_ORDER = ["A", "B", "C", "D", "F"]
INCOME_ORDER = ["Low", "Medium", "High"]
YES_NO_ORDER = ["Yes", "No"]
PARENT_EDU_ORDER = ["None", "High School", "Bachelor's", "Master's", "PhD", "Not Reported"]
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "Student_ID": "Unique identifier for each student.",
    "First_Name": "Student’s first name.",
    "Last_Name": "Student’s last name.",
    "Email": "Student email (personally identifying).",
    "Gender": "Male / Female (standardized).",
    "Age": "Student age.",
    "Department": "Student department (e.g., CS, Engineering, Business, Mathematics).",
    "Attendance (%)": "Attendance percentage (0–100).",
    "Midterm_Score": "Midterm exam score (out of 100).",
    "Final_Score": "Final exam score (out of 100).",
    "Assignments_Avg": "Average assignment score (out of 100).",
    "Quizzes_Avg": "Average quiz score (out of 100).",
    "Participation_Score": "Class participation score (0–10).",
    "Projects_Score": "Project evaluation score (out of 100).",
    "Total_Score": "Weighted sum / overall score.",
    "Grade": "Letter grade (A, B, C, D, F).",
    "Study_Hours_per_Week": "Average study hours per week.",
    "Extracurricular_Activities": "Whether the student participates in extracurriculars (Yes/No).",
    "Internet_Access_at_Home": "Whether the student has internet at home (Yes/No).",
    "Parent_Education_Level": "Highest education level of parents (including 'Not Reported').",
    "Family_Income_Level": "Low / Medium / High.",
    "Stress_Level (1-10)": "Self-reported stress level (1 low – 10 high).",
    "Sleep_Hours_per_Night": "Average sleep hours per night.",
}

# Notebook-authored markdown summaries (extracted from the EDA sections).
# These are shown in the Per-column EDA to closely mirror the notebook narrative.
NOTEBOOK_COLUMN_SUMMARIES: dict[str, str] = {
    "Grade": """
## Summary (from notebook)

### **📊 Summary Of Grade Column Analysis**

**1️⃣ Grade Distribution**
- **A**: 🟢 1495 students (**29.9%**)
- **B**: 🟡 978 students (**19.6%**)
- **C**: 🟠 794 students (**15.9%**)
- **D**: 🔴 889 students (**17.8%**)
- **F**: ⚫ 844 students (**16.9%**)

---

**2️⃣ Attendance & Grades Relationship**
- 📈 **Attendance (%) has a strong positive correlation with grades** (**r = 0.57**).
- **Average Attendance by Grade:**
  - **A**: 🟢 **85.32%**
  - **B**: 🟡 **81.42%**
  - **C-F**: 🔴 **65.7% – 68.1%**
- **Attendance Thresholds:**
  - ✅ **≥90% Attendance** → **A (68.2%) & B (31.8%)**
  - ⚠️ **<60% Attendance** → **D (42%) & F (29.9%)**

---

### **🔍 Key Takeaways**
✔️ **Attendance is the primary driver of academic success.**  
✔️ **Clear thresholds (60%, 80%, 90%) define grade patterns.**  
✔️ **Students with high attendance rarely fail, while low attendance strongly correlates with lower grades.**  

🚀 **Final Insight: Want an A? Keep attendance ≥90%!**
""".strip(),
    "Gender": """
## Summary (from notebook)

### **📊 Summary Of Gender Column Analysis**

**1️⃣ Gender Distribution**
- **Males**: 👨 **62.06%** (3103 students)
- **Females**: 👩 **37.94%** (1897 students)

---

**3️⃣ Attendance by Gender**
- **Mean Attendance (%):**  
  - 👩 **75.57%** | 👨 **75.35%**
- 📊 Attendance levels are nearly identical.

---

**1️⃣1️⃣ Statistical Significance Testing**
- 📊 **Midterm, Final, Total Scores, Study Hours, Stress Levels** → **p > 0.05**  
- **Conclusion:** Differences between genders are **statistically insignificant**.

---

### **📌 Key Takeaways**
✔️ **Gender has no significant impact on academic performance.**  
✔️ **Grade distributions, attendance, and extracurricular involvement are nearly identical.**  
""".strip(),
    "Age": """
## Summary (from notebook)

### **📊 Summary Of Age Column Analysis**

**1️⃣ Age Distribution**
- **Range:** 18 to 24 years old (balanced representation).

---

**9️⃣ Statistical Significance**
- **ANOVA Test Results:** 📊 **p > 0.05**
- **Conclusion:** **No statistically significant differences** in academic or well-being metrics across age groups.

---

### **📌 Key Takeaways**
✔️ **Age has no significant impact** on academic performance.  
✔️ **Attendance, grades, and participation rates remain stable across ages.**  
""".strip(),
    "Department": """
## Summary (from notebook)

### **📊 Summary Of Department Column Analysis**

**1️⃣ Department Distribution**
- **CS:** 40.44% | **Engineering:** 29.38% | **Business:** 20.12% | **Mathematics:** 10.06%

---

**7️⃣ Statistical Significance**
- **ANOVA Test Results:** 📊 **p > 0.05**
- No statistically significant differences across departments in core metrics.

---

### **📌 Key Takeaways**
✔️ **Academic performance and attendance are stable across departments.**  
✔️ **No major performance gaps between departments.**
""".strip(),
    "Total_Score": """
## Summary (from notebook)

### **📊 Summary Of Total_Score Analysis**

**1️⃣ Distribution**
- Range: **50.02 – 99.99**, Mean: **75.12**

---

**2️⃣ Relationship with Grades**
- Weak correlation; high total scores do not strongly predict letter grades.

---

### **📌 Key Takeaways**
✔️ **Total_Score is weakly related to grades, attendance, and study habits.**  
✔️ **Further investigation is needed to understand the grading structure.**
""".strip(),
    "Final_Score": """
## Summary (from notebook)

### **📊 Summary Of Final_Score Analysis**

**1️⃣ Distribution**
- Range: **40.00 – 99.98**, Mean: **69.64**

---

### **📌 Key Takeaways**
✔️ **Final_Score has weak or negligible correlations** with most factors.  
✔️ **Final_Score alone does not define academic performance.**
""".strip(),
    "Study_Hours_per_Week": """
## Summary (from notebook)

### **📊 Summary Of Study_Hours_per_Week Analysis**

**1️⃣ Distribution**
- Range: **5 – 30**, Mean: **17.66 hours/week**

---

### **📌 Key Takeaways**
✔️ **Longer study hours do not directly translate into better academic performance.**  
✔️ **Academic success is influenced by multiple factors beyond just study time.**
""".strip(),
    "Extracurricular_Activities": """
## Summary (from notebook)

### **📊 Summary Of Extracurricular Activities Analysis**

**1️⃣ Participation Rates**
- **69.8%** No | **30.2%** Yes

---

### **📌 Key Takeaways**
✔️ **Extracurricular activities have weak or negligible impact** on academic performance.  
✔️ **Students in extracurriculars report slightly lower stress but also slightly less sleep.**
""".strip(),
    "Internet_Access_at_Home": """
## Summary (from notebook)

### **📊 Summary Of Internet Access at Home Analysis**

**1️⃣ Internet Access Rates**
- **89.5%** Yes | **10.5%** No

---

### **📌 Key Takeaways**
✔️ **Internet access has negligible influence** on academic performance and well-being.  
""".strip(),
    "Parent_Education_Level": """
## Summary (from notebook)

### **📊 Summary Of Parent Education Level Analysis**

**1️⃣ Largest Category**
- **"Not Reported" (35.6%)** → underreporting

---

### **📌 Key Takeaways**
✔️ **Parental education shows weak relationships** with academic and well-being metrics.  
""".strip(),
    "Family_Income_Level": """
## Summary (from notebook)

### **📊 Summary Of Family Income Level Analysis**

**1️⃣ Distribution**
- **Medium (40.0%)** largest, then Low (39.6%), High (20.4%)

---

### **📌 Key Takeaways**
✔️ **Family income level has minimal impact** on academic performance.  
""".strip(),
    "Stress_Level (1-10)": """
## Summary (from notebook)

### **📊 Summary Of Stress Level Analysis**

**1️⃣ Distribution**
- Range **1–10**, Mean **5.48**

---

### **📌 Key Takeaways**
✔️ **Stress level has minimal impact** on academic performance.  
""".strip(),
    "Sleep_Hours_per_Night": """
## Summary (from notebook)

### **📊 Summary Of Sleep Hours per Night Analysis**

**1️⃣ Distribution**
- Range **3–9**, Mean **6.48**

---

### **📌 Key Takeaways**
✔️ **Sleep hours have no strong impact** on academic performance.  
""".strip(),
}


@dataclass(frozen=True)
class Filters:
    departments: list[str]
    genders: list[str]
    ages: tuple[int, int]
    income_levels: list[str]
    parent_edu: list[str]
    internet: list[str]
    extracurricular: list[str]


@st.cache_data(show_spinner=False)
def load_raw_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def _standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    valid_categories: dict[str, list[str]] = {
        "Gender": ["Male", "Female"],
        "Family_Income_Level": INCOME_ORDER,
        "Internet_Access_at_Home": YES_NO_ORDER,
        "Extracurricular_Activities": ["Yes", "No"],
    }

    # Simple standardization (matches notebook intent)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"M": "Male", "F": "Female"})

    for col, valid in valid_categories.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if x in valid else "Not Reported")

    return df


def _clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numerical_columns = [
        c
        for c in [
            "Age",
            "Attendance (%)",
            "Midterm_Score",
            "Final_Score",
            "Total_Score",
            "Study_Hours_per_Week",
            "Stress_Level (1-10)",
            "Sleep_Hours_per_Night",
        ]
        if c in df.columns
    ]
    for col in numerical_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            lower_cap = df[col].quantile(0.05)
            upper_cap = df[col].quantile(0.95)
            df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
    return df


def _gender_correction_by_first_name(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Notebook logic: map a few common first names to gender and correct mismatches.
    Returns (df_fixed, mismatches_count).
    """
    df = df.copy()
    if not {"First_Name", "Gender"}.issubset(df.columns):
        return df, 0

    name_gender_mapping = {
        "Omar": "Male",
        "Maria": "Female",
        "Ahmed": "Male",
        "John": "Male",
        "Liam": "Male",
        "Sara": "Female",
        "Emma": "Female",
        "Ali": "Male",
    }

    expected = df["First_Name"].map(name_gender_mapping)
    mismatches = (expected.notna()) & (df["Gender"] != expected)
    mismatches_count = int(mismatches.sum())

    df.loc[mismatches, "Gender"] = expected[mismatches]
    return df, mismatches_count


def clean_data(
    df: pd.DataFrame,
    *,
    apply_gender_correction: bool = True,
    standardize_categories: bool = True,
    clip_outliers: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    df = df.copy()
    report: dict[str, object] = {}

    # Missing values (matches the notebook’s intent)
    if "Attendance (%)" in df.columns:
        missing = int(df["Attendance (%)"].isna().sum())
        report["missing_attendance"] = missing
        df["Attendance (%)"] = df["Attendance (%)"].fillna(df["Attendance (%)"].mean())
    if "Assignments_Avg" in df.columns:
        missing = int(df["Assignments_Avg"].isna().sum())
        report["missing_assignments"] = missing
        df["Assignments_Avg"] = df["Assignments_Avg"].fillna(df["Assignments_Avg"].median())
    if "Parent_Education_Level" in df.columns:
        missing = int(df["Parent_Education_Level"].isna().sum())
        report["missing_parent_edu"] = missing
        df["Parent_Education_Level"] = df["Parent_Education_Level"].fillna("Not Reported")

    # Duplicates by Student_ID (notebook checks this, result is 0 in provided dataset)
    if "Student_ID" in df.columns:
        dup = int(df.duplicated(subset=["Student_ID"]).sum())
        report["duplicate_student_id_rows"] = dup
        if dup:
            df = df.drop_duplicates(subset=["Student_ID"], keep="first")

    # Gender correction (name-based mapping in the notebook)
    if apply_gender_correction:
        df, mismatches = _gender_correction_by_first_name(df)
        report["gender_mismatches_corrected"] = mismatches

    # Standardize categorical columns (notebook intent)
    if standardize_categories:
        df = _standardize_categories(df)

    # Outlier treatment (notebook uses clipping at 5th/95th percentiles as precaution)
    if clip_outliers:
        df = _clip_outliers(df)

    # Common derived fields used throughout the notebook
    if "Grade" in df.columns:
        grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        df["Grade_Numeric"] = df["Grade"].map(grade_map)

    if "Attendance (%)" in df.columns:
        bins = [0, 60, 70, 80, 90, 100]
        labels = ["0-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
        df["Attendance_Tier"] = pd.cut(df["Attendance (%)"], bins=bins, labels=labels)

    if "Total_Score" in df.columns:
        df["Performance_Category"] = pd.cut(
            df["Total_Score"],
            bins=[0, 60, 70, 80, 90, 100],
            labels=["Poor", "Below Average", "Average", "Good", "Excellent"],
        )

    report["rows"] = int(df.shape[0])
    report["cols"] = int(df.shape[1])
    return df, report


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in ["Student_ID", "First_Name", "Last_Name", "Email"] if c in df.columns]
    return df.drop(columns=cols_to_drop)


def sidebar_filters(df: pd.DataFrame) -> Filters:
    st.sidebar.header("Filters")

    def multiselect(col: str, label: str) -> list[str]:
        if col not in df.columns:
            return []
        options = sorted([x for x in df[col].dropna().unique().tolist() if str(x).strip() != ""])
        default = options
        return st.sidebar.multiselect(label, options=options, default=default)

    departments = multiselect("Department", "Department")
    genders = multiselect("Gender", "Gender")
    income_levels = multiselect("Family_Income_Level", "Family income")
    parent_edu = multiselect("Parent_Education_Level", "Parent education")
    internet = multiselect("Internet_Access_at_Home", "Internet access at home")
    extracurricular = multiselect("Extracurricular_Activities", "Extracurricular activities")

    if "Age" in df.columns and pd.api.types.is_numeric_dtype(df["Age"]):
        min_age = int(df["Age"].min())
        max_age = int(df["Age"].max())
        ages = st.sidebar.slider("Age range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
    else:
        ages = (0, 10_000)

    return Filters(
        departments=departments,
        genders=genders,
        ages=ages,
        income_levels=income_levels,
        parent_edu=parent_edu,
        internet=internet,
        extracurricular=extracurricular,
    )


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    dff = df.copy()

    def keep_in(col: str, allowed: list[str]) -> None:
        nonlocal dff
        if col in dff.columns and allowed:
            dff = dff[dff[col].isin(allowed)]

    keep_in("Department", f.departments)
    keep_in("Gender", f.genders)
    keep_in("Family_Income_Level", f.income_levels)
    keep_in("Parent_Education_Level", f.parent_edu)
    keep_in("Internet_Access_at_Home", f.internet)
    keep_in("Extracurricular_Activities", f.extracurricular)

    if "Age" in dff.columns and pd.api.types.is_numeric_dtype(dff["Age"]):
        dff = dff[(dff["Age"] >= f.ages[0]) & (dff["Age"] <= f.ages[1])]

    return dff


def kpis(dff: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(dff):,}")
    if "Total_Score" in dff.columns:
        c2.metric("Avg total score", f"{dff['Total_Score'].mean():.2f}")
    if "Attendance (%)" in dff.columns:
        c3.metric("Avg attendance", f"{dff['Attendance (%)'].mean():.2f}%")
    if "Grade" in dff.columns:
        top_grade = dff["Grade"].value_counts().idxmax() if len(dff) else "—"
        c4.metric("Most common grade", str(top_grade))


def grade_distribution(dff: pd.DataFrame, *, key: str = "grade_distribution") -> None:
    if "Grade" not in dff.columns:
        return
    counts = dff["Grade"].value_counts().reindex(GRADE_ORDER).dropna()
    fig = px.bar(
        counts.reset_index(),
        x="Grade",
        y="count",
        title="Grade distribution",
        labels={"count": "Students", "Grade": "Grade"},
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def grade_vs_attendance(dff: pd.DataFrame, *, key: str = "grade_vs_attendance") -> None:
    if not {"Grade", "Attendance (%)"}.issubset(dff.columns):
        return
    fig = px.box(
        dff,
        x="Grade",
        y="Attendance (%)",
        category_orders={"Grade": GRADE_ORDER},
        title="Distribution of Attendance (%) by Grade (box plot)",
        points="outliers",
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

    avg = dff.groupby("Grade", dropna=False)["Attendance (%)"].mean().reindex(GRADE_ORDER)
    st.caption("Notebook-style summary")
    st.dataframe(
        pd.DataFrame(
            {
                "avg_attendance_%": avg.round(3),
                "students": dff["Grade"].value_counts().reindex(GRADE_ORDER).fillna(0).astype(int),
            }
        ),
        use_container_width=True,
    )


def grade_vs_total_score(dff: pd.DataFrame, *, key_box: str = "grade_vs_total_score_box", key_bar: str = "grade_vs_total_score_bar") -> None:
    if not {"Grade", "Total_Score"}.issubset(dff.columns):
        return
    fig = px.box(
        dff,
        x="Grade",
        y="Total_Score",
        category_orders={"Grade": GRADE_ORDER},
        title="Distribution of Total Score by Grade (box plot)",
        points="outliers",
    )
    st.plotly_chart(fig, use_container_width=True, key=key_box)

    tmp = dff.groupby("Grade", dropna=False)["Total_Score"].mean().reindex(GRADE_ORDER).reset_index()
    fig2 = px.bar(tmp, x="Grade", y="Total_Score", title="Average total score by grade")
    st.plotly_chart(fig2, use_container_width=True, key=key_bar)


def attendance_vs_performance(dff: pd.DataFrame, *, key: str = "attendance_vs_performance") -> None:
    needed = {"Attendance (%)", "Total_Score", "Grade"}
    if not needed.issubset(set(dff.columns)):
        return
    fig = px.scatter(
        dff,
        x="Attendance (%)",
        y="Total_Score",
        color="Grade",
        opacity=0.7,
        title="Attendance vs total score (colored by grade)",
        hover_data=["Department", "Gender", "Age"] if {"Department", "Gender", "Age"}.issubset(dff.columns) else None,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def score_by_attendance_tier(dff: pd.DataFrame, *, key: str = "score_by_attendance_tier") -> None:
    needed = {"Attendance_Tier", "Total_Score"}
    if not needed.issubset(set(dff.columns)):
        return
    tmp = (
        dff.groupby("Attendance_Tier", observed=True)["Total_Score"]
        .mean()
        .reset_index()
        .sort_values("Attendance_Tier")
    )
    fig = px.bar(
        tmp,
        x="Attendance_Tier",
        y="Total_Score",
        title="Average total score by attendance tier",
        labels={"Total_Score": "Avg total score", "Attendance_Tier": "Attendance tier"},
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def correlations_heatmap(dff: pd.DataFrame, *, key: str = "correlations_heatmap") -> None:
    numeric_cols = [
        c
        for c in [
            "Attendance (%)",
            "Midterm_Score",
            "Final_Score",
            "Assignments_Avg",
            "Quizzes_Avg",
            "Participation_Score",
            "Projects_Score",
            "Total_Score",
            "Study_Hours_per_Week",
            "Stress_Level (1-10)",
            "Sleep_Hours_per_Night",
            "Grade_Numeric",
        ]
        if c in dff.columns
    ]
    if len(numeric_cols) < 2:
        return

    corr = dff[numeric_cols].corr(numeric_only=True).round(3)

    # Use a diverging palette centered at 0 (clearer positives vs negatives)
    fig = px.imshow(
        corr,
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlation heatmap (numeric features)",
    )
    fig.update_layout(height=820)
    st.plotly_chart(fig, use_container_width=True, key=key)


def top_correlations_with_target(dff: pd.DataFrame, target: str = "Total_Score", *, key: str = "top_correlations_with_target") -> None:
    if target not in dff.columns:
        return
    numeric = dff.select_dtypes(include="number")
    if target not in numeric.columns or numeric.shape[1] < 2:
        return
    corrs = numeric.corr(numeric_only=True)[target].drop(labels=[target]).sort_values(key=lambda s: s.abs(), ascending=False)
    top = corrs.head(10)[::-1]
    fig = px.bar(
        top.reset_index(),
        x=target,
        y="index",
        orientation="h",
        title=f"Top correlations with {target} (absolute)",
        labels={"index": "feature", target: "correlation"},
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_{target}")


def department_violin(dff: pd.DataFrame, *, key: str = "department_violin") -> None:
    if not {"Department", "Total_Score"}.issubset(dff.columns):
        return
    fig = px.violin(
        dff,
        x="Department",
        y="Total_Score",
        box=True,
        points="outliers",
        title="Total score distribution by department (violin)",
    )
    fig.update_layout(xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True, key=key)


def heatmap_grade_by_department(dff: pd.DataFrame, *, key: str = "heatmap_grade_by_department") -> None:
    if not {"Grade", "Department"}.issubset(dff.columns):
        return
    ctab = pd.crosstab(dff["Grade"], dff["Department"], normalize="columns") * 100
    ctab = ctab.reindex(GRADE_ORDER).fillna(0)
    fig = px.imshow(
        ctab.round(1),
        text_auto=True,
        aspect="auto",
        title="Grade distribution by department (% within department)",
        labels={"x": "Department", "y": "Grade", "color": "%"},
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def histograms(dff: pd.DataFrame) -> None:
    numeric_cols = [c for c in dff.columns if pd.api.types.is_numeric_dtype(dff[c])]
    if not numeric_cols:
        return
    col = st.selectbox("Pick a numeric column", options=numeric_cols, index=numeric_cols.index("Total_Score") if "Total_Score" in numeric_cols else 0)
    bins = st.slider("Bins", min_value=5, max_value=60, value=30)
    fig = px.histogram(dff, x=col, nbins=bins, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True, key=f"hist_{col}_{bins}")


def per_column_eda(dff: pd.DataFrame, col: str) -> None:
    st.markdown(f"### Column EDA: `{col}`")
    if col not in dff.columns:
        st.warning("Column not found in the filtered dataset.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(dff):,}")
    c2.metric("Missing", f"{int(dff[col].isna().sum()):,}")
    c3.metric("Unique", f"{int(dff[col].nunique(dropna=True)):,}")

    st.caption(COLUMN_DESCRIPTIONS.get(col, ""))
    if col in NOTEBOOK_COLUMN_SUMMARIES:
        st.markdown(NOTEBOOK_COLUMN_SUMMARIES[col])

    series = dff[col]

    # Numeric EDA
    if pd.api.types.is_numeric_dtype(series):
        st.markdown("#### Summary statistics")
        st.dataframe(series.describe().to_frame(name=col), use_container_width=True)

        bins = st.slider("Histogram bins", min_value=5, max_value=80, value=30, key=f"bins_{col}")
        fig = px.histogram(dff, x=col, nbins=bins, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True, key=f"percol_{col}_hist_{bins}")

        # Relationship with Grade
        if "Grade" in dff.columns:
            fig2 = px.box(
                dff,
                x="Grade",
                y=col,
                category_orders={"Grade": GRADE_ORDER},
                title=f"{col} by Grade (box plot)",
                points="outliers",
            )
            st.plotly_chart(fig2, use_container_width=True, key=f"percol_{col}_by_grade_box")

        # Relationship with Total_Score (if available)
        if "Total_Score" in dff.columns and col != "Total_Score":
            fig3 = px.scatter(
                dff,
                x=col,
                y="Total_Score",
                opacity=0.6,
                title=f"{col} vs Total_Score",
                trendline="ols" if st.checkbox("Add trendline", value=False, key=f"trend_{col}") else None,
            )
            st.plotly_chart(fig3, use_container_width=True, key=f"percol_{col}_total_scatter")
        return

    # Categorical / object EDA
    st.markdown("#### Value counts")
    # NOTE: series can be pandas.Categorical; fillna on categorical requires the category to exist.
    series_str = series.astype("string").fillna("Not Reported")
    vc = series_str.value_counts().reset_index()
    vc.columns = [col, "count"]
    st.dataframe(vc, use_container_width=True)
    fig = px.bar(vc, x=col, y="count", title=f"{col} distribution")
    fig.update_layout(xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True, key=f"percol_{col}_bar")

    # Crosstab with Grade
    if "Grade" in dff.columns:
        st.markdown("#### Grade distribution within categories (%)")
        ctab = pd.crosstab(dff["Grade"], series_str, normalize="columns") * 100
        ctab = ctab.reindex(GRADE_ORDER).fillna(0)
        fig2 = px.imshow(
            ctab.round(1),
            text_auto=True,
            aspect="auto",
            title=f"Grade distribution by {col} (% within {col})",
            labels={"x": col, "y": "Grade", "color": "%"},
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig2, use_container_width=True, key=f"percol_{col}_grade_heatmap")

    # Total score by category
    if "Total_Score" in dff.columns:
        st.markdown("#### Total score by category")
        fig3 = px.box(
            dff,
            x=col,
            y="Total_Score",
            points="outliers",
            title=f"Total_Score by {col}",
        )
        fig3.update_layout(xaxis_title=None)
        st.plotly_chart(fig3, use_container_width=True, key=f"percol_{col}_total_box")


def insight_questions_section(dff: pd.DataFrame) -> None:
    st.markdown("### Insightful questions (with answers + charts)")
    st.caption("These questions are based on the themes and conclusions in the notebook, answered using the currently filtered data.")

    questions = [
        "1) What does the grade distribution look like?",
        "2) Does attendance differ by grade?",
        "3) How do grades change across attendance tiers?",
        "4) Is Final_Score strongly aligned with Grade?",
        "5) Does participation relate to Total_Score?",
        "6) Does sleep relate to attendance?",
        "7) Do study hours relate to Total_Score?",
        "8) Are there meaningful department differences in Total_Score?",
        "9) Do extracurricular activities relate to performance?",
        "10) Does internet access at home relate to performance?",
        "11) Does parent education relate to grade distribution?",
        "12) Does stress level relate to grades or total score?",
    ]
    q = st.selectbox("Choose a question", options=questions, key="qa_question_select")
    q_key = q.split(")")[0].strip()

    if q.startswith("1") and "Grade" in dff.columns:
        counts = dff["Grade"].value_counts().reindex(GRADE_ORDER).dropna()
        st.write("**Answer:** Most common grades and their counts/percentages.")
        dfq = counts.reset_index()
        dfq.columns = ["Grade", "Students"]
        dfq["Percent"] = (dfq["Students"] / dfq["Students"].sum() * 100).round(1)
        fig = px.bar(dfq, x="Grade", y="Students", title="Grade distribution")
        st.plotly_chart(fig, use_container_width=True, key=f"qa_{q_key}_grade_dist")
        st.dataframe(dfq, use_container_width=True)
        return

    if q.startswith("2") and {"Grade", "Attendance (%)"}.issubset(dff.columns):
        st.write("**Answer:** Attendance is compared across grades using a box plot and grade-level averages.")
        grade_vs_attendance(dff, key=f"qa_{q_key}_grade_att_box")
        return

    if q.startswith("3") and {"Attendance_Tier", "Grade"}.issubset(dff.columns):
        st.write("**Answer:** This heatmap shows the grade mix within each attendance tier.")
        ctab = pd.crosstab(dff["Grade"], dff["Attendance_Tier"], normalize="columns") * 100
        ctab = ctab.reindex(GRADE_ORDER).fillna(0)
        fig = px.imshow(
            ctab.round(1),
            text_auto=True,
            aspect="auto",
            title="Grade distribution by attendance tier (% within tier)",
            labels={"x": "Attendance tier", "y": "Grade", "color": "%"},
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"qa_{q_key}_att_tier_heatmap")
        return

    if q.startswith("4") and {"Final_Score", "Grade"}.issubset(dff.columns):
        st.write("**Answer:** If Final_Score alone determined Grade, we'd expect a strong monotonic pattern. Here we visualize the relationship.")
        fig = px.box(
            dff,
            x="Grade",
            y="Final_Score",
            category_orders={"Grade": GRADE_ORDER},
            title="Final_Score by Grade",
            points="outliers",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"qa_{q_key}_final_by_grade")
        if "Grade_Numeric" in dff.columns:
            corr = dff[["Final_Score", "Grade_Numeric"]].corr(numeric_only=True).iloc[0, 1]
            st.caption(f"Correlation (Final_Score vs Grade_Numeric): **{corr:.3f}**")
        return

    if q.startswith("5") and {"Participation_Score", "Total_Score"}.issubset(dff.columns):
        st.write("**Answer:** We split participation into high/low (median) and compare Total_Score.")
        median_part = float(dff["Participation_Score"].median())
        tmp = dff.assign(Participation_Group=dff["Participation_Score"].apply(lambda v: "High (>= median)" if v >= median_part else "Low (< median)"))
        fig = px.box(tmp, x="Participation_Group", y="Total_Score", title="Total_Score by participation group", points="outliers")
        st.plotly_chart(fig, use_container_width=True, key=f"qa_{q_key}_part_box")
        high = tmp.loc[tmp["Participation_Group"] == "High (>= median)", "Total_Score"].dropna()
        low = tmp.loc[tmp["Participation_Group"] == "Low (< median)", "Total_Score"].dropna()
        if len(high) > 1 and len(low) > 1:
            t, p = stats.ttest_ind(high, low, nan_policy="omit")
            st.caption(f"T-test p-value: **{p:.4f}** (p < 0.05 is commonly considered significant)")
        return

    if q.startswith("6") and {"Sleep_Hours_per_Night", "Attendance (%)"}.issubset(dff.columns):
        st.write("**Answer:** We compare attendance across higher vs lower sleep (median split) and visualize the trend.")
        fig = px.scatter(dff, x="Sleep_Hours_per_Night", y="Attendance (%)", opacity=0.6, title="Sleep hours vs Attendance (%)")
        st.plotly_chart(fig, use_container_width=True, key=f"qa_{q_key}_sleep_att_scatter")
        median_sleep = float(dff["Sleep_Hours_per_Night"].median())
        high = dff.loc[dff["Sleep_Hours_per_Night"] >= median_sleep, "Attendance (%)"].dropna()
        low = dff.loc[dff["Sleep_Hours_per_Night"] < median_sleep, "Attendance (%)"].dropna()
        if len(high) > 1 and len(low) > 1:
            t, p = stats.ttest_ind(high, low, nan_policy="omit")
            st.caption(f"T-test p-value (high vs low sleep): **{p:.4f}**")
        return

    if q.startswith("7") and {"Study_Hours_per_Week", "Total_Score"}.issubset(dff.columns):
        st.write("**Answer:** We visualize the relationship between study hours and total score.")
        fig = px.scatter(dff, x="Study_Hours_per_Week", y="Total_Score", opacity=0.6, title="Study hours per week vs Total_Score")
        st.plotly_chart(fig, use_container_width=True, key=f"qa_{q_key}_study_total_scatter")
        corr = dff[["Study_Hours_per_Week", "Total_Score"]].corr(numeric_only=True).iloc[0, 1]
        st.caption(f"Correlation: **{corr:.3f}**")
        return

    if q.startswith("8") and {"Department", "Total_Score"}.issubset(dff.columns):
        st.write("**Answer:** Compare total score distributions across departments.")
        department_violin(dff, key=f"qa_{q_key}_dept_violin")
        return

    if q.startswith("9") and {"Extracurricular_Activities", "Total_Score"}.issubset(dff.columns):
        st.write("**Answer:** Compare total score by extracurricular participation.")
        fig = px.box(dff, x="Extracurricular_Activities", y="Total_Score", points="outliers", title="Total_Score by extracurricular activities")
        st.plotly_chart(fig, use_container_width=True, key=f"qa_{q_key}_extra_box")
        return

    if q.startswith("10") and {"Internet_Access_at_Home", "Total_Score"}.issubset(dff.columns):
        st.write("**Answer:** Compare total score by home internet access.")
        fig = px.box(dff, x="Internet_Access_at_Home", y="Total_Score", points="outliers", title="Total_Score by internet access at home")
        st.plotly_chart(fig, use_container_width=True, key=f"qa_{q_key}_internet_box")
        return

    if q.startswith("11") and {"Parent_Education_Level", "Grade"}.issubset(dff.columns):
        st.write("**Answer:** Grade distribution within each parent education category.")
        edu = dff["Parent_Education_Level"].astype("string").fillna("Not Reported")
        ctab = pd.crosstab(dff["Grade"], edu, normalize="columns") * 100
        ctab = ctab.reindex(GRADE_ORDER).fillna(0)
        fig = px.imshow(
            ctab.round(1),
            text_auto=True,
            aspect="auto",
            title="Grade distribution by parent education (% within education level)",
            labels={"x": "Parent education", "y": "Grade", "color": "%"},
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"qa_{q_key}_parentedu_heatmap")
        return

    if q.startswith("12") and "Stress_Level (1-10)" in dff.columns:
        st.write("**Answer:** Visualize stress distribution and its relationship to grades / total score.")
        fig = px.histogram(dff, x="Stress_Level (1-10)", nbins=10, title="Stress level distribution")
        st.plotly_chart(fig, use_container_width=True, key=f"qa_{q_key}_stress_hist")
        if {"Grade", "Stress_Level (1-10)"}.issubset(dff.columns):
            fig2 = px.box(
                dff,
                x="Grade",
                y="Stress_Level (1-10)",
                category_orders={"Grade": GRADE_ORDER},
                title="Stress level by Grade",
                points="outliers",
            )
            st.plotly_chart(fig2, use_container_width=True, key=f"qa_{q_key}_stress_by_grade")
        if {"Total_Score", "Stress_Level (1-10)"}.issubset(dff.columns):
            fig3 = px.scatter(dff, x="Stress_Level (1-10)", y="Total_Score", opacity=0.6, title="Stress level vs Total_Score")
            st.plotly_chart(fig3, use_container_width=True, key=f"qa_{q_key}_stress_total_scatter")
        return

    st.info("This question can't be answered with the current filters/columns. Try widening filters or choosing another question.")

def notebook_hypothesis_tests(dff: pd.DataFrame) -> None:
    """
    Re-implements the notebook's early hypothesis testing block:
    - Attendance tier vs grades (T-Test idea in notebook)
    - Study hours by gender (T-Test)
    - Income vs study hours (ANOVA)
    - Participation vs total score (T-Test)
    - Sleep vs attendance (T-Test)
    """
    st.markdown("### Statistical tests (notebook-style)")
    st.caption("These tests help decide whether observed differences are likely real or due to random variation. We report p-values (p < 0.05 is commonly treated as 'significant').")

    rows: list[dict[str, object]] = []

    # H1: Attendance affects grades -> compare high vs low attendance total score/grade proxy
    if "Attendance (%)" in dff.columns:
        median_att = float(dff["Attendance (%)"].median())
        high = dff.loc[dff["Attendance (%)"] >= median_att, "Attendance (%)"].dropna()
        low = dff.loc[dff["Attendance (%)"] < median_att, "Attendance (%)"].dropna()
        if len(high) > 1 and len(low) > 1:
            t, p = stats.ttest_ind(high, low, nan_policy="omit")
            rows.append({"test": "T-Test", "question": "High vs low attendance (median split)", "metric": "Attendance (%)", "p_value": float(p)})

    # H2: Study hours by gender
    if {"Gender", "Study_Hours_per_Week"}.issubset(dff.columns):
        male = dff.loc[dff["Gender"] == "Male", "Study_Hours_per_Week"].dropna()
        female = dff.loc[dff["Gender"] == "Female", "Study_Hours_per_Week"].dropna()
        if len(male) > 1 and len(female) > 1:
            t, p = stats.ttest_ind(male, female, nan_policy="omit")
            rows.append({"test": "T-Test", "question": "Study hours differ by gender?", "metric": "Study_Hours_per_Week", "p_value": float(p)})

    # H3: Income affects study hours (ANOVA)
    if {"Family_Income_Level", "Study_Hours_per_Week"}.issubset(dff.columns):
        groups = [dff.loc[dff["Family_Income_Level"] == lvl, "Study_Hours_per_Week"].dropna() for lvl in INCOME_ORDER]
        if all(len(g) > 1 for g in groups):
            f, p = stats.f_oneway(*groups)
            rows.append({"test": "ANOVA", "question": "Study hours differ by income level?", "metric": "Study_Hours_per_Week", "p_value": float(p)})

    # H4: Participation impacts total score (T-Test)
    if {"Participation_Score", "Total_Score"}.issubset(dff.columns):
        median_part = float(dff["Participation_Score"].median())
        high = dff.loc[dff["Participation_Score"] >= median_part, "Total_Score"].dropna()
        low = dff.loc[dff["Participation_Score"] < median_part, "Total_Score"].dropna()
        if len(high) > 1 and len(low) > 1:
            t, p = stats.ttest_ind(high, low, nan_policy="omit")
            rows.append({"test": "T-Test", "question": "High vs low participation impacts total score?", "metric": "Total_Score", "p_value": float(p)})

    # H5: Sleep affects attendance (T-Test)
    if {"Sleep_Hours_per_Night", "Attendance (%)"}.issubset(dff.columns):
        median_sleep = float(dff["Sleep_Hours_per_Night"].median())
        high = dff.loc[dff["Sleep_Hours_per_Night"] >= median_sleep, "Attendance (%)"].dropna()
        low = dff.loc[dff["Sleep_Hours_per_Night"] < median_sleep, "Attendance (%)"].dropna()
        if len(high) > 1 and len(low) > 1:
            t, p = stats.ttest_ind(high, low, nan_policy="omit")
            rows.append({"test": "T-Test", "question": "High vs low sleep impacts attendance?", "metric": "Attendance (%)", "p_value": float(p)})

    if not rows:
        st.info("Not enough data to run tests with the current filters.")
        return

    out = pd.DataFrame(rows)
    out["significant_(p<0.05)"] = out["p_value"] < 0.05
    st.dataframe(out.sort_values("p_value"), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
    st.title("Student Performance EDA (Streamlit)")
    st.caption("This app mirrors the structure and key analyses from `challenge-student-performance.ipynb`.")

    with st.sidebar:
        st.caption("Data source: `Students_Grading_Dataset.csv`")
        csv_path = st.text_input("CSV path", value=str(DATA_PATH_DEFAULT))
        st.divider()
        st.subheader("Notebook cleaning options")
        apply_gender_correction = st.checkbox("Apply name-based gender correction", value=True)
        standardize_categories = st.checkbox("Standardize categorical values", value=True)
        clip_outliers = st.checkbox("Clip numeric outliers (5th–95th pct)", value=True)
        st.divider()

    df_raw = load_raw_data(csv_path)
    df_cleaned, clean_report = clean_data(
        df_raw,
        apply_gender_correction=apply_gender_correction,
        standardize_categories=standardize_categories,
        clip_outliers=clip_outliers,
    )
    # Keep a full cleaned version (with names) for preview only,
    # but use a de-identified version for analysis (matches notebook intent).
    df_full_preview = df_cleaned
    df = drop_irrelevant_columns(df_cleaned)

    f = sidebar_filters(df)
    dff = apply_filters(df, f)
    dff_full_preview = apply_filters(df_full_preview, f)

    section = st.sidebar.radio(
        "Sections",
        options=[
            "Overview",
            "Cleaning",
            "EDA",
            "Conclusion",
        ],
    )

    if section == "Overview":
        tab_intro, tab_overview, tab_dictionary = st.tabs(["Intro", "Data overview", "Data dictionary"])
        with tab_intro:
            st.markdown(
                """
### Problem statement
We explore a **5,000-student** grading dataset to understand **performance patterns** and how factors like
**attendance, study habits, and well-being** relate to outcomes (especially **Grade**).
"""
            )
            kpis(dff)
            st.markdown("### Quick preview")
            include_names = st.checkbox("Include First/Last name in preview", value=True)
            preview_df = dff_full_preview if include_names else dff
            st.dataframe(preview_df.head(25), use_container_width=True)

        with tab_overview:
            st.markdown("### Data overview (with optional filters)")
            use_filters = st.checkbox("Apply current sidebar filters to this overview", value=True)
            raw_view = apply_filters(df_raw, f) if use_filters else df_raw
            cleaned_view = apply_filters(df_full_preview, f) if use_filters else df_full_preview
            analysis_view = apply_filters(df, f) if use_filters else df

            st.markdown("### Dataset comparison: before vs after cleaning")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Before cleaning (raw)")
                st.write(f"**Rows:** {raw_view.shape[0]:,}  |  **Columns:** {raw_view.shape[1]:,}")
                st.dataframe(raw_view.head(15), use_container_width=True)
            with c2:
                st.markdown("#### After cleaning (cleaned)")
                st.write(f"**Rows:** {cleaned_view.shape[0]:,}  |  **Columns:** {cleaned_view.shape[1]:,}")
                st.dataframe(cleaned_view.head(15), use_container_width=True)

            st.markdown("#### Missing values per column (raw vs cleaned)")
            missing_compare = (
                pd.concat(
                    [
                        raw_view.isna().sum().rename("missing_raw"),
                        cleaned_view.isna().sum().rename("missing_cleaned"),
                    ],
                    axis=1,
                )
                .fillna(0)
                .astype(int)
                .sort_values(["missing_raw", "missing_cleaned"], ascending=False)
            )
            st.dataframe(missing_compare, use_container_width=True)

            st.markdown("### Analysis dataset (de-identified)")
            st.caption("This is the cleaned dataset **after dropping personal identifiers** (used for charts/tests).")
            st.write(f"**Rows:** {analysis_view.shape[0]:,}  |  **Columns:** {analysis_view.shape[1]:,}")
            st.dataframe(analysis_view.select_dtypes(include='number').describe().T, use_container_width=True)

        with tab_dictionary:
            st.markdown("### Column meanings (from the notebook)")
            dd = (
                pd.DataFrame([{"column": c, "meaning": COLUMN_DESCRIPTIONS.get(c, "")} for c in df_raw.columns])
                .sort_values("column")
                .reset_index(drop=True)
            )
            st.dataframe(dd, use_container_width=True)

    elif section == "Cleaning":
        st.markdown("### Cleaning steps (from the notebook)")
        st.markdown(
            """
- **Missing values**:
  - `Attendance (%)` → filled with **mean**
  - `Assignments_Avg` → filled with **median**
  - `Parent_Education_Level` → filled with **"Not Reported"**
- **Duplicates**: checked by `Student_ID`
- **Gender correction**: name-based mapping on `First_Name` for a small set of common names (as in the notebook)
- **Categorical standardization**: invalid values → **"Not Reported"**
- **Outliers**: clipped at **5th–95th percentiles** (precaution)
"""
        )

        st.markdown("### Cleaning report (this run)")
        st.json(clean_report)

        st.markdown("### What we drop before analysis")
        st.caption("The notebook drops personal identifiers: `Student_ID`, `First_Name`, `Last_Name`, `Email`.")

        with st.expander("Preview cleaned & de-identified data"):
            st.dataframe(df.head(50), use_container_width=True)

    elif section == "EDA":
        tab_global, tab_per_col, tab_corr, tab_qa = st.tabs(["Global EDA", "Per-column EDA", "Correlation & tests", "Insightful Q&A"])

        with tab_global:
            st.markdown("### Global EDA")
            st.markdown("#### Target focus: `Grade`")
            grade_distribution(dff)
            st.markdown("#### Grades vs attendance")
            grade_vs_attendance(dff)
            st.markdown("#### Grades vs total score")
            grade_vs_total_score(dff)
            st.markdown("#### Attendance vs total score")
            attendance_vs_performance(dff)
            st.markdown("#### Department analysis")
            department_violin(dff)
            heatmap_grade_by_department(dff)

        with tab_per_col:
            st.markdown("### Per-column EDA (notebook-style)")
            st.caption("Choose a column to see its notebook summary + charts.")
            notebook_eda_columns = [
                "Grade",
                "Gender",
                "Age",
                "Department",
                "Total_Score",
                "Final_Score",
                "Study_Hours_per_Week",
                "Extracurricular_Activities",
                "Internet_Access_at_Home",
                "Parent_Education_Level",
                "Family_Income_Level",
                "Stress_Level (1-10)",
                "Sleep_Hours_per_Night",
            ]
            available = [c for c in notebook_eda_columns if c in dff.columns]
            chosen = st.selectbox("Choose a column", options=available, index=0)
            per_column_eda(dff, chosen)

        with tab_corr:
            st.markdown("### Correlation & hypothesis tests")
            correlations_heatmap(dff)
            top_correlations_with_target(dff, target="Total_Score")
            st.divider()
            notebook_hypothesis_tests(dff)

        with tab_qa:
            insight_questions_section(dff)

    else:  # Conclusion
        st.markdown(
            """
### 📌 6️⃣ Conclusion & Final insights
### ✨ Complete Final Summary of Analysis ✨

---

**1️⃣ Grade Analysis**

**📊 Grade Distribution**
- **A**: 🟢 1495 students (**29.9%**)
- **B**: 🟡 978 students (**19.6%**)
- **C**: 🟠 794 students (**15.9%**)
- **D**: 🔴 889 students (**17.8%**)
- **F**: ⚫ 844 students (**16.9%**)

**📈 Attendance & Grades Relationship**
- **Attendance has a strong positive correlation with grades** (**r = 0.57**).
- **Grade Distribution by Attendance:**
  - **≥90% Attendance:** 🟢 **A (68.2%)** | 🟡 **B (31.8%)** (No C, D, or F)
  - **≥80% Attendance:** 🟢 **A (56.45%)** | 🟡 **B (27.25%)**
  - **<60% Attendance:** 🟠 **C (27.96%)** | 🔴 **D (42.07%)** | ⚫ **F (29.96%)**
- **Final Score and Midterm Scores have weak predictive power on grades** (**r < 0.03**).

**🔍 Key Takeaway**
✔️ **Attendance is the strongest predictor of grades, with clear thresholds at 60%, 80%, and 90%.**  
✔️ **Students with high attendance rarely fail, while low attendance is strongly associated with lower grades.**  
🚀 **Final Insight: Want an A? Keep attendance ≥90%!**

---

**2️⃣ Gender Analysis**
- **👥 Gender Distribution:** **62.06% Male | 37.94% Female**
- **📊 Performance Trends:**
  - **📌 Females:** **Slightly better in participation scores and sleep more.**
  - **📌 Males:** **Marginal edge in quizzes and project scores.**
- **📑 Statistical Tests:** No significant differences in academic or well-being metrics by gender.

---

**3️⃣ Age Analysis**
- **📊 Age Range:** **18 to 24 years** (fairly uniform representation).
- **📌 Key Findings:**
  - **🔹 Age 18:** **Higher attendance (75.87%).**
  - **🔹 Ages 22 & 23:** **Higher Grade A rates.**
- **📑 ANOVA Tests:** No significant variations in academic/well-being outcomes by age.

---

**4️⃣ Department Analysis**
- **🏫 Department Distribution:**
  - **📌 CS:** **40.44%**
  - **📌 Engineering:** **29.38%**
  - **📌 Business:** **20.12%**
  - **📌 Mathematics:** **10.06%**
- **📊 Performance Trends:**
  - **📌 Engineering:** **Highest A grade rates (32.27%) and top attendance (76.08%).**
  - **📌 Mathematics:** **Excels in projects but reports higher stress.**
- **📑 Statistical Tests:** Negligible differences between departments in academic metrics.

---

**5️⃣ Performance Metrics**

**📈 Total Score**
- **📊 Range:** **50.02 to 99.99** | **Mean:** **75.12**
- **📌 Correlations:**
  - **Attendance:** **-0.018**
  - **Grades:** **-0.023**
- **🧐 Insight:** Does not align with traditional academic evaluation systems.

**📉 Final Score**
- **📊 Mean:** **69.64**
- **📌 Correlations:**
  - **Attendance:** **-0.024**
  - **Grade:** **0.036 (weak correlation)**
- **🧐 Insight:** **Final score is not strongly related to grade, indicating differences in evaluation criteria.**

**📚 Study Hours per Week**
- **📊 Mean:** **17.66 hours**
- **📌 Findings:**
  - **Longer study hours ≠ higher scores.**
  - **High-income students study slightly more (18.06 hours).**

---

**6️⃣ Extracurricular Activities**
- **🏆 Participation Rate:** **30.2%**
- **📌 Findings:**
  - **Slight benefit in reducing stress levels.**
  - **No significant academic impact.**

---

**7️⃣ Internet Access at Home**
- **🌐 Access Rate:** **89.5%**
- **📌 Findings:**
  - **Negligible relation to attendance, performance, or other academic metrics.**

---

**8️⃣ Parent Education Level**
- **📊 Largest Category:** **"Not Reported" (35.6%)**
- **📌 Findings:**
  - **Slight link between higher parental education and Grade A outcomes.**
  - **Weak statistical impact overall.**

---

**9️⃣ Family Income**
- **📊 Grade A Rates:**
  - **Low-income students:** **30.2% (slightly higher)**
- **📌 Findings:**
  - **No tangible disadvantages in digital access or academic performance.**

---

**🔟 Stress and Sleep Analysis**

**📉 Stress Levels**
- **📊 Range:** **1-10** | **Mean:** **5.48**
- **📌 Findings:**
  - **Negligible impact on academic performance.**
  - **Higher stress (8-10) correlates with more Grade A outcomes.**

**🛌 Sleep Hours per Night**
- **📊 Mean:** **6.48 hours**
- **📌 Findings:**
  - **Consistent across demographics, income, departments, and grades.**

---

### 📌 Key Takeaways
✔️ **Attendance is the primary driver of academic success.**  
✔️ **Clear thresholds (60%, 80%, 90%) define grade patterns.**  
✔️ **Final score is weakly related to grade (correlation of 0.036), suggesting differences in grading methodology.**  
✔️ **Gender, internet access, parent education, and sleep have minimal impact on academic performance.**  
✔️ **Stress slightly affects Grade A outcomes at higher levels.**  
✔️ **Total and Final scores show weak correlation with attendance or grading, questioning traditional evaluation methods.**  
✔️ **Consistency across age, department, and family backgrounds highlights a need to explore non-measurable performance factors.**

### 📌 Practical Steps for Improvement
✔️ **Boost Attendance: Since students with high attendance perform better, schools could offer rewards or include participation in grading.**

✔️ **Encourage Smart Study Habits: Students should aim for 10-20 study hours per week to slightly improve their grades.**

✔️ **Support Low-Income Students: Free tutoring and study materials can help ensure financial struggles don’t hold students back.**

✔️ **Focus on Computer Science Students: Since CS has the most students, extra support and personalized learning could improve their success.**

✔️ **Balance Academics & Activities: While extracurriculars don’t strongly impact grades, structured programs that mix learning with social activities can be beneficial.**

🚀 **Final Tip: Keep attendance above 90% for the best chances of success!**

**Thanks!**
"""
        )


if __name__ == "__main__":
    main()

