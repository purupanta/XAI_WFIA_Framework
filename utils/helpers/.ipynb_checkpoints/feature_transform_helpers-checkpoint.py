# utils/helpers/feature_transform_helpers.py

__all__ = [
    "show_table",
    "remove_null_rows",
    "get_feature_alias_by_category",
    "enrich_with_alias_and_category",
    "reorder_columns"
]


from IPython.display import display, HTML

def show_table(df, caption=None, fmt=None):
    """Display a DataFrame with optional caption and per-column fmt, without using .style."""
    if caption:
        display(HTML(f"<h4 style='margin:6px 0'>{caption}</h4>"))
    if fmt:
        df_disp = df.copy()
        for col, f in fmt.items():
            if col in df_disp.columns:
                df_disp[col] = df_disp[col].map(lambda x: f.format(x))
    else:
        df_disp = df
    display(df_disp)
    

def remove_null_rows(df, verbose=0):
    import pandas as pd
    # Columns to check
    cols = df.columns
    cols = ['COHORT', 'VISIT_NUM', 'PATIENT_NUM']

    print(f">> Removing null/blank rows <<")
    print(f"Consideration cols: {cols}")

    # Treat blanks/whitespace as NaN
    df = df.replace(r'^\s*$', pd.NA, regex=True)

    # Drop rows where ANY of the cols are missing
    df_clean = df.dropna(subset=cols)

    if verbose > 0:
        print(f"Rows before: {len(df)}")
        print(f"Rows after:  {len(df_clean)}")
        print(f"Rows deleted: {len(df) - len(df_clean)}")

    return df_clean

    import pandas as pd

# --- 1) Category → alias mapping (edit as needed) ---
def get_feature_alias_by_category():
    return {
        "Demographics": {
            "Age": "Age (Yrs)",
            "BirthGender": "Biological Sex",
        },
        "Clinical Health": {
            "BMI": "Body Mass Index",
            "Deaf": "Hearing Impaired",
            "EverHadCancer": "Ever Diagnosed with Cancer",
            "GeneralHealth_Excellent": "Excellent General Health",
            "GeneralHealth_Poor": "Poor General Health",
            "GeneralHealth_Fair": "Fair General Health",
            "GeneralHealth_Good": "Good General Health",
            "GeneralHealth_VeryGood": "Very Good General Health",
            "MedConditions_Depression": "Diagnosed Depression",
            "MedConditions_Diabetes": "Diagnosed Diabetes",
            "MedConditions_HighBP": "Diagnosed High Blood Pressure",
            "MedConditions_LungDisease": "Diagnosed Lung Disease",
            "PHQ4": "PHQ-4 (Anxiety/Depression Score)",
            "MedConditions_HeartCondition": "Diagnosed Heart Condition",
        },
        "Health Behaviors": {
            "AverageSleepNight": "Average Sleep (hrs/night)",
            "AverageTimeSitting": "Avg Daily Sitting Time",
            "FreqGoProvider": "Number of Doctor Visits",
            "WeeklyMinutesModerateExercise": "Moderate Exercise Minutes/Week",
        },
        "Substance Use": {
            "AvgDrinksPerWeek": "Average Drinks/Week",
            "eCigUse_Current": "Currently Uses E-Cigarettes",
            "eCigUse_Former": "Former E-Cigarette User",
            "eCigUse_Never": "Never Used E-Cigarettes",
            "smokeStat_Current": "Currently Smokes (Cigarettes)",
            "smokeStat_Former": "Former Smoker",
            "smokeStat_Never": "Never Smoked",
        },
    }

# --- 2) Enrich with alias & category ---
def enrich_with_alias_and_category(df, feature_alias_by_category, feature_col="feature"):
    alias_map = {raw: alias
                 for group in feature_alias_by_category.values()
                 for raw, alias in group.items()}
    category_map = {raw: cat
                    for cat, group in feature_alias_by_category.items()
                    for raw in group.keys()}
    out = df.copy()
    out["alias"] = out[feature_col].map(alias_map).fillna(out[feature_col])
    out["category"] = out[feature_col].map(category_map).fillna("Other")
    return out

# --- 3) Reorder columns (keeps only those that exist) ---
def reorder_columns(df):
    wanted = [
        "feature", "category", "alias",
        "mean_abs_shap_xgb", "mean_shap_xgb",
        "mean_abs_shap_tabnet", "mean_shap_tabnet",
        "mean_abs_shap_lr", "mean_shap_lr",
    ]
    cols = [c for c in wanted if c in df.columns]
    # keep any extra columns at the end
    extras = [c for c in df.columns if c not in cols]
    return df[cols + extras]


