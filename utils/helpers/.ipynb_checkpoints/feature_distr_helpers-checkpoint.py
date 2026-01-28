# utils/helpers/feature_distr_helpers.py

__all__ = [
    "count01",
    "print_shapes_and_features",
    "summarize_y_distributions"
]

def count01(df, col, verbose=0):
    """Return counts of 0/1 in `col`, ignoring other codes (e.g., -9, NaN)."""
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    s = df[col]
    counts = s.loc[s.isin([0, 1])].value_counts().sort_index()
    if verbose > 0:
        total = int(counts.sum())
        print(f"Counts for column '{col}' (only 0 and 1):")
        print(counts)
        print(f"Total (0/1 only): {total}")
    return counts

def print_shapes_and_features(result, verbose = 0):
    if(verbose > 0):
        
        print(f"X_train shape:            X = {result['X_train'].shape},      y = {result['y_train'].shape}")
        print(f"X_train_res shape:        X = {result['X_train_res'].shape},  y = {result['y_train_res'].shape}")
        print(f"X_val shape:              X = {result['X_val'].shape},        y = {result['y_val'].shape}")
        print(f"X_test shape:             X = {result['X_test'].shape},       y = {result['y_test'].shape}")
        print(f"X_train_res_scaled shape: X = {result['X_train_res_scaled'].shape}")
        print(f"X_val_scaled shape:       X = {result['X_val_scaled'].shape}")
        print(f"X_test_scaled shape:      X = {result['X_test_scaled'].shape}")
        print(f"features length:          n = {len(result['features'])}")

        if(verbose > 1):
            print("feature names:")
            for i, name in enumerate(result['features']):
                print(f"  {i:>3}: {name}")

import pandas as pd
def summarize_y_distributions(result):
    out = {}
    for name, y in result.items():
        if not name.startswith("y_"):
            continue
        s = pd.Series(y).replace({True: 1, False: 0})
        s = pd.to_numeric(s, errors="coerce")  # coerce any weird types
        counts = s.value_counts().reindex([0, 1], fill_value=0)
        pct = (counts / len(s) * 100).round(2)
        out[name] = {
            "n": len(s),
            "0_count": int(counts.loc[0]),
            "0_pct": float(pct.loc[0]),
            "1_count": int(counts.loc[1]),
            "1_pct": float(pct.loc[1]),
        }
    return pd.DataFrame.from_dict(out, orient="index").sort_index()

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



