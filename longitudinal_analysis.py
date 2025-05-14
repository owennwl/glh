#!/usr/bin/env python3
"""
Longitudinal Mediation Analysis
-------------------------------
This script performs longitudinal mediation analysis for two types of pathways:
1. Initial Mobility → Change in Mental Health → Later Mobility
2. Initial Mental Health → Change in Mobility → Later Mental Health

Usage: python longitudinal_analysis.py --input-file data.xlsx --output-file results.xlsx
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import zscore
from statsmodels.regression.linear_model import WLS
from statsmodels.api import add_constant
import statsmodels.formula.api as smf
from tqdm import tqdm


def format_pvalue(p):
    """Format p-values to 2 significant figures, unless p < 0.0001"""
    try:
        p_float = float(p)
        if p_float < 0.0001:
            return "<0.0001"
        else:
            return f"{p_float:.2g}"
    except (ValueError, TypeError):
        return str(p)


def analyze_mediation_pathway(
    df,
    initial_vars,
    mediator_vars,
    outcome_vars,
    initial_type,
    mediator_type,
    outcome_type,
):
    """
    Generic function to analyze how initial variables affect mediator changes,
    which in turn affect outcome variables.

    Parameters:
    -----------
    df : DataFrame
        The dataset containing all variables
    initial_vars : list
        Variables that serve as predictors
    mediator_vars : list
        Variables that serve as mediators
    outcome_vars : list
        Variables that serve as outcomes
    initial_type : str
        Description of initial variable type (e.g., "Mobility")
    mediator_type : str
        Description of mediator variable type (e.g., "Mental Health")
    outcome_type : str
        Description of outcome variable type (e.g., "Mobility")

    Returns:
    --------
    DataFrame containing mediation analysis results
    """
    time_pairs = [(1, 2), (2, 3)]
    results = []
    n_bootstraps = 10000

    # Create descriptive names for variables
    mh_name = {
        "phq9": "PHQ9 (Depression)",
        "gad7": "GAD7 (Anxiety)",
        "pcl5": "PCL5 (PTSD)",
    }
    mobility_name = {
        "journey_diversity": "Journey Diversity",
        "immobility": "Immobility",
        "remoteness": "Remoteness",
    }

    # Generic variable naming helper function
    def get_name(var_type, var):
        if var_type == "Mental Health":
            return mh_name.get(var, var)
        else:  # Mobility
            return mobility_name.get(var, var)

    for time_pair in time_pairs:
        t1, t2 = time_pair
        for mediator_var in mediator_vars:
            # Filter rows where there are non-NaN values for both time points and weight
            df_filtered = df.dropna(
                subset=[f"{mediator_var}_{t1}", f"{mediator_var}_{t2}", "weight"]
            ).copy()

            # Calculate intercepts and slopes for mediator variables
            intercepts = []
            slopes = []
            for _, row in df_filtered.iterrows():
                y = row[[f"{mediator_var}_{t1}", f"{mediator_var}_{t2}"]].astype(float)
                x = np.array([t1, t2])
                slope, intercept = np.polyfit(x, y, 1)
                intercepts.append(intercept)
                slopes.append(slope)

            # Standardize intercepts and slopes
            df_filtered.loc[:, f"{mediator_var}_intercept"] = zscore(intercepts)
            df_filtered.loc[:, f"{mediator_var}_slope"] = zscore(slopes)

            for initial_var in initial_vars:
                for outcome_var in outcome_vars:
                    # Only analyze where initial and outcome variables are the same if mobility->mh->mobility
                    # or if mental health->mobility->mental health
                    if (
                        initial_type == outcome_type == "Mobility"
                        and initial_var != outcome_var
                    ) or (
                        initial_type == outcome_type == "Mental Health"
                        and initial_var != outcome_var
                    ):
                        continue

                    # Print current model being analyzed
                    print(
                        f"Analyzing: Initial {get_name(initial_type, initial_var)} → "
                        f"Change in {get_name(mediator_type, mediator_var)} → "
                        f"Later {get_name(outcome_type, outcome_var)} (Time {t1} to {t2})"
                    )

                    # Filter for valid data in initial and outcome variables
                    valid_df = df_filtered.dropna(
                        subset=[f"{initial_var}_{t1}", f"{outcome_var}_{t2}"]
                    ).copy()

                    # Skip if no valid data
                    if valid_df.empty:
                        print("  No valid data for this combination, skipping.")
                        continue

                    # Standardize initial and outcome variables
                    valid_df.loc[:, f"{initial_var}_{t1}_std"] = zscore(
                        valid_df[f"{initial_var}_{t1}"]
                    )
                    valid_df.loc[:, f"{outcome_var}_{t2}_std"] = zscore(
                        valid_df[f"{outcome_var}_{t2}"]
                    )

                    # Path A: Initial -> Change in mediator
                    X = valid_df[
                        [f"{mediator_var}_intercept", f"{initial_var}_{t1}_std"]
                    ]
                    X = add_constant(X)
                    y1 = valid_df[f"{mediator_var}_slope"]
                    model1 = WLS(
                        y1, X, missing="drop", weights=valid_df["weight"]
                    ).fit()

                    # Path B: Change in mediator -> Later outcome
                    X = valid_df[[f"{mediator_var}_intercept", f"{mediator_var}_slope"]]
                    X = add_constant(X)
                    y2 = valid_df[f"{outcome_var}_{t2}_std"]
                    model2 = WLS(
                        y2, X, missing="drop", weights=valid_df["weight"]
                    ).fit()

                    # Direct effect: Initial -> Later outcome
                    direct_model = smf.wls(
                        f"{outcome_var}_{t2}_std ~ {initial_var}_{t1}_std",
                        data=valid_df,
                        weights=valid_df["weight"],
                    ).fit()

                    # Bootstrap to calculate indirect effect confidence intervals
                    indirect_effects = []
                    for _ in tqdm(range(n_bootstraps), desc="Bootstrap", leave=False):
                        sample_df = valid_df.sample(n=len(valid_df), replace=True)

                        # Path A bootstrap
                        X_sample = sample_df[
                            [f"{mediator_var}_intercept", f"{initial_var}_{t1}_std"]
                        ]
                        X_sample = add_constant(X_sample)
                        y_sample = sample_df[f"{mediator_var}_slope"]
                        model1_sample = WLS(
                            y_sample, X_sample, weights=sample_df["weight"]
                        ).fit()

                        # Path B bootstrap
                        X_sample = sample_df[
                            [f"{mediator_var}_intercept", f"{mediator_var}_slope"]
                        ]
                        X_sample = add_constant(X_sample)
                        y_sample = sample_df[f"{outcome_var}_{t2}_std"]
                        model2_sample = WLS(
                            y_sample, X_sample, weights=sample_df["weight"]
                        ).fit()

                        # Indirect effect calculation
                        indirect_effect = (
                            model1_sample.params[f"{initial_var}_{t1}_std"]
                            * model2_sample.params[f"{mediator_var}_slope"]
                        )
                        indirect_effects.append(indirect_effect)

                    # Create descriptive model name
                    model_description = (
                        f"Initial {get_name(initial_type, initial_var)} → "
                        f"Change in {get_name(mediator_type, mediator_var)} → "
                        f"Later {get_name(outcome_type, outcome_var)}"
                    )

                    # Collect results
                    results.append(
                        {
                            "Time_Pair": f"{t1}_{t2}",
                            "N": len(valid_df),
                            "Conf_Index": (
                                initial_var
                                if initial_type == "Mobility"
                                else mediator_var
                            ),
                            "MH_Index": (
                                mediator_var
                                if mediator_type == "Mental Health"
                                else initial_var
                            ),
                            "Model": model_description,
                            "Model_Type": f"{initial_type} → {mediator_type} → {outcome_type}",
                            "PathA_Coef_Std": round(
                                model1.params[f"{initial_var}_{t1}_std"], 2
                            ),
                            "PathA_p_Std": format_pvalue(
                                float(
                                    model1.pvalues[f"{initial_var}_{t1}_std"]
                                )  # Ensure it's a float
                            ),
                            "PathB_Coef_Std": round(
                                model2.params[f"{mediator_var}_slope"], 2
                            ),
                            "PathB_p_Std": format_pvalue(
                                float(
                                    model2.pvalues[f"{mediator_var}_slope"]
                                )  # Ensure it's a float
                            ),
                            "Direct_Effect_Coef_Std": round(
                                direct_model.params[f"{initial_var}_{t1}_std"], 2
                            ),
                            "Direct_Effect_p_Std": format_pvalue(
                                float(
                                    direct_model.pvalues[f"{initial_var}_{t1}_std"]
                                )  # Ensure it's a float
                            ),
                            "Indirect_Effect_Mean_Std": round(
                                np.mean(indirect_effects), 2
                            ),
                            "CI_Lower_Std": round(
                                np.percentile(indirect_effects, 2.5), 5
                            ),
                            "CI_Upper_Std": round(
                                np.percentile(indirect_effects, 97.5), 5
                            ),
                        }
                    )

    return pd.DataFrame(results)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Longitudinal Mediation Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to input Excel file containing longitudinal data",
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to save the output Excel file with results",
    )

    parser.add_argument(
        "--bootstraps",
        type=int,
        default=10000,
        help="Number of bootstrap samples for confidence intervals",
    )

    parser.add_argument(
        "--mobility-vars",
        nargs="+",
        default=["journey_diversity", "immobility", "remoteness"],
        help="Mobility variables to analyze",
    )

    parser.add_argument(
        "--mh-vars",
        nargs="+",
        default=["phq9", "gad7", "pcl5"],
        help="Mental health variables to analyze",
    )

    return parser.parse_args()


def main():
    """Main function to run the analysis"""
    # Parse command-line arguments
    args = parse_arguments()

    try:
        df = pd.read_excel(args.input_file)

        # Define variable lists
        mobility_vars = args.mobility_vars
        mh_vars = args.mh_vars

        print("\n=== Running Mobility → Mental Health → Mobility pathways ===")
        results_mob_mh_mob = analyze_mediation_pathway(
            df,
            initial_vars=mobility_vars,
            mediator_vars=mh_vars,
            outcome_vars=mobility_vars,
            initial_type="Mobility",
            mediator_type="Mental Health",
            outcome_type="Mobility",
        )

        print("\n=== Running Mental Health → Mobility → Mental Health pathways ===")
        results_mh_mob_mh = analyze_mediation_pathway(
            df,
            initial_vars=mh_vars,
            mediator_vars=mobility_vars,
            outcome_vars=mh_vars,
            initial_type="Mental Health",
            mediator_type="Mobility",
            outcome_type="Mental Health",
        )

        # Combine results
        all_results = pd.concat(
            [results_mob_mh_mob, results_mh_mob_mh], ignore_index=True
        )

        # Save results to Excel
        all_results.to_excel(args.output_file, index=False)

    except Exception as e:
        print(f"Error: {e}")
        import sys
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(4)
    main()
