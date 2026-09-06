import shap


def explain_prediction(model, X_row, X_background=None, top_n=3):
    """
    Explain a single prediction using SHAP TreeExplainer.

    Uses tree_path_dependent perturbation so XGBoost models containing
    categorical splits are supported.

    Returns the top_n features by absolute SHAP contribution.
    Returns None if explanation fails.
    """
    try:
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )

        shap_values = explainer.shap_values(X_row)

        if isinstance(shap_values, list):
            values = shap_values[0][0]
        else:
            values = shap_values[0]

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return None

    contributions = list(zip(X_row.columns, values))
    contributions.sort(key=lambda pair: abs(float(pair[1])), reverse=True)

    return contributions[:top_n]