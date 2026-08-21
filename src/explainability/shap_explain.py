import shap


def explain_prediction(model, X_row, X_background, top_n=3):
    """
    Returns the top_n features contributing most to a single prediction.
    Uses shap.Explainer's automatic dispatch (TreeExplainer for RF/XGBoost,
    LinearExplainer for Ridge) rather than hardcoding which explainer to
    use for which model -- keeps this working if models are swapped later.

    X_row: single-row DataFrame being predicted on.
    X_background: reference data for the explainer (a small sample of
    training data works fine; using X_row itself is a reasonable fallback
    for tree models where no separate background set is required).

    Returns None if explanation fails, rather than raising -- a missing
    explanation shouldn't break the prediction endpoint.
    """
    try:
        explainer = shap.Explainer(model, X_background)
        shap_values = explainer(X_row)
        values = shap_values.values[0]
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return None

    contributions = list(zip(X_row.columns, values))
    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return contributions[:top_n]
