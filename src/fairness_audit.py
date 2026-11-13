
import numpy as np
import pandas as pd

def fairness_audit(y_actual, y_predicted, groups):

    audit_data = pd.DataFrame({
        "Actual": np.asarray(y_actual),
        "Predicted": np.asarray(y_predicted),
        "Group": np.asarray(groups)
    })

    results = []

    for group in audit_data["Group"].dropna().unique():

        data = audit_data[audit_data["Group"] == group]

        tp = ((data["Actual"] == 1) & (data["Predicted"] == 1)).sum()
        fp = ((data["Actual"] == 0) & (data["Predicted"] == 1)).sum()
        fn = ((data["Actual"] == 1) & (data["Predicted"] == 0)).sum()
        tn = ((data["Actual"] == 0) & (data["Predicted"] == 0)).sum()

        results.append({
            "Group": group,
            "Selection Rate": data["Predicted"].mean(),
            "True Positive Rate": tp / (tp + fn) if tp + fn else np.nan,
            "False Positive Rate": fp / (fp + tn) if fp + tn else np.nan
        })

    return pd.DataFrame(results)
