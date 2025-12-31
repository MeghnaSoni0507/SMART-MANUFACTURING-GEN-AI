import pandas as pd

from ml.preprocessing import preprocess_input_data
from ml.failure_risk_logic import predict_failure_with_explainability

def process_csv_file(df):
    results = []

    # Use dict-iteration (orient='index') which is faster than `iterrows()`
    for idx, row_dict in df.to_dict(orient="index").items():
        input_df = preprocess_input_data(pd.DataFrame([row_dict]))

        prediction = predict_failure_with_explainability(input_df)

        prediction["row_id"] = int(idx)
        results.append(prediction)

    return results
