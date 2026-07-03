import pandas as pd
import os
from datetime import datetime
import sweetviz as sv
import pandera.pandas as pa

# Schema defining validation rules for the loan dataset
# Applied to raw data BEFORE transformation and encoding
loan_schema = pa.DataFrameSchema(
    {
    " cibil_score": pa.Column(int, pa.Check.between(300, 900)),    # credit score range in India
    " income_annum": pa.Column(int, pa.Check.gt(0)),               # income must be positive
    " loan_amount": pa.Column(int, pa.Check.gt(0)),                # loan amount must be positive
    " loan_term": pa.Column(int, pa.Check.gt(0)),                  # term in months, must be positive
    " loan_status": pa.Column(str, pa.Check.isin([" Approved", " Rejected"])),  # only valid statuses
    " no_of_dependents": pa.Column(int, pa.Check.ge(0)),           # dependents can be zero
}
)

def ingest_data(filepath):
    """
    Ingests raw data from csv file.

    Parameters:
        filepath(str): path to the csv file

    Returns: 
        raw DataFrame or None if file not found     
    """
    try:
        df= pd.read_csv(filepath)
        print(f"[INGEST] data loaded sccessfully. Shape: {df.shape}")
        print(f"[INGEST] Timestamp: {datetime.now()}")
        return df
    except FileNotFoundError:
        print(f"[INGEST] ERROR: File '{filepath} not found.")
        return None
    

def validate_data(df):
    """
    Validates the incoming data against the loan schema.
    Parameters:
        df: raw DataFrame from ingest_data
    Returns:
        DataFrame if valid, None if validation fails
    """
    print(f"\n[VALIDATE] Running schema validation...")

    # Check 1 — no empty dataframe
    if df.empty:
        print("[VALIDATE] FAILED: DataFrame is empty.")
        return None

    # Check 2 — validate against Pandera schema
    try:
        loan_schema.validate(df, lazy=True)
        print("[VALIDATE] All schema checks passed. Data is clean.")
        return df
    except pa.errors.SchemaErrors as e:
        print("[VALIDATE] FAILED: Schema validation errors found.")
        print(e.failure_cases) # for debugging
        return None

def transform_data(df):
    """
    Cleans and transforms the validated DataFrame.
    Parameters:
        df: validated DataFrame from validate_data
    Returns:
        transformed DataFrame ready for model training
    """
    print(f"\n[TRANSFORM] Starting transformation...")
    
    # Remove missing values
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"[TRANSFORM] Rows removed: {before - after}")
    
    # Encode categorical columns
    df[' education'] = df[' education'].map({' Graduate': 1, ' Not Graduate': 0})
    df[' self_employed'] = df[' self_employed'].map({' Yes': 1, ' No': 0})
    df[' loan_status'] = df[' loan_status'].map({' Approved': 1, ' Rejected': 0})
    
    print(f"[TRANSFORM] Transformation complete. Shape: {df.shape}")
    return df

def load_data_output(df, output_folder="phase2_output"):
    """
    Saves the transformed DataFrame to a CSV file.
    Parameters:
        df: transformed DataFrame
        output_folder: folder to save the file in (default: 'phase2_output')
    Returns:
        filepath of saved file
    """
    print(f"\n[LOAD] Saving transformed data...")
    
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transformed_data_{timestamp}.csv"
    filepath = os.path.join(output_folder, filename)
    
    df.to_csv(filepath, index=False)
    print(f"[LOAD] Data saved to: {filepath}")
    return filepath

def generate_report(df, output_folder="phase2_output"):
    """
    Generates a summary report of the pipeline run.
    Parameters:
        df: transformed DataFrame
        output_folder: folder to save the report
    Returns:
        None
    """
    print(f"\n[REPORT] Generating pipeline report...")
    
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Build report content
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_lines = [
        f"PIPELINE REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{'='*50}",
        f"Total rows processed: {len(df)}",
        f"Total columns: {len(df.columns)}",
        f"",
        f"Loan Status Distribution:",
        f"  Approved: {(df[' loan_status'] == 1).sum()}",
        f"  Rejected: {(df[' loan_status'] == 0).sum()}",
        f"",
        f"Average annual income: {df[' income_annum'].mean():,.0f}",
        f"Average loan amount: {df[' loan_amount'].mean():,.0f}",
        f"Average cibil score: {df[' cibil_score'].mean():.1f}",
        f"",
        f"Pipeline completed successfully.",
    ]
    
    # Save report
    report_path = os.path.join(output_folder, f"report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"[REPORT] Report saved to: {report_path}")
    for line in report_lines:
        print(line)

def generate_eda_report(df, output_folder="phase2_output"):
    """
    Generates an automated EDA report using sweetviz.
    Parameters:
        df: transformed DataFrame
        output_folder: folder to save the HTML report
    Returns:
        None
    """
    print(f"\n[EDA] Generating automated EDA report...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_folder, f"eda_report_{timestamp}.html")
    
    report = sv.analyze(df)
    report.show_html(report_path, open_browser=False)
    
    print(f"[EDA] Report saved to: {report_path}")
    print(f"[EDA] Open the HTML file in your browser to view the full report.")

if __name__ == "__main__":
    
    # Stage 1 — Ingest
    raw = ingest_data("loan_approval_dataset.csv")
    if raw is None:
        print("[PIPELINE] Stopped at INGEST stage.")
    
    # Stage 2 — Validate
    else:
        validated = validate_data(raw)
        if validated is None:
            print("[PIPELINE] Stopped at VALIDATE stage.")
        
        # Stage 3 — Transform
        else:
            transformed = transform_data(validated)
            if transformed is None:
                print("[PIPELINE] Stopped at TRANSFORM stage.")
            
            # Stage 4 — Load
            else:
                saved_path = load_data_output(transformed)
                
                # Stage 5 — Report
                generate_report(transformed)
                generate_eda_report(transformed)
                
                print("\n[PIPELINE] All stages completed successfully.")

# TEST BLOCK — uncomment to test schema validation with bad data
    # print("\n--- TESTING VALIDATION WITH BAD DATA ---")
    # bad_df = raw.copy()
    # bad_df[' cibil_score'] = -500
    # test_result = validate_data(bad_df)
    # if test_result is None:
    #     print("--- VALIDATION CORRECTLY REJECTED BAD DATA ---")