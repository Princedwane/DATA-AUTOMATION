from prefect import flow, task
import pandas as pd
import pandera.pandas as pa
from datetime import datetime
import os
import sweetviz as sv

@task(name="Ingest Data", log_prints=True)
def ingest_data(filepath):
    """
    Ingests raw data from CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[INGEST] Data loaded successfully. Shape: {df.shape}")
        print(f"[INGEST] Timestamp: {datetime.now()}")
        return df
    except FileNotFoundError:
        print(f"[INGEST] ERROR: File '{filepath}' not found.")
        return None
    
@task(name="Validate Data", log_prints=True)
def validate_data(df):
    """
    Validates incoming data against loan schema.
    """
    print(f"\n[VALIDATE] Running schema validation...")

    if df is None:
        print("[VALIDATE] FAILED: No data received.")
        return None

    if df.empty:
        print("[VALIDATE] FAILED: DataFrame is empty.")
        return None

    loan_schema = pa.DataFrameSchema({
        " cibil_score": pa.Column(int, pa.Check.between(300, 900)),
        " income_annum": pa.Column(int, pa.Check.gt(0)),
        " loan_amount": pa.Column(int, pa.Check.gt(0)),
        " loan_term": pa.Column(int, pa.Check.gt(0)),
        " loan_status": pa.Column(str, pa.Check.isin([" Approved", " Rejected"])),
        " no_of_dependents": pa.Column(int, pa.Check.ge(0)),
    })

    try:
        loan_schema.validate(df, lazy=True)
        print("[VALIDATE] All schema checks passed.")
        return df
    except pa.errors.SchemaErrors as e:
        print("[VALIDATE] FAILED: Schema validation errors found.")
        print(e.failure_cases)
        return None

@task(name="Transform Data", log_prints=True)
def transform_data(df):
    """
    Cleans and encodes the validated DataFrame.
    """
    print(f"\n[TRANSFORM] Starting transformation...")

    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"[TRANSFORM] Rows removed: {before - after}")

    df[' education'] = df[' education'].map({' Graduate': 1, ' Not Graduate': 0})
    df[' self_employed'] = df[' self_employed'].map({' Yes': 1, ' No': 0})
    df[' loan_status'] = df[' loan_status'].map({' Approved': 1, ' Rejected': 0})

    print(f"[TRANSFORM] Transformation complete. Shape: {df.shape}")
    return df

@task(name="Load Data", log_prints=True)
def load_data_output(df, output_folder="phase2_output"):
    """
    Saves transformed DataFrame to timestamped CSV.
    """
    print(f"\n[LOAD] Saving transformed data...")
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prefect_transformed_data_{timestamp}.csv"
    filepath = os.path.join(output_folder, filename)
    df.to_csv(filepath, index=False)
    print(f"[LOAD] Data saved to: {filepath}")
    return filepath


@task(name="Generate Report", log_prints=True)
def generate_report(df, output_folder="phase2_output"):
    """
    Generates a summary text report.
    """
    print(f"\n[REPORT] Generating pipeline report...")
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_lines = [
        f"PIPELINE REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{'='*50}",
        f"Total rows processed: {len(df)}",
        f"Total columns: {len(df.columns)}",
        f"Approved: {(df[' loan_status'] == 1).sum()}",
        f"Rejected: {(df[' loan_status'] == 0).sum()}",
        f"Average cibil score: {df[' cibil_score'].mean():.1f}",
        f"Pipeline completed successfully.",
    ]
    report_path = os.path.join(output_folder, f"prefect_report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[REPORT] Report saved to: {report_path}")


@task(name="Generate EDA Report", log_prints=True)
def generate_eda_report(df, output_folder="phase2_output"):
    """
    Generates automated EDA report using sweetviz.
    """
    print(f"\n[EDA] Generating automated EDA report...")
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_folder, f"eda_report_{timestamp}.html")
    report = sv.analyze(df)
    report.show_html(report_path, open_browser=False)
    print(f"[EDA] Report saved to: {report_path}")


@flow(name="Loan Pipeline")
def loan_pipeline():
    """
    Main Prefect flow — orchestrates all pipeline stages.
    """
    raw = ingest_data('loan_approval_dataset.csv')
    if raw is None:
        return

    validated = validate_data(raw)
    if validated is None:
        return

    transformed = transform_data(validated)
    if transformed is None:
        return

    load_data_output(transformed)
    generate_report(transformed)
    generate_eda_report(transformed)
    print("\n[PIPELINE] All stages completed successfully.")


if __name__ == "__main__":
    loan_pipeline()