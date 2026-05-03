import pandas as pd
import mlflow
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://54.206.46.48:5000"))

def explore_and_log_data(filepath='data/online_retail.csv'):
    """
    Step 1 of pipeline: Load raw data, check missing values, 
    log all data exploration results to MLflow 'customer' experiment.
    """
    
    # ========================================
    # 1. LOAD RAW DATASET
    # ========================================
    print("=" * 60)
    print("STEP 1: DATA EXPLORATION & MISSING VALUE CHECK")
    print("=" * 60)
    
    try:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
        print(f"✅ Dataset loaded successfully from: {filepath}")
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    # ========================================
    # 2. BASIC DATASET INFO
    # ========================================
    total_rows, total_cols = df.shape
    column_names = list(df.columns)
    dtypes_info = df.dtypes.astype(str).to_dict()
    duplicates_count = df.duplicated().sum()
    
    print(f"\n📊 Dataset Shape: {total_rows} rows × {total_cols} columns")
    print(f"📋 Columns: {column_names}")
    print(f"🔄 Duplicate Rows: {duplicates_count}")
    
    # ========================================
    # 3. MISSING VALUES ANALYSIS
    # ========================================
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df) * 100).round(2)
    total_missing = missing_values.sum()
    
    # Build a summary table
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'DataType': df.dtypes.astype(str).values,
        'Non_Null_Count': df.notnull().sum().values,
        'Missing_Count': missing_values.values,
        'Missing_Percentage': missing_percentage.values
    })
    
    print(f"\n🔍 Missing Values Summary:")
    print(missing_df.to_string(index=False))
    print(f"\n📌 Total Missing Values: {total_missing}")
    
    # ========================================
    # 4. BASIC STATISTICS
    # ========================================
    stats = df.describe()
    print(f"\n📈 Basic Statistics:")
    print(stats)
    
    # ========================================
    # 5. LOG EVERYTHING TO MLFLOW
    # ========================================
    print(f"\n📡 Logging to MLflow experiment: 'customer'")
    
    mlflow.set_experiment("customer")
    
    with mlflow.start_run(run_name="Step1_Data_Exploration"):
        # Tags
        mlflow.set_tag("stage", "data_exploration")
        mlflow.set_tag("model_type", "none")
        mlflow.set_tag("description", "Raw data loading and missing value analysis")
        
        # --- LOG PARAMETERS ---
        mlflow.log_param("dataset_path", filepath)
        mlflow.log_param("total_rows", total_rows)
        mlflow.log_param("total_columns", total_cols)
        mlflow.log_param("column_names", ", ".join(column_names))
        mlflow.log_param("duplicate_rows", duplicates_count)
        
        # Log data types for each column
        for col, dtype in dtypes_info.items():
            mlflow.log_param(f"dtype_{col}", dtype)
        
        # --- LOG METRICS ---
        mlflow.log_metric("total_missing_values", total_missing)
        mlflow.log_metric("total_rows", total_rows)
        mlflow.log_metric("total_columns", total_cols)
        mlflow.log_metric("duplicate_rows", duplicates_count)
        
        # Log missing count per column as metrics
        for col in df.columns:
            safe_col = col.replace(" ", "_")
            mlflow.log_metric(f"missing_{safe_col}", int(missing_values[col]))
            mlflow.log_metric(f"missing_pct_{safe_col}", float(missing_percentage[col]))
        
        # --- LOG ARTIFACTS ---
        # Save missing values summary as CSV
        os.makedirs("data", exist_ok=True)
        
        missing_csv_path = "data/missing_values_report.csv"
        missing_df.to_csv(missing_csv_path, index=False)
        mlflow.log_artifact(missing_csv_path)
        
        # Save basic statistics as CSV
        stats_csv_path = "data/basic_statistics.csv"
        stats.to_csv(stats_csv_path)
        mlflow.log_artifact(stats_csv_path)
        
        # Save dataset info as text file
        info_path = "data/dataset_info.txt"
        with open(info_path, "w") as f:
            f.write(f"Dataset: {filepath}\n")
            f.write(f"Shape: {total_rows} rows × {total_cols} columns\n")
            f.write(f"Duplicate Rows: {duplicates_count}\n")
            f.write(f"Total Missing Values: {total_missing}\n")
            f.write(f"\n{'='*50}\n")
            f.write(f"COLUMNS & DATA TYPES:\n")
            f.write(f"{'='*50}\n")
            for col in df.columns:
                f.write(f"  {col}: {dtypes_info[col]} | Missing: {missing_values[col]} ({missing_percentage[col]}%)\n")
            f.write(f"\n{'='*50}\n")
            f.write(f"FIRST 5 ROWS:\n")
            f.write(f"{'='*50}\n")
            f.write(df.head().to_string())
            f.write(f"\n\n{'='*50}\n")
            f.write(f"BASIC STATISTICS:\n")
            f.write(f"{'='*50}\n")
            f.write(stats.to_string())
        mlflow.log_artifact(info_path)
        
        print("✅ All data exploration results logged to MLflow!")
        print(f"   📄 Artifacts: {missing_csv_path}, {stats_csv_path}, {info_path}")
    
    return df

if __name__ == "__main__":
    df = explore_and_log_data()
    
    if df is not None:
        print("\n✅ Data exploration complete. Check MLflow UI → 'customer' experiment → 'Step1_Data_Exploration' run.")
    else:
        print("\n❌ Data exploration failed.")
