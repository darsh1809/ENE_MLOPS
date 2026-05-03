import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import mlflow
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))

def load_and_clean_data(filepath):
    """
    Loads data and handles missing values.
    - Drops rows with missing Description.
    - Generates unique Customer IDs for missing values.
    """
    print("Loading dataset...")
    try:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    if df is None:
        return None

    # ── Normalise column names ────────────────────────────────────────────
    # Supports both UCI Online Retail I (InvoiceNo / UnitPrice / CustomerID)
    # and UCI Online Retail II (Invoice / Price / Customer ID) variants.
    # Every downstream script always sees the canonical set defined here.
    col_aliases = {
        # Invoice number
        'InvoiceNo': 'Invoice',
        # Unit price
        'UnitPrice': 'Price',
        # Customer identifier (space variant → space-less not wanted; keep 'Customer ID')
        'CustomerID': 'Customer ID',
    }
    df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns}, inplace=True)
    print(f"Columns after normalisation: {list(df.columns)}")
    # ─────────────────────────────────────────────────────────────────────

    # 1. Drop rows with missing Description
    print(f"Original shape: {df.shape}")
    df_clean = df.dropna(subset=['Description']).copy()
    print(f"Shape after dropping missing Description: {df_clean.shape}")

    # 2. Drop rows with missing Customer ID
    # (RFM analysis requires a real customer identifier;
    #  generating synthetic IDs creates fake customers that distort clustering)
    if 'Customer ID' in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=['Customer ID'])
        df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
        dropped = before - len(df_clean)
        print(f"Dropped {dropped} rows with missing Customer ID.")
    else:
        print("Warning: 'Customer ID' column not found.")
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    print(f"Shape after duplicate removal: {df_clean.shape}")
    
    return df_clean

def feature_engineering(df):
    """
    Creates new features from Date and Transaction info.
    """
    print("Engineering features...")
    # Convert InvoiceDate to datetime
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Date components
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Day'] = df['InvoiceDate'].dt.day
        df['Hour'] = df['InvoiceDate'].dt.hour
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    
    # Transaction Features
    # Check if Quantity and Price exist
    if 'Quantity' in df.columns and 'Price' in df.columns: # Sometimes it's UnitPrice
        price_col = 'Price' if 'Price' in df.columns else 'UnitPrice'
        # Just to be safe based on common dataset variants
        if 'UnitPrice' in df.columns and 'Price' not in df.columns:
             price_col = 'UnitPrice'
        
        df['TotalPrice'] = df['Quantity'] * df[price_col]
        
        # Session Behavior: Aggregations per Invoice
        # Creates features representing the "basket" size and variety
        if 'Invoice' in df.columns and 'StockCode' in df.columns:
            # Group by Invoice to calculate session-level metrics
            invoice_stats = df.groupby('Invoice').agg(
                TransactionSize=('Quantity', 'sum'),
                UniqueItems=('StockCode', 'nunique')
            ).reset_index()
            
            # Merge back to original dataframe
            df = df.merge(invoice_stats, on='Invoice', how='left')
            
    return df

def encode_and_scale(df):
    """
    Encodes categorical variables and scales numerical ones.
    """
    print("Encoding and Scaling...")
    
    # One-Hot Encoding for Country
    if 'Country' in df.columns:
        print(f"Number of countries: {df['Country'].nunique()}")
        # Using get_dummies for simplicity; in production pipelines use OneHotEncoder
        df = pd.get_dummies(df, columns=['Country'], prefix='Country', dtype=int)
    
    # Define columns to scale
    features_to_scale = ['Quantity', 'Price', 'TotalPrice', 'TransactionSize', 'UniqueItems']
    # Adjust based on column availability (e.g. UnitPrice vs Price)
    if 'UnitPrice' in df.columns and 'Price' not in df.columns:
        features_to_scale = [f.replace('Price', 'UnitPrice') for f in features_to_scale]

    # Filter only columns that actually exist in the dataframe
    existing_scale_cols = [col for col in features_to_scale if col in df.columns]
    
    if existing_scale_cols:
        scaler = StandardScaler()
        df[existing_scale_cols] = scaler.fit_transform(df[existing_scale_cols])
        print(f"Scaled columns: {existing_scale_cols}")
    
    return df, existing_scale_cols

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 60)

    INPUT_PATH = 'data/online_retail.csv'
    OUTPUT_PATH = 'data/processed_online_retail.csv'
    
    mlflow.set_experiment("customer")
    
    with mlflow.start_run(run_name="Step2_Data_Preprocessing"):
        # Tags
        mlflow.set_tag("stage", "data_preprocessing")
        mlflow.set_tag("model_type", "none")
        mlflow.set_tag("description", "Data cleaning, feature engineering, encoding & scaling")
        
        # =============================================
        # PHASE 1: DATA CLEANING
        # =============================================
        raw_df = pd.read_csv(INPUT_PATH, encoding='ISO-8859-1')
        original_rows, original_cols = raw_df.shape
        
        processed_df = load_and_clean_data(INPUT_PATH)
        
        if processed_df is not None:
            cleaned_rows, cleaned_cols = processed_df.shape
            rows_dropped_description = original_rows - len(raw_df.dropna(subset=['Description']))
            rows_dropped_duplicates = len(raw_df.dropna(subset=['Description'])) - cleaned_rows
            missing_customerid_filled = int(raw_df['Customer ID'].isna().sum()) if 'Customer ID' in raw_df.columns else 0
            unique_customers = processed_df['Customer ID'].nunique() if 'Customer ID' in processed_df.columns else 0
            unique_invoices = processed_df['Invoice'].nunique() if 'Invoice' in processed_df.columns else 0
            unique_products = processed_df['StockCode'].nunique() if 'StockCode' in processed_df.columns else 0
            unique_countries = processed_df['Country'].nunique() if 'Country' in processed_df.columns else 0
            
            # Log Cleaning Parameters
            mlflow.log_param("input_file", INPUT_PATH)
            mlflow.log_param("output_file", OUTPUT_PATH)
            mlflow.log_param("encoding", "ISO-8859-1")
            mlflow.log_param("cleaning_steps", "drop_null_description, fill_customerid, remove_duplicates")
            
            # Log Cleaning Metrics
            mlflow.log_metric("original_rows", original_rows)
            mlflow.log_metric("original_columns", original_cols)
            mlflow.log_metric("rows_after_cleaning", cleaned_rows)
            mlflow.log_metric("rows_dropped_null_description", rows_dropped_description)
            mlflow.log_metric("rows_dropped_duplicates", rows_dropped_duplicates)
            mlflow.log_metric("missing_customerid_filled", missing_customerid_filled)
            mlflow.log_metric("unique_customers", unique_customers)
            mlflow.log_metric("unique_invoices", unique_invoices)
            mlflow.log_metric("unique_products", unique_products)
            mlflow.log_metric("unique_countries", unique_countries)
            
            print(f"\n📊 Cleaning Summary:")
            print(f"   Original: {original_rows} rows → Cleaned: {cleaned_rows} rows")
            print(f"   Dropped (null Description): {rows_dropped_description}")
            print(f"   Dropped (duplicates): {rows_dropped_duplicates}")
            print(f"   Customer IDs filled: {missing_customerid_filled}")
            
            # =============================================
            # PHASE 2: FEATURE ENGINEERING
            # =============================================
            processed_df = feature_engineering(processed_df)
            
            new_features = [col for col in processed_df.columns if col not in raw_df.columns]
            mlflow.log_param("new_features_created", ", ".join(new_features))
            mlflow.log_metric("total_features_after_engineering", len(processed_df.columns))
            
            print(f"\n🔧 Feature Engineering:")
            print(f"   New features created: {new_features}")
            
            # =============================================
            # PHASE 3: ENCODING & SCALING
            # =============================================
            processed_df, scaled_columns = encode_and_scale(processed_df)
            
            final_rows, final_cols = processed_df.shape
            mlflow.log_param("scaler_type", "StandardScaler")
            mlflow.log_param("encoding_method", "One-Hot (pd.get_dummies)")
            mlflow.log_param("scaled_columns", ", ".join(scaled_columns))
            mlflow.log_metric("final_rows", final_rows)
            mlflow.log_metric("final_columns", final_cols)
            
            print(f"\n⚙️ Encoding & Scaling:")
            print(f"   Scaled columns: {scaled_columns}")
            print(f"   Final shape: {final_rows} rows × {final_cols} columns")
            
            # =============================================
            # SAVE & LOG ARTIFACTS
            # =============================================
            processed_df.to_csv(OUTPUT_PATH, index=False)
            
            # Save preprocessing report
            report_path = "data/preprocessing_report.txt"
            with open(report_path, "w") as f:
                f.write("DATA PREPROCESSING REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Input: {INPUT_PATH}\n")
                f.write(f"Output: {OUTPUT_PATH}\n\n")
                f.write("PHASE 1 - CLEANING:\n")
                f.write(f"  Original rows: {original_rows}\n")
                f.write(f"  Rows dropped (null Description): {rows_dropped_description}\n")
                f.write(f"  Rows dropped (duplicates): {rows_dropped_duplicates}\n")
                f.write(f"  Customer IDs filled: {missing_customerid_filled}\n")
                f.write(f"  Rows after cleaning: {cleaned_rows}\n\n")
                f.write("PHASE 2 - FEATURE ENGINEERING:\n")
                f.write(f"  New features: {new_features}\n\n")
                f.write("PHASE 3 - ENCODING & SCALING:\n")
                f.write(f"  Scaler: StandardScaler\n")
                f.write(f"  Encoding: One-Hot (Country)\n")
                f.write(f"  Scaled columns: {scaled_columns}\n")
                f.write(f"  Final shape: {final_rows} rows × {final_cols} columns\n\n")
                f.write("COLUMN SUMMARY:\n")
                for col in processed_df.columns[:30]:  # First 30 to keep it readable
                    f.write(f"  {col}: {processed_df[col].dtype}\n")
                if len(processed_df.columns) > 30:
                    f.write(f"  ... and {len(processed_df.columns) - 30} more columns\n")
            
            mlflow.log_artifact(report_path)
            
            # Save sample of processed data (first 100 rows)
            sample_path = "data/processed_sample.csv"
            processed_df.head(100).to_csv(sample_path, index=False)
            mlflow.log_artifact(sample_path)
            
            print(f"\n✅ All preprocessing results logged to MLflow!")
            print(f"   📄 Artifacts: {report_path}, {sample_path}")
            print(f"   💾 Processed data saved to {OUTPUT_PATH}")
        else:
            print("❌ Data processing failed due to loading error.")

