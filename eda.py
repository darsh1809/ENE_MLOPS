import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import mlflow

from data_preprocessing import load_and_clean_data, feature_engineering

mlflow.set_tracking_uri("http://localhost:5000")

# Set style
sns.set(style="whitegrid")

def create_output_dir(directory='data/eda_plots'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# =============================================
# EXISTING GRAPHS (4)
# =============================================

def plot_monthly_sales(df, output_dir):
    """Graph 1: Monthly Sales Trend (Line Chart)"""
    print("📈 [1/10] Generating Monthly Sales Trend...")
    monthly_sales = df.groupby(['Year', 'Month'])['TotalPrice'].sum().reset_index()
    monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_sales, x='Date', y='TotalPrice', marker='o', color='#2196F3')
    plt.title('Monthly Sales Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Total Sales (£)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_sales_trend.png'), dpi=150)
    plt.close()

def plot_top_countries(df, output_dir):
    """Graph 2: Top 10 Countries by Revenue (Horizontal Bar)"""
    print("📊 [2/10] Generating Top 10 Countries by Revenue...")
    country_sales = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=country_sales, x='TotalPrice', y='Country', palette='viridis')
    plt.title('Top 10 Countries by Revenue', fontsize=14, fontweight='bold')
    plt.xlabel('Total Revenue (£)')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_10_countries_revenue.png'), dpi=150)
    plt.close()

def plot_top_products(df, output_dir):
    """Graph 3: Top 10 Products by Quantity (Horizontal Bar)"""
    print("📊 [3/10] Generating Top 10 Products by Quantity...")
    product_sales = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=product_sales, x='Quantity', y='Description', palette='magma')
    plt.title('Top 10 Products by Quantity Sold', fontsize=14, fontweight='bold')
    plt.xlabel('Total Quantity Sold')
    plt.ylabel('Product')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_10_products_quantity.png'), dpi=150)
    plt.close()

def plot_hourly_sales(df, output_dir):
    """Graph 4: Hourly Sales Heatmap (Day vs Hour)"""
    print("🔥 [4/10] Generating Hourly Sales Heatmap...")
    pivot = df.pivot_table(index='DayOfWeek', columns='Hour', values='Invoice', aggfunc='nunique')
    
    days = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    pivot.index = pivot.index.map(days)
    order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot = pivot.reindex([d for d in order if d in pivot.index])
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f', linewidths=0.5)
    plt.title('Hourly Sales Heatmap (Number of Invoices)', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hourly_sales_heatmap.png'), dpi=150)
    plt.close()

# =============================================
# EXTRA GRAPHS (6) - Added for mentor
# =============================================

def plot_revenue_distribution(df, output_dir):
    """Graph 5: Revenue per Transaction Distribution (Histogram)"""
    print("📊 [5/10] Generating Revenue Distribution...")
    # Filter reasonable range to avoid extreme outliers
    revenue_data = df['TotalPrice'][(df['TotalPrice'] > 0) & (df['TotalPrice'] < df['TotalPrice'].quantile(0.99))]
    
    plt.figure(figsize=(12, 6))
    plt.hist(revenue_data, bins=50, color='#4CAF50', edgecolor='white', alpha=0.8)
    plt.title('Revenue per Transaction Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Transaction Value (£)')
    plt.ylabel('Frequency')
    plt.axvline(revenue_data.mean(), color='red', linestyle='--', label=f'Mean: £{revenue_data.mean():.2f}')
    plt.axvline(revenue_data.median(), color='orange', linestyle='--', label=f'Median: £{revenue_data.median():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'revenue_distribution.png'), dpi=150)
    plt.close()

def plot_sales_by_day_of_week(df, output_dir):
    """Graph 6: Sales by Day of Week (Bar Chart)"""
    print("📊 [6/10] Generating Sales by Day of Week...")
    days_map = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
    daily_sales = df.groupby('DayOfWeek')['TotalPrice'].sum().reset_index()
    daily_sales['DayName'] = daily_sales['DayOfWeek'].map(days_map)
    
    plt.figure(figsize=(10, 6))
    colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF']
    sns.barplot(data=daily_sales, x='DayName', y='TotalPrice', palette=colors)
    plt.title('Total Sales by Day of Week', fontsize=14, fontweight='bold')
    plt.xlabel('Day')
    plt.ylabel('Total Sales (£)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sales_by_day_of_week.png'), dpi=150)
    plt.close()

def plot_customer_purchase_frequency(df, output_dir):
    """Graph 7: Customer Purchase Frequency Distribution (Histogram)"""
    print("📊 [7/10] Generating Customer Purchase Frequency...")
    customer_freq = df.groupby('Customer ID')['Invoice'].nunique().reset_index()
    customer_freq.columns = ['Customer ID', 'PurchaseCount']
    
    # Cap at 50 for visualization
    freq_capped = customer_freq['PurchaseCount'].clip(upper=50)
    
    plt.figure(figsize=(12, 6))
    plt.hist(freq_capped, bins=50, color='#9C27B0', edgecolor='white', alpha=0.8)
    plt.title('Customer Purchase Frequency Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Purchases')
    plt.ylabel('Number of Customers')
    plt.axvline(customer_freq['PurchaseCount'].mean(), color='red', linestyle='--', 
                label=f'Mean: {customer_freq["PurchaseCount"].mean():.1f} purchases')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'customer_purchase_frequency.png'), dpi=150)
    plt.close()
    
    return customer_freq

def plot_monthly_customers(df, output_dir):
    """Graph 8: Monthly Active Customers (Line Chart)"""
    print("📈 [8/10] Generating Monthly Active Customers...")
    monthly_customers = df.groupby(['Year', 'Month'])['Customer ID'].nunique().reset_index()
    monthly_customers.columns = ['Year', 'Month', 'ActiveCustomers']
    monthly_customers['Date'] = pd.to_datetime(monthly_customers[['Year', 'Month']].assign(DAY=1))
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_customers, x='Date', y='ActiveCustomers', marker='s', color='#E91E63')
    plt.title('Monthly Active Customers', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Number of Unique Customers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_active_customers.png'), dpi=150)
    plt.close()

def plot_quantity_price_scatter(df, output_dir):
    """Graph 9: Quantity vs Price Scatter Plot"""
    print("📊 [9/10] Generating Quantity vs Price Scatter...")
    # Sample for performance
    sample = df[(df['Quantity'] > 0) & (df['Price'] > 0)].sample(n=min(5000, len(df)), random_state=42)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(sample['Quantity'], sample['Price'], alpha=0.3, c='#009688', s=10)
    plt.title('Quantity vs Price (Sampled)', fontsize=14, fontweight='bold')
    plt.xlabel('Quantity')
    plt.ylabel('Price (£)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quantity_vs_price_scatter.png'), dpi=150)
    plt.close()

def plot_top_customers(df, output_dir):
    """Graph 10: Top 10 Customers by Revenue (Bar Chart)"""
    print("📊 [10/10] Generating Top 10 Customers by Revenue...")
    top_customers = df.groupby('Customer ID')['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
    top_customers['Customer ID'] = top_customers['Customer ID'].astype(str)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_customers, x='TotalPrice', y='Customer ID', palette='coolwarm')
    plt.title('Top 10 Customers by Revenue', fontsize=14, fontweight='bold')
    plt.xlabel('Total Revenue (£)')
    plt.ylabel('Customer ID')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_10_customers_revenue.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    INPUT_PATH = 'data/online_retail.csv'
    OUTPUT_DIR = 'data/eda_plots'
    
    try:
        df = load_and_clean_data(INPUT_PATH)
        if df is not None:
            df = feature_engineering(df)
            output_dir = create_output_dir(OUTPUT_DIR)
            
            # MLflow Logging
            mlflow.set_experiment("customer")
            
            with mlflow.start_run(run_name="Step3_EDA"):
                # Tags
                mlflow.set_tag("stage", "eda")
                mlflow.set_tag("model_type", "none")
                mlflow.set_tag("description", "Exploratory Data Analysis with 10 visualizations")
                
                # --- LOG DATASET SUMMARY METRICS ---
                mlflow.log_metric("total_transactions", len(df))
                mlflow.log_metric("unique_customers", df['Customer ID'].nunique())
                mlflow.log_metric("unique_products", df['StockCode'].nunique())
                mlflow.log_metric("unique_countries", df['Country'].nunique())
                mlflow.log_metric("total_revenue", float(df['TotalPrice'].sum()))
                mlflow.log_metric("avg_transaction_value", float(df['TotalPrice'].mean()))
                mlflow.log_metric("median_transaction_value", float(df['TotalPrice'].median()))
                mlflow.log_metric("date_range_days", (df['InvoiceDate'].max() - df['InvoiceDate'].min()).days)
                
                # Log Parameters
                mlflow.log_param("dataset", INPUT_PATH)
                mlflow.log_param("total_graphs_generated", 10)
                mlflow.log_param("graph_list", "monthly_sales, top_countries, top_products, hourly_heatmap, revenue_dist, sales_by_day, purchase_freq, monthly_customers, qty_vs_price, top_customers")
                
                # --- GENERATE ALL 10 GRAPHS ---
                
                # Existing 4 graphs
                plot_monthly_sales(df, output_dir)
                plot_top_countries(df, output_dir)
                plot_top_products(df, output_dir)
                plot_hourly_sales(df, output_dir)
                
                # Extra 6 graphs
                plot_revenue_distribution(df, output_dir)
                plot_sales_by_day_of_week(df, output_dir)
                customer_freq = plot_customer_purchase_frequency(df, output_dir)
                plot_monthly_customers(df, output_dir)
                plot_quantity_price_scatter(df, output_dir)
                plot_top_customers(df, output_dir)
                
                # --- LOG ALL GRAPHS AS ARTIFACTS ---
                graph_files = [
                    'monthly_sales_trend.png',
                    'top_10_countries_revenue.png',
                    'top_10_products_quantity.png',
                    'hourly_sales_heatmap.png',
                    'revenue_distribution.png',
                    'sales_by_day_of_week.png',
                    'customer_purchase_frequency.png',
                    'monthly_active_customers.png',
                    'quantity_vs_price_scatter.png',
                    'top_10_customers_revenue.png'
                ]
                
                for graph_file in graph_files:
                    graph_path = os.path.join(output_dir, graph_file)
                    if os.path.exists(graph_path):
                        mlflow.log_artifact(graph_path)
                        print(f"   ✅ Logged: {graph_file}")
                
                # --- LOG EXTRA STATS ---
                # Purchase frequency stats
                if customer_freq is not None:
                    mlflow.log_metric("avg_purchases_per_customer", float(customer_freq['PurchaseCount'].mean()))
                    mlflow.log_metric("max_purchases_by_customer", int(customer_freq['PurchaseCount'].max()))
                    mlflow.log_metric("single_purchase_customers", int((customer_freq['PurchaseCount'] == 1).sum()))
                
                print(f"\n✅ EDA complete! All 10 graphs logged to MLflow.")
                print(f"   📁 Graphs saved to: {OUTPUT_DIR}")
        else:
            print("❌ Failed to load data.")
            
    except Exception as e:
        print(f"❌ An error occurred during EDA: {e}")
        import traceback
        traceback.print_exc()
