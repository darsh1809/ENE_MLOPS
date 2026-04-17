import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import mlflow
import mlflow.sklearn
import os
from data_preprocessing import load_and_clean_data

mlflow.set_tracking_uri("http://localhost:5000")

# Set style
sns.set(style="whitegrid")

def create_output_dir(directory='data'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def rfm_analysis(df):
    """
    Performs Recency, Frequency, Monetary (RFM) analysis.
    """
    print("Performing RFM Analysis...")
    
    # Ensure InvoiceDate is datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate TotalPrice if not present
    if 'TotalPrice' not in df.columns:
        df['TotalPrice'] = df['Quantity'] * df['Price']
    
    # Set reference date as one day after the last transaction
    current_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Group by Customer ID
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (current_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalPrice': 'Monetary'
    }).reset_index()
    
    # Filter out negative or zero monetary/frequency values if any (data anomalies)
    rfm = rfm[rfm['Monetary'] > 0]
    
    return rfm

def plot_elbow_curve(X_scaled, output_dir, max_k=10):
    """Graph 1: Elbow Curve - finding optimal K"""
    print("📈 [1/5] Generating Elbow Curve...")
    inertias = []
    silhouettes = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        
        # Sample for silhouette if large dataset
        if len(X_scaled) > 10000:
            from sklearn.utils import resample
            X_sample, labels_sample = resample(X_scaled, km.labels_, n_samples=10000, random_state=42)
            sil = silhouette_score(X_sample, labels_sample)
        else:
            sil = silhouette_score(X_scaled, km.labels_)
        silhouettes.append(sil)
        print(f"   k={k}: Inertia={km.inertia_:.0f}, Silhouette={sil:.4f}")
    
    # Plot Elbow + Silhouette side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Elbow
    ax1.plot(K_range, inertias, 'bo-', markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax1.set_title('Elbow Curve', fontsize=14, fontweight='bold')
    ax1.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Chosen k=4')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Silhouette
    ax2.plot(K_range, silhouettes, 'rs-', markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score vs k', fontsize=14, fontweight='bold')
    ax2.axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Chosen k=4')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'elbow_curve.png'), dpi=150)
    plt.close()
    
    return inertias, silhouettes

def plot_cluster_pairplot(rfm_df, output_dir):
    """Graph 2: Cluster Pair Plot"""
    print("📊 [2/5] Generating Cluster Pair Plot...")
    if len(rfm_df) > 5000:
        plot_df = rfm_df.sample(n=5000, random_state=42)
    else:
        plot_df = rfm_df
    
    g = sns.pairplot(plot_df, hue='Cluster', palette='viridis', 
                     vars=['Recency', 'Frequency', 'Monetary'],
                     plot_kws={'alpha': 0.5, 's': 20})
    g.figure.suptitle(f'Customer Segments (k={rfm_df["Cluster"].nunique()})', y=1.02, fontsize=14, fontweight='bold')
    g.savefig(os.path.join(output_dir, 'cluster_plot.png'), dpi=150)
    plt.close('all')

def plot_cluster_boxplots(rfm_df, output_dir):
    """Graph 3: Cluster Characteristics Boxplots"""
    print("📊 [3/5] Generating Cluster Boxplots...")
    if len(rfm_df) > 5000:
        plot_df = rfm_df.sample(n=5000, random_state=42)
    else:
        plot_df = rfm_df
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.boxplot(x='Cluster', y='Recency', data=plot_df, ax=axes[0], palette='viridis')
    axes[0].set_title('Recency by Cluster', fontsize=12, fontweight='bold')
    sns.boxplot(x='Cluster', y='Frequency', data=plot_df, ax=axes[1], palette='viridis')
    axes[1].set_title('Frequency by Cluster', fontsize=12, fontweight='bold')
    sns.boxplot(x='Cluster', y='Monetary', data=plot_df, ax=axes[2], palette='viridis')
    axes[2].set_title('Monetary by Cluster', fontsize=12, fontweight='bold')
    plt.suptitle('Cluster Characteristics (RFM)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_characteristics.png'), dpi=150)
    plt.close()

def plot_cluster_distribution(rfm_df, output_dir):
    """Graph 4: Cluster Size Distribution (Pie Chart)"""
    print("📊 [4/5] Generating Cluster Distribution...")
    cluster_counts = rfm_df['Cluster'].value_counts().sort_index()
    
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart
    ax1.pie(cluster_counts, labels=[f'Cluster {i}\n({c:,} customers)' for i, c in cluster_counts.items()],
            autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize': 11})
    ax1.set_title('Customer Distribution by Cluster', fontsize=14, fontweight='bold')
    
    # Bar chart
    ax2.bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of Customers')
    ax2.set_title('Cluster Sizes', fontsize=14, fontweight='bold')
    for i, v in enumerate(cluster_counts.values):
        ax2.text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'), dpi=150)
    plt.close()

def plot_rfm_heatmap(rfm_df, output_dir):
    """Graph 5: RFM Means Heatmap per Cluster"""
    print("📊 [5/5] Generating RFM Heatmap per Cluster...")
    cluster_means = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Normalize for heatmap (0-1 scale)
    cluster_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Raw values heatmap
    sns.heatmap(cluster_means, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1, linewidths=1)
    ax1.set_title('Cluster Mean Values (Raw)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cluster')
    
    # Normalized heatmap
    sns.heatmap(cluster_normalized, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax2, linewidths=1)
    ax2.set_title('Cluster Mean Values (Normalized 0-1)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cluster')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rfm_cluster_heatmap.png'), dpi=150)
    plt.close()
    
    return cluster_means

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4: K-MEANS CLUSTERING (RFM)")
    print("=" * 60)
    
    INPUT_PATH = 'data/online_retail.csv'
    OUTPUT_DIR = 'data'
    
    try:
        df = load_and_clean_data(INPUT_PATH)
        
        if df is not None:
            # RFM Analysis
            rfm = rfm_analysis(df)
            print(f"RFM Data Shape: {rfm.shape}")
            
            # Print RFM stats
            print(f"\n📊 RFM Summary:")
            print(f"   Recency  - Mean: {rfm['Recency'].mean():.1f}, Median: {rfm['Recency'].median():.1f}")
            print(f"   Frequency - Mean: {rfm['Frequency'].mean():.1f}, Median: {rfm['Frequency'].median():.1f}")
            print(f"   Monetary  - Mean: {rfm['Monetary'].mean():.1f}, Median: {rfm['Monetary'].median():.1f}")
            
            # Scale features
            features = ['Recency', 'Frequency', 'Monetary']
            X = rfm[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # MLflow Tracking
            mlflow.set_experiment("customer")
            
            with mlflow.start_run(run_name="Step4_KMeans_Clustering"):
                # Tags
                mlflow.set_tag("stage", "clustering")
                mlflow.set_tag("model_type", "kmeans")
                mlflow.set_tag("description", "K-Means clustering on RFM features with elbow analysis")
                
                # === ELBOW CURVE (tests k=2 to k=10) ===
                output_dir = create_output_dir(OUTPUT_DIR)
                inertias, silhouettes = plot_elbow_curve(X_scaled, output_dir, max_k=10)
                mlflow.log_artifact(os.path.join(output_dir, 'elbow_curve.png'))
                
                # Log elbow data for each k
                for i, k_val in enumerate(range(2, 11)):
                    mlflow.log_metric(f"inertia_k{k_val}", inertias[i])
                    mlflow.log_metric(f"silhouette_k{k_val}", silhouettes[i])
                
                # === TRAIN FINAL MODEL (k=4) ===
                k = 4
                print(f"\n🤖 Training final K-Means with k={k}...")
                
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                
                cluster_labels = kmeans.labels_
                rfm['Cluster'] = cluster_labels
                
                # === CALCULATE ALL METRICS ===
                inertia = kmeans.inertia_
                
                if len(X_scaled) > 10000:
                    from sklearn.utils import resample
                    X_sample, labels_sample = resample(X_scaled, cluster_labels, n_samples=10000, random_state=42)
                    silhouette = silhouette_score(X_sample, labels_sample)
                    db_score = davies_bouldin_score(X_sample, labels_sample)
                    ch_score = calinski_harabasz_score(X_sample, labels_sample)
                else:
                    silhouette = silhouette_score(X_scaled, cluster_labels)
                    db_score = davies_bouldin_score(X_scaled, cluster_labels)
                    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
                
                print(f"\n📏 Clustering Metrics:")
                print(f"   Inertia: {inertia:.2f}")
                print(f"   Silhouette Score: {silhouette:.4f} (closer to 1 = better)")
                print(f"   Davies-Bouldin Index: {db_score:.4f} (lower = better)")
                print(f"   Calinski-Harabasz Score: {ch_score:.2f} (higher = better)")
                
                # === LOG PARAMETERS ===
                mlflow.log_param("n_clusters", k)
                mlflow.log_param("random_state", 42)
                mlflow.log_param("n_init", 10)
                mlflow.log_param("scaler_type", "StandardScaler")
                mlflow.log_param("total_customers", len(rfm))
                mlflow.log_param("features_used", "Recency, Frequency, Monetary")
                mlflow.log_param("elbow_range_tested", "k=2 to k=10")
                
                # === LOG METRICS ===
                mlflow.log_metric("inertia", inertia)
                mlflow.log_metric("silhouette_score", silhouette)
                mlflow.log_metric("davies_bouldin_score", db_score)
                mlflow.log_metric("calinski_harabasz_score", ch_score)
                
                # RFM Stats
                mlflow.log_metric("rfm_recency_mean", float(rfm['Recency'].mean()))
                mlflow.log_metric("rfm_frequency_mean", float(rfm['Frequency'].mean()))
                mlflow.log_metric("rfm_monetary_mean", float(rfm['Monetary'].mean()))
                
                # Cluster sizes
                for cluster_id in sorted(rfm['Cluster'].unique()):
                    count = int((rfm['Cluster'] == cluster_id).sum())
                    pct = count / len(rfm) * 100
                    mlflow.log_metric(f"cluster_{cluster_id}_count", count)
                    mlflow.log_metric(f"cluster_{cluster_id}_pct", round(pct, 2))
                
                # === GENERATE ALL GRAPHS ===
                plot_cluster_pairplot(rfm, output_dir)
                plot_cluster_boxplots(rfm, output_dir)
                plot_cluster_distribution(rfm, output_dir)
                cluster_means = plot_rfm_heatmap(rfm, output_dir)
                
                # === LOG ALL ARTIFACTS ===
                graph_files = [
                    'cluster_plot.png',
                    'cluster_characteristics.png',
                    'cluster_distribution.png',
                    'rfm_cluster_heatmap.png'
                ]
                for gf in graph_files:
                    path = os.path.join(output_dir, gf)
                    if os.path.exists(path):
                        mlflow.log_artifact(path)
                        print(f"   ✅ Logged: {gf}")
                
                # Log the model
                mlflow.sklearn.log_model(kmeans, "kmeans_model")
                
                # Save & log cluster summary
                summary = rfm.groupby('Cluster').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean',
                    'Monetary': 'mean',
                    'Customer ID': 'count'
                }).rename(columns={'Customer ID': 'Count'}).reset_index()
                summary['Percentage'] = (summary['Count'] / summary['Count'].sum() * 100).round(2)
                
                summary_path = os.path.join(output_dir, 'cluster_summary.csv')
                summary.to_csv(summary_path, index=False)
                mlflow.log_artifact(summary_path)
                
                print(f"\n📋 Cluster Summary:")
                print(summary.to_string(index=False))
                
                # Save segments
                output_csv = os.path.join(OUTPUT_DIR, 'customer_segments.csv')
                rfm.to_csv(output_csv, index=False)
                print(f"\n💾 Segments saved to {output_csv}")
                
                # Save RFM stats
                rfm_stats_path = os.path.join(output_dir, 'rfm_statistics.csv')
                rfm[features].describe().to_csv(rfm_stats_path)
                mlflow.log_artifact(rfm_stats_path)
                
                print(f"\n✅ Step 4 complete! All results logged to MLflow.")
    
    except Exception as e:
        print(f"❌ An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
