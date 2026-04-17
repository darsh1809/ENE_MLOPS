import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

def analyze_and_label_clusters(segments_path='data/customer_segments.csv', output_path='data/labeled_customers.csv'):
    print("Analyzing Clusters...")
    
    # Load data
    try:
        df = pd.read_csv(segments_path)
    except FileNotFoundError:
        print(f"Error: {segments_path} not found.")
        return

    # Calculate mean metrics per cluster
    cluster_summary = df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Customer ID': 'count'
    }).rename(columns={'Customer ID': 'Count'}).reset_index()
    
    cluster_summary['Percentage'] = (cluster_summary['Count'] / cluster_summary['Count'].sum()) * 100
    
    print("\nCluster Summary (Means):")
    print(cluster_summary)
    
    # =============================================
    # LABELING LOGIC (RFM-based scoring)
    # =============================================
    cluster_summary['R_Score'] = cluster_summary['Recency'].rank(ascending=False)
    cluster_summary['F_Score'] = cluster_summary['Frequency'].rank(ascending=True)
    cluster_summary['M_Score'] = cluster_summary['Monetary'].rank(ascending=True)
    cluster_summary['Total_Score'] = cluster_summary['R_Score'] + cluster_summary['F_Score'] + cluster_summary['M_Score']
    
    sorted_clusters = cluster_summary.sort_values('Total_Score', ascending=False)
    
    labels = {}
    labels[sorted_clusters.iloc[0]['Cluster']] = "Champions"
    labels[sorted_clusters.iloc[1]['Cluster']] = "Potential Loyalists"
    labels[sorted_clusters.iloc[2]['Cluster']] = "At Risk"
    labels[sorted_clusters.iloc[3]['Cluster']] = "Lost"
    
    print("\n🏷️ Assigned Labels:")
    for cluster, label in labels.items():
        print(f"   Cluster {int(cluster)}: {label}")
        
    df['Segment'] = df['Cluster'].map(labels)
    
    # =============================================
    # BUSINESS METRICS
    # =============================================
    
    # Revenue Share
    total_revenue = df['Monetary'].sum()
    revenue_share = df.groupby('Segment')['Monetary'].sum() / total_revenue * 100
    
    # Churn Probability
    max_recency = df['Recency'].max()
    df['ChurnProbability'] = df['Recency'] / max_recency
    avg_churn_prob = df.groupby('Segment')['ChurnProbability'].mean()
    
    # Average Order Value per segment
    avg_monetary = df.groupby('Segment')['Monetary'].mean()
    avg_frequency = df.groupby('Segment')['Frequency'].mean()
    avg_recency = df.groupby('Segment')['Recency'].mean()
    
    # Segment counts
    segment_counts = df['Segment'].value_counts()
    
    print(f"\n💰 Revenue Share by Segment:")
    for seg, share in revenue_share.items():
        print(f"   {seg}: {share:.1f}%")
    
    print(f"\n⚠️ Average Churn Probability by Segment:")
    for seg, prob in avg_churn_prob.items():
        print(f"   {seg}: {prob:.2f}")
    
    # =============================================
    # GRAPHS
    # =============================================
    output_dir = 'data'
    
    # Graph 1: Revenue Share Pie Chart
    print("\n📊 [1/4] Generating Revenue Share Pie Chart...")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'Champions': '#4CAF50', 'Potential Loyalists': '#2196F3', 'At Risk': '#FF9800', 'Lost': '#E91E63'}
    segment_colors = [colors.get(seg, '#999') for seg in revenue_share.index]
    wedges, texts, autotexts = ax.pie(revenue_share, labels=revenue_share.index, 
                                       autopct='%1.1f%%', colors=segment_colors,
                                       startangle=90, textprops={'fontsize': 12})
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    ax.set_title('Revenue Share by Customer Segment', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'revenue_share_pie.png'), dpi=150)
    plt.close()
    
    # Graph 2: Segment Profile Radar-style Bar Chart
    print("📊 [2/4] Generating Segment Profile Chart...")
    seg_profile = df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    segment_order = ['Champions', 'Potential Loyalists', 'At Risk', 'Lost']
    seg_profile_ordered = seg_profile.reindex([s for s in segment_order if s in seg_profile.index])
    bar_colors = [colors.get(s, '#999') for s in seg_profile_ordered.index]
    
    seg_profile_ordered['Recency'].plot(kind='bar', ax=axes[0], color=bar_colors)
    axes[0].set_title('Avg Recency (days)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Days')
    axes[0].tick_params(axis='x', rotation=30)
    
    seg_profile_ordered['Frequency'].plot(kind='bar', ax=axes[1], color=bar_colors)
    axes[1].set_title('Avg Frequency (orders)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Orders')
    axes[1].tick_params(axis='x', rotation=30)
    
    seg_profile_ordered['Monetary'].plot(kind='bar', ax=axes[2], color=bar_colors)
    axes[2].set_title('Avg Monetary (£)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Revenue (£)')
    axes[2].tick_params(axis='x', rotation=30)
    
    plt.suptitle('Segment Profiles (RFM Averages)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segment_profile.png'), dpi=150)
    plt.close()
    
    # Graph 3: Churn Probability by Segment
    print("📊 [3/4] Generating Churn Probability Chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    churn_ordered = avg_churn_prob.reindex([s for s in segment_order if s in avg_churn_prob.index])
    churn_colors = [colors.get(s, '#999') for s in churn_ordered.index]
    bars = ax.bar(churn_ordered.index, churn_ordered.values, color=churn_colors)
    ax.set_title('Average Churn Probability by Segment', fontsize=14, fontweight='bold')
    ax.set_ylabel('Churn Probability (0-1)')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, churn_ordered.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', fontweight='bold', fontsize=12)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'churn_probability.png'), dpi=150)
    plt.close()
    
    # Graph 4: Segment Customer Count
    print("📊 [4/4] Generating Segment Count Chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    counts_ordered = segment_counts.reindex([s for s in segment_order if s in segment_counts.index])
    count_colors = [colors.get(s, '#999') for s in counts_ordered.index]
    bars = ax.bar(counts_ordered.index, counts_ordered.values, color=count_colors)
    ax.set_title('Number of Customers per Segment', fontsize=14, fontweight='bold')
    ax.set_ylabel('Customer Count')
    for bar, val in zip(bars, counts_ordered.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{val:,}', ha='center', fontweight='bold', fontsize=12)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segment_customer_count.png'), dpi=150)
    plt.close()
    
    # =============================================
    # MLFLOW LOGGING
    # =============================================
    mlflow.set_experiment("customer")
    
    with mlflow.start_run(run_name="Step5_Cluster_Labeling"):
        # Tags
        mlflow.set_tag("stage", "cluster_labeling")
        mlflow.set_tag("model_type", "none")
        mlflow.set_tag("description", "Business label assignment, revenue analysis, churn estimation")
        
        # Parameters
        mlflow.log_param("labeling_method", "RFM Score Ranking")
        mlflow.log_param("segments", "Champions, Potential Loyalists, At Risk, Lost")
        mlflow.log_param("total_customers", len(df))
        mlflow.log_param("churn_formula", "Recency / Max_Recency")
        
        # Log label mapping
        for cluster_id, label in labels.items():
            mlflow.log_param(f"cluster_{int(cluster_id)}_label", label)
        
        # Metrics - per segment
        for seg in segment_order:
            if seg in segment_counts.index:
                safe_seg = seg.replace(" ", "_")
                mlflow.log_metric(f"count_{safe_seg}", int(segment_counts[seg]))
                mlflow.log_metric(f"revenue_share_{safe_seg}", round(float(revenue_share[seg]), 2))
                mlflow.log_metric(f"churn_prob_{safe_seg}", round(float(avg_churn_prob[seg]), 4))
                mlflow.log_metric(f"avg_monetary_{safe_seg}", round(float(avg_monetary[seg]), 2))
                mlflow.log_metric(f"avg_frequency_{safe_seg}", round(float(avg_frequency[seg]), 2))
                mlflow.log_metric(f"avg_recency_{safe_seg}", round(float(avg_recency[seg]), 2))
        
        # Overall metrics
        mlflow.log_metric("total_revenue", float(total_revenue))
        mlflow.log_metric("champion_revenue_pct", round(float(revenue_share.get('Champions', 0)), 2))
        mlflow.log_metric("at_risk_count", int(segment_counts.get('At Risk', 0)))
        mlflow.log_metric("lost_count", int(segment_counts.get('Lost', 0)))
        
        # Log Graphs
        graph_files = ['revenue_share_pie.png', 'segment_profile.png', 
                       'churn_probability.png', 'segment_customer_count.png']
        for gf in graph_files:
            path = os.path.join(output_dir, gf)
            if os.path.exists(path):
                mlflow.log_artifact(path)
                print(f"   ✅ Logged: {gf}")
        
        # Save & log detailed report
        report_path = os.path.join(output_dir, 'segment_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CUSTOMER SEGMENT ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write("SEGMENT LABELS:\n")
            for cluster_id, label in labels.items():
                f.write(f"  Cluster {int(cluster_id)} → {label}\n")
            f.write(f"\nREVENUE SHARE:\n")
            for seg, share in revenue_share.items():
                f.write(f"  {seg}: {share:.1f}%\n")
            f.write(f"\nCHURN PROBABILITY:\n")
            for seg, prob in avg_churn_prob.items():
                f.write(f"  {seg}: {prob:.2f}\n")
            f.write(f"\nSEGMENT DETAILS:\n")
            for seg in segment_order:
                if seg in segment_counts.index:
                    f.write(f"\n  {seg}:\n")
                    f.write(f"    Customers: {segment_counts[seg]:,}\n")
                    f.write(f"    Avg Recency: {avg_recency[seg]:.0f} days\n")
                    f.write(f"    Avg Frequency: {avg_frequency[seg]:.1f} orders\n")
                    f.write(f"    Avg Monetary: £{avg_monetary[seg]:,.2f}\n")
                    f.write(f"    Revenue Share: {revenue_share[seg]:.1f}%\n")
                    f.write(f"    Churn Prob: {avg_churn_prob[seg]:.2f}\n")
        
        mlflow.log_artifact(report_path)
        
        # Save labeled data
        labeled_csv = os.path.join(output_dir, 'labeled_summary.csv')
        summary_df = pd.DataFrame({
            'Segment': segment_order,
            'Count': [int(segment_counts.get(s, 0)) for s in segment_order],
            'Revenue_Share_%': [round(float(revenue_share.get(s, 0)), 1) for s in segment_order],
            'Avg_Recency': [round(float(avg_recency.get(s, 0)), 0) for s in segment_order],
            'Avg_Frequency': [round(float(avg_frequency.get(s, 0)), 1) for s in segment_order],
            'Avg_Monetary': [round(float(avg_monetary.get(s, 0)), 2) for s in segment_order],
            'Churn_Probability': [round(float(avg_churn_prob.get(s, 0)), 2) for s in segment_order]
        })
        summary_df.to_csv(labeled_csv, index=False)
        mlflow.log_artifact(labeled_csv)
    
    # Save full labeled data
    df.to_csv(output_path, index=False)
    print(f"\n💾 Labeled data saved to {output_path}")
    print(f"✅ Step 5 complete! All results logged to MLflow.")

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5: CLUSTER LABELING & BUSINESS ANALYSIS")
    print("=" * 60)
    analyze_and_label_clusters()
