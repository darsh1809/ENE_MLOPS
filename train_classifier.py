import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, label_binarize
import mlflow
import mlflow.sklearn

from data_preprocessing import load_and_clean_data

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://54.206.46.48:5000"))

def prepare_classification_data(labels_path='data/labeled_customers.csv', transactions_path='data/online_retail.csv'):
    print("Preparing data for classification...")
    
    # Load Labels
    try:
        labels_df = pd.read_csv(labels_path)
    except FileNotFoundError:
        print(f"Labels file {labels_path} not found.")
        return None
        
    # Load Transactions
    transactions_df = load_and_clean_data(transactions_path)
    if transactions_df is None:
        return None
        
    transactions_df['InvoiceDate'] = pd.to_datetime(transactions_df['InvoiceDate'])
    
    if 'TotalPrice' not in transactions_df.columns:
        transactions_df['TotalPrice'] = transactions_df['Quantity'] * transactions_df['Price']
        
    # Average Basket Value & Size
    basket_metrics = transactions_df.groupby(['Customer ID', 'Invoice']).agg({
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    }).reset_index().groupby('Customer ID').agg({
        'TotalPrice': ['mean', 'std'],
        'Quantity': 'mean'
    })
    
    basket_metrics.columns = ['AvgBasketValue', 'Volatility', 'AvgBasketSize']
    basket_metrics = basket_metrics.reset_index()
    basket_metrics['Volatility'] = basket_metrics['Volatility'].fillna(0)
    
    # Favorite Shopping Day & Country
    transactions_df['DayOfWeek'] = transactions_df['InvoiceDate'].dt.day_name()
    
    def get_mode(x):
        return x.mode().iloc[0] if not x.mode().empty else np.nan

    categorical_features = transactions_df.groupby('Customer ID').agg({
        'DayOfWeek': get_mode,
        'Country': get_mode
    }).rename(columns={'DayOfWeek': 'FavoriteDay'}).reset_index()
    
    # Merge Features
    features_df = basket_metrics.merge(categorical_features, on='Customer ID', how='inner')
    
    # Merge with Labels (Target)
    full_df = features_df.merge(labels_df[['Customer ID', 'Segment']], on='Customer ID', how='inner')
    
    return full_df

def plot_confusion_matrix(y_test, y_pred, class_names, output_dir):
    """Graph 1: Confusion Matrix Heatmap"""
    print("   [1/4] Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax1, linewidths=1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Actual')
    ax1.set_xlabel('Predicted')
    
    # Percentage
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens', xticklabels=class_names,
                yticklabels=class_names, ax=ax2, linewidths=1)
    ax2.set_title('Confusion Matrix (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

def plot_roc_curves(y_test, y_prob, class_names, output_dir):
    """Graph 2: Multi-class ROC Curves"""
    print("   [2/4] Generating ROC Curves...")
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc_val:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150)
    plt.close()

def plot_feature_importance(importances_df, output_dir):
    """Graph 3: Feature Importance Bar Chart"""
    print("   [3/4] Generating Feature Importance Chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4CAF50' if v > 0.1 else '#2196F3' if v > 0.01 else '#BDBDBD' 
              for v in importances_df['Importance']]
    ax.barh(importances_df['Feature'], importances_df['Importance'], color=colors)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(importances_df['Importance']):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')
    
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_chart.png'), dpi=150)
    plt.close()

def plot_per_class_metrics(report_dict, class_names, output_dir):
    """Graph 4: Per-class Precision, Recall, F1"""
    print("   [4/4] Generating Per-Class Metrics Chart...")
    metrics_data = []
    for cls in class_names:
        if cls in report_dict:
            metrics_data.append({
                'Segment': cls,
                'Precision': report_dict[cls]['precision'],
                'Recall': report_dict[cls]['recall'],
                'F1-Score': report_dict[cls]['f1-score'],
                'Support': report_dict[cls]['support']
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_df))
    width = 0.25
    
    bars1 = ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='#4CAF50')
    bars2 = ax.bar(x, metrics_df['Recall'], width, label='Recall', color='#2196F3')
    bars3 = ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#FF9800')
    
    ax.set_xlabel('Segment')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Segment'], rotation=30)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                    f'{height:.2f}', ha='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150)
    plt.close()
    
    return metrics_df

def train_and_evaluate_classifier(df):
    print("\nTraining Random Forest Classifier...")
    output_dir = 'data'
    
    # Encode Categoricals
    le_day = LabelEncoder()
    df['FavoriteDay_Encoded'] = le_day.fit_transform(df['FavoriteDay'].astype(str))
    
    le_country = LabelEncoder()
    df['Country_Encoded'] = le_country.fit_transform(df['Country'].astype(str))
    
    # Features & Target
    feature_cols = ['AvgBasketValue', 'AvgBasketSize', 'Volatility', 'FavoriteDay_Encoded', 'Country_Encoded']
    X = df[feature_cols]
    y = df['Segment']
    
    # Encode Target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    class_names = list(le_target.classes_)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # MLflow
    mlflow.set_experiment("customer")
    
    with mlflow.start_run(run_name="Step6_RandomForest_Classifier"):
        # Tags
        mlflow.set_tag("stage", "classification")
        mlflow.set_tag("model_type", "random_forest")
        mlflow.set_tag("description", "Supervised classification to predict customer segments")
        
        # Parameters
        mlflow.log_param("algorithm", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("total_rows", len(df))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("features_used", ", ".join(feature_cols))
        mlflow.log_param("target_classes", ", ".join(class_names))
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("stratified_split", True)
        
        # Train
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)
        
        # =============================================
        # METRICS
        # =============================================
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        except ValueError:
            roc_auc = 0.0
            
        report_str = classification_report(y_test, y_pred, target_names=class_names)
        report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        
        print("\n" + "=" * 50)
        print("CLASSIFICATION RESULTS")
        print("=" * 50)
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        print(f"\n{report_str}")
        
        # Feature Importance
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Feature Importance:")
        print(importances.to_string(index=False))
        
        # =============================================
        # LOG METRICS
        # =============================================
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_score_weighted", f1)
        mlflow.log_metric("roc_auc_weighted", roc_auc)
        
        # Per-class metrics
        for cls in class_names:
            if cls in report_dict:
                safe_cls = cls.replace(" ", "_")
                mlflow.log_metric(f"precision_{safe_cls}", round(report_dict[cls]['precision'], 4))
                mlflow.log_metric(f"recall_{safe_cls}", round(report_dict[cls]['recall'], 4))
                mlflow.log_metric(f"f1_{safe_cls}", round(report_dict[cls]['f1-score'], 4))
                mlflow.log_metric(f"support_{safe_cls}", int(report_dict[cls]['support']))
        
        # Feature importance as metrics
        for _, row in importances.iterrows():
            mlflow.log_metric(f"importance_{row['Feature']}", round(row['Importance'], 4))
        
        # =============================================
        # GENERATE ALL GRAPHS
        # =============================================
        print("\nGenerating graphs...")
        plot_confusion_matrix(y_test, y_pred, class_names, output_dir)
        plot_roc_curves(y_test, y_prob, class_names, output_dir)
        plot_feature_importance(importances, output_dir)
        metrics_df = plot_per_class_metrics(report_dict, class_names, output_dir)
        
        # =============================================
        # LOG ALL ARTIFACTS
        # =============================================
        # Graphs
        graph_files = ['confusion_matrix.png', 'roc_curves.png', 
                       'feature_importance_chart.png', 'per_class_metrics.png']
        for gf in graph_files:
            path = os.path.join(output_dir, gf)
            if os.path.exists(path):
                mlflow.log_artifact(path)
                print(f"   Logged: {gf}")
        
        # Feature importance CSV
        importances.to_csv("data/feature_importance.csv", index=False)
        mlflow.log_artifact("data/feature_importance.csv")
        
        # Classification report text
        report_path = "data/classification_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Algorithm: Random Forest (n_estimators=100)\n")
            f.write(f"Train/Test Split: 80/20 (stratified)\n")
            f.write(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}\n\n")
            f.write(f"OVERALL METRICS:\n")
            f.write(f"  Accuracy:  {accuracy:.4f}\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall:    {recall:.4f}\n")
            f.write(f"  F1 Score:  {f1:.4f}\n")
            f.write(f"  ROC-AUC:   {roc_auc:.4f}\n\n")
            f.write(f"PER-CLASS REPORT:\n")
            f.write(report_str)
            f.write(f"\n\nFEATURE IMPORTANCE:\n")
            f.write(importances.to_string(index=False))
        mlflow.log_artifact(report_path)
        
        # Log the model
        mlflow.sklearn.log_model(rf, "random_forest_model")
        
        print(f"\n   Model logged to MLflow!")
        print(f"   All artifacts and metrics logged to MLflow!")
        
        return rf, le_target

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6: RANDOM FOREST CLASSIFICATION")
    print("=" * 60)
    
    df = prepare_classification_data()
    
    if df is not None:
        model, label_encoder = train_and_evaluate_classifier(df)
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print("Check MLflow UI -> 'customer' experiment -> 'Step6_RandomForest_Classifier'")
    else:
        print("Classification failed - data preparation error.")
