import os
import subprocess
import sys
import time

def run_script(script_name, step_number, step_description):
    """Run a single pipeline step and track timing."""
    print(f"\n{'='*60}")
    print(f"  STEP {step_number}: {step_description}")
    print(f"  Script: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name], 
            check=True, 
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n  ✅ Step {step_number} completed in {elapsed:.1f}s")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n  ❌ Step {step_number} FAILED after {elapsed:.1f}s: {e}")
        return False, elapsed

def main():
    print("=" * 60)
    print("  CUSTOMER SEGMENTATION - FULL PIPELINE")
    print("  All 6 steps will run sequentially")
    print("  Results logged to MLflow experiment: 'customer'")
    print("=" * 60)
    
    pipeline_start = time.time()
    
    # Define all 6 pipeline steps
    steps = [
        ("check_missing_values.py",  "DATA EXPLORATION & MISSING VALUES"),
        ("data_preprocessing.py",    "DATA PREPROCESSING"),
        ("eda.py",                   "EXPLORATORY DATA ANALYSIS (EDA)"),
        ("train_model.py",           "K-MEANS CLUSTERING (RFM)"),
        ("analyze_clusters.py",      "CLUSTER LABELING & BUSINESS ANALYSIS"),
        ("train_classifier.py",      "RANDOM FOREST CLASSIFICATION"),
    ]
    
    results = []
    
    for i, (script, description) in enumerate(steps, 1):
        success, elapsed = run_script(script, i, description)
        results.append((i, script, description, success, elapsed))
        
        if not success:
            print(f"\n{'='*60}")
            print(f"  ❌ PIPELINE FAILED at Step {i}: {description}")
            print(f"{'='*60}")
            break
    
    # Summary
    total_time = time.time() - pipeline_start
    
    print(f"\n\n{'='*60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Step':<6} {'Status':<8} {'Time':>8}   {'Description'}")
    print(f"{'-'*60}")
    
    for step_num, script, desc, success, elapsed in results:
        status = "✅" if success else "❌"
        print(f"  {step_num:<4} {status:<6}  {elapsed:>6.1f}s   {desc}")
    
    print(f"{'-'*60}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    all_passed = all(r[3] for r in results)
    
    if all_passed:
        print(f"\n  🎉 ALL 6 STEPS COMPLETED SUCCESSFULLY!")
        print(f"  📊 Open MLflow UI: http://localhost:5000")
        print(f"  🔍 Experiment: 'customer' (6 runs)")
        print(f"\n  Runs created:")
        print(f"    1. Step1_Data_Exploration")
        print(f"    2. Step2_Data_Preprocessing")
        print(f"    3. Step3_EDA (10 graphs)")
        print(f"    4. Step4_KMeans_Clustering (5 graphs)")
        print(f"    5. Step5_Cluster_Labeling (4 graphs)")
        print(f"    6. Step6_RandomForest_Classifier (4 graphs)")
    else:
        print(f"\n  ⚠️ Pipeline did not complete. Fix the failed step and re-run.")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
