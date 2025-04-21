#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import joblib
from scipy.ndimage import median_filter

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score, 
    roc_curve, roc_auc_score, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Paths
CATCH22_DATA_DIR = "/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/catch22_feats_sleepedf"
RESULTS_DIR = "/Users/tereza/spring_2025/STAT_4830/sleep_staging/results"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def get_true_subject_id(filename):
    """Extract true subject ID ignoring the night number"""
    basename = os.path.basename(filename).split('_')[0]
    if basename.startswith('SC4'):
        return basename[:5]  # SC4xx - first 5 chars for Sleep Cassette
    elif basename.startswith('ST7'):
        return basename[:5]  # ST7xx - first 5 chars for Sleep Telemetry
    else:
        return basename[:6]  # Fallback to original logic

def load_data(data_dir):
    """Load and concatenate all CSV feature files"""
    print("Loading data...")
    
    # Find all CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    # Get mapping of true subject IDs to recordings
    subject_map = {}
    all_data = []
    subject_ids = []
    recording_ids = []
    
    # Load each file and check for potential issues
    for file in csv_files:
        basename = os.path.basename(file)
        
        # Skip mapping files
        if "mapping" in basename:
            print(f"Skipping {file} - likely a mapping file")
            continue
            
        try:
            data = pd.read_csv(file)
            
            # Check if this is a valid data file with features and labels
            if 'label' not in data.columns:
                print(f"Skipping {file} - no label column")
                continue
                
            # Get metadata
            recording_id = basename.split('_')[0]
            true_subject = get_true_subject_id(file)
            
            # Add to subject map
            if true_subject not in subject_map:
                subject_map[true_subject] = []
            subject_map[true_subject].append(file)
            
            # Check for NaN values
            nan_count = data.isna().sum().sum()
            if nan_count > 0:
                print(f"Warning: {file} has {nan_count} NaN values")
            
            # Add to data list
            all_data.append(data)
            
            # Add metadata
            n_samples = len(data)
            subject_ids.extend([true_subject] * n_samples)
            recording_ids.extend([recording_id] * n_samples)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data files found! Check your data directory.")
    
    # Concatenate all data
    full_data = pd.concat(all_data, ignore_index=True)
    
    # Add metadata
    full_data['subject_id'] = subject_ids
    full_data['recording_id'] = recording_ids
    
    # Print data stats
    print(f"Loaded {len(full_data)} samples from {len(subject_map)} subjects")
    print(f"Data shape: {full_data.shape}")
    print(f"Missing values: {full_data.isna().sum().sum()}")
    
    # Print class distribution
    if 'label' in full_data.columns:
        label_counts = full_data['label'].value_counts().sort_index()
        print("\nOverall class distribution:")
        for label, count in label_counts.items():
            print(f"  Class {label}: {count} samples ({count/len(full_data)*100:.2f}%)")
    
    return full_data, subject_map

def split_by_subject(data, subject_map, train_ratio=0.8, val_ratio=0.25):
    """Split data by subject (not by recording or sample)"""
    # Get unique subjects
    subjects = list(subject_map.keys())
    np.random.shuffle(subjects)
    
    # Split subjects into train and test
    n_train_subjects = int(len(subjects) * train_ratio)
    train_subjects = subjects[:n_train_subjects]
    test_subjects = subjects[n_train_subjects:]
    
    # Further split train into train and validation
    n_val_subjects = int(len(train_subjects) * val_ratio)
    val_subjects = train_subjects[:n_val_subjects]
    train_subjects = train_subjects[n_val_subjects:]
    
    # Create masks
    train_mask = data['subject_id'].isin(train_subjects)
    val_mask = data['subject_id'].isin(val_subjects)
    test_mask = data['subject_id'].isin(test_subjects)
    
    # Split data
    X_train = data[train_mask].drop(['label', 'subject_id', 'recording_id'], axis=1)
    y_train = data[train_mask]['label']
    
    X_val = data[val_mask].drop(['label', 'subject_id', 'recording_id'], axis=1)
    y_val = data[val_mask]['label']
    
    X_test = data[test_mask].drop(['label', 'subject_id', 'recording_id'], axis=1)
    y_test = data[test_mask]['label']
    
    print(f"Train: {len(X_train)} samples from {len(train_subjects)} subjects")
    print(f"Validation: {len(X_val)} samples from {len(val_subjects)} subjects")
    print(f"Test: {len(X_test)} samples from {len(test_subjects)} subjects")
    
    # Print class distribution for each split
    print("\nClass distribution:")
    for name, y in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        print(f"  {name}: {dict(zip(unique, counts))}")
    
    # Save split info
    split_info = {
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test)
    }
    
    return X_train, X_val, X_test, y_train, y_val, y_test, split_info

def plot_class_distribution(y_train, y_val, y_test):
    """Plot the class distribution for each data split"""
    # Count classes in each split
    train_counts = pd.Series(y_train).value_counts().sort_index()
    val_counts = pd.Series(y_val).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    
    # Create a DataFrame for easy plotting
    class_dist = pd.DataFrame({
        'Train': train_counts,
        'Validation': val_counts,
        'Test': test_counts
    })
    
    # Ensure all classes are represented in each split
    all_classes = sorted(list(set(y_train) | set(y_val) | set(y_test)))
    class_dist = class_dist.reindex(all_classes, fill_value=0)
    
    # Map class indices to sleep stage names
    stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
    class_dist.index = [stage_names[i] for i in class_dist.index]
    
    # Plot
    plt.figure(figsize=(10, 6))
    class_dist.plot(kind='bar')
    plt.title('Class Distribution Across Dataset Splits')
    plt.xlabel('Sleep Stage')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plots", "class_distribution.png"))
    plt.close()
    
    # Also plot normalized distribution
    plt.figure(figsize=(10, 6))
    class_dist_norm = class_dist.div(class_dist.sum(axis=0), axis=1) * 100
    class_dist_norm.plot(kind='bar')
    plt.title('Normalized Class Distribution (%)')
    plt.xlabel('Sleep Stage')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plots", "class_distribution_normalized.png"))
    plt.close()
    
    # Save class distribution to text file
    with open(os.path.join(RESULTS_DIR, "metrics", "class_distribution.txt"), 'w') as f:
        f.write("Class Distribution (Counts):\n")
        f.write(class_dist.to_string())
        f.write("\n\nClass Distribution (Percentage):\n")
        f.write(class_dist_norm.to_string())
    
    return class_dist

def evaluate_model(model, X_test, y_test, model_name, class_names):
    """Evaluate a trained model and save detailed metrics"""
    # Get predictions
    y_pred = model.predict(X_test)
    
    # For probabilistic models, get prediction probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        # For non-probabilistic models (like SVM without probability=True)
        y_prob = None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate macro-average F1, precision, and recall
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    macro_precision = precision_score(y_test, y_pred, average='macro')
    macro_recall = recall_score(y_test, y_pred, average='macro')
    
    # Calculate weighted-average F1, precision, and recall
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    weighted_precision = precision_score(y_test, y_pred, average='weighted')
    weighted_recall = recall_score(y_test, y_pred, average='weighted')
    
    # Apply median smoothing
    y_pred_smoothed = median_filter(y_pred, size=5)
    report_smoothed = classification_report(y_test, y_pred_smoothed, target_names=class_names, output_dict=True)
    report_smoothed_text = classification_report(y_test, y_pred_smoothed, target_names=class_names)
    cm_smoothed = confusion_matrix(y_test, y_pred_smoothed)
    
    # Calculate metrics with smoothing
    accuracy_smoothed = accuracy_score(y_test, y_pred_smoothed)
    macro_f1_smoothed = f1_score(y_test, y_pred_smoothed, average='macro')
    
    # Save results
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'weighted_f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'class_report': report,
        'confusion_matrix': cm,
        'accuracy_smoothed': accuracy_smoothed,
        'macro_f1_smoothed': macro_f1_smoothed,
        'class_report_smoothed': report_smoothed,
        'confusion_matrix_smoothed': cm_smoothed,
    }
    
    # For models with probability estimates, calculate ROC AUC and PR AUC
    if y_prob is not None:
        roc_auc = {}
        pr_auc = {}
        
        # Calculate ROC AUC and PR AUC for each class
        for i in range(len(class_names)):
            # ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
            roc_auc[i] = auc(fpr, tpr)
            
            # Precision-Recall curve and AUC
            precision, recall, _ = precision_recall_curve(y_test == i, y_prob[:, i])
            pr_auc[i] = auc(recall, precision)
        
        # Calculate macro-average ROC AUC and PR AUC
        results['roc_auc'] = roc_auc
        results['pr_auc'] = pr_auc
        results['roc_auc_macro'] = sum(roc_auc.values()) / len(roc_auc)
        results['pr_auc_macro'] = sum(pr_auc.values()) / len(pr_auc)
    
    # Save results to file
    joblib.dump(results, os.path.join(RESULTS_DIR, "metrics", f"{model_name}_metrics.pkl"))
    
    # Also save text version of results for easier reading
    with open(os.path.join(RESULTS_DIR, "metrics", f"{model_name}_results.txt"), 'w') as f:
        f.write(f"Results for {model_name}:\n")
        f.write(f"============================\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro-average F1: {macro_f1:.4f}\n")
        f.write(f"Macro-average Precision: {macro_precision:.4f}\n")
        f.write(f"Macro-average Recall: {macro_recall:.4f}\n")
        f.write(f"Weighted-average F1: {weighted_f1:.4f}\n")
        f.write(f"Weighted-average Precision: {weighted_precision:.4f}\n")
        f.write(f"Weighted-average Recall: {weighted_recall:.4f}\n\n")
        
        if y_prob is not None:
            f.write(f"ROC AUC (macro-average): {results['roc_auc_macro']:.4f}\n")
            f.write(f"PR AUC (macro-average): {results['pr_auc_macro']:.4f}\n\n")
            
            f.write("Per-class ROC AUC:\n")
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name}: {roc_auc[i]:.4f}\n")
            f.write("\nPer-class PR AUC:\n")
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name}: {pr_auc[i]:.4f}\n")
        
        f.write("\nDetailed Classification Report:\n")
        f.write(report_text)
        
        f.write("\n\nWith Median Smoothing (window size = 5):\n")
        f.write(f"Accuracy: {accuracy_smoothed:.4f}\n")
        f.write(f"Macro-average F1: {macro_f1_smoothed:.4f}\n\n")
        f.write("Detailed Classification Report (smoothed):\n")
        f.write(report_smoothed_text)
    
    # Print summary
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-average F1: {macro_f1:.4f}")
    if y_prob is not None:
        print(f"Macro-average ROC AUC: {results['roc_auc_macro']:.4f}")
    print(f"With median smoothing - Accuracy: {accuracy_smoothed:.4f}, F1: {macro_f1_smoothed:.4f}")
    print(f"Detailed results saved to: {os.path.join(RESULTS_DIR, 'metrics', f'{model_name}_results.txt')}")
    
    return results

def plot_confusion_matrix(cm, class_names, model_name, title="Confusion Matrix", normalized=False, smoothed=False):
    """Plot and save a confusion matrix"""
    smooth_str = "_smoothed" if smoothed else ""
    norm_str = "_normalized" if normalized else ""
    
    plt.figure(figsize=(10, 8))
    
    if normalized:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_to_plot = cm_norm
        fmt = '.2f'
    else:
        cm_to_plot = cm
        fmt = 'd'
    
    sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f"{model_name}_confusion_matrix{smooth_str}{norm_str}.png"
    plt.savefig(os.path.join(RESULTS_DIR, "plots", filename))
    plt.close()
    
    # Also save confusion matrix as text
    filename_txt = f"{model_name}_confusion_matrix{smooth_str}{norm_str}.txt"
    with open(os.path.join(RESULTS_DIR, "metrics", filename_txt), 'w') as f:
        f.write(f"{title}:\n\n")
        
        # Write header
        f.write("True\\Pred")
        for name in class_names:
            f.write(f"\t{name}")
        f.write("\n")
        
        # Write matrix
        matrix_to_write = cm_norm if normalized else cm
        for i, name in enumerate(class_names):
            f.write(f"{name}")
            for j in range(len(class_names)):
                if normalized:
                    f.write(f"\t{matrix_to_write[i, j]:.2f}")
                else:
                    f.write(f"\t{matrix_to_write[i, j]}")
            f.write("\n")

def plot_roc_curves(y_test, y_prob, class_names, model_name):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    
    # Store AUC values
    roc_auc_values = {}
    
    # Plot ROC curve for each class
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_values[class_names[i]] = roc_auc
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    # Plot random chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULTS_DIR, "plots", f"{model_name}_roc_curves.png"))
    plt.close()
    
    # Save AUC values to text file
    with open(os.path.join(RESULTS_DIR, "metrics", f"{model_name}_roc_auc.txt"), 'w') as f:
        f.write(f"ROC AUC values for {model_name}:\n\n")
        for class_name, auc_value in roc_auc_values.items():
            f.write(f"{class_name}: {auc_value:.4f}\n")
        f.write(f"\nMacro-average AUC: {sum(roc_auc_values.values()) / len(roc_auc_values):.4f}")

def plot_precision_recall_curves(y_test, y_prob, class_names, model_name):
    """Plot precision-recall curves for each class"""
    plt.figure(figsize=(10, 8))
    
    # Store AUC values
    pr_auc_values = {}
    
    # Plot precision-recall curve for each class
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_test == i, y_prob[:, i])
        pr_auc = auc(recall, precision)
        pr_auc_values[class_names[i]] = pr_auc
        plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AUC = {pr_auc:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULTS_DIR, "plots", f"{model_name}_pr_curves.png"))
    plt.close()
    
    # Save AUC values to text file
    with open(os.path.join(RESULTS_DIR, "metrics", f"{model_name}_pr_auc.txt"), 'w') as f:
        f.write(f"Precision-Recall AUC values for {model_name}:\n\n")
        for class_name, auc_value in pr_auc_values.items():
            f.write(f"{class_name}: {auc_value:.4f}\n")
        f.write(f"\nMacro-average AUC: {sum(pr_auc_values.values()) / len(pr_auc_values):.4f}")

def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Select top N features
        indices = indices[:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_importances)), top_importances, align='center')
        plt.yticks(range(len(top_importances)), top_features)
        plt.xlabel('Importance')
        plt.title(f'Feature Importance ({model_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "plots", f"{model_name}_feature_importance.png"))
        plt.close()
        
        # Also save numerical values
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        feature_importance_df.to_csv(os.path.join(RESULTS_DIR, "metrics", f"{model_name}_feature_importance.csv"), index=False)
        
        # Save top features to text file
        with open(os.path.join(RESULTS_DIR, "metrics", f"{model_name}_feature_importance.txt"), 'w') as f:
            f.write(f"Top {top_n} Important Features for {model_name}:\n\n")
            for i, (feature, importance) in enumerate(zip(top_features, top_importances), 1):
                f.write(f"{i}. {feature}: {importance:.6f}\n")
        
        return feature_importance_df
    return None

def save_summary_table(all_results):
    """Save a summary table of all model results"""
    summary = []
    
    for result in all_results:
        model_name = result['model_name']
        
        # Basic metrics
        row = {
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Macro F1': result['macro_f1'],
            'Weighted F1': result['weighted_f1'],
            'Accuracy (Smoothed)': result['accuracy_smoothed'],
            'Macro F1 (Smoothed)': result['macro_f1_smoothed']
        }
        
        # Add per-class F1 scores
        for cls_name, metrics in result['class_report'].items():
            if cls_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            row[f'F1 ({cls_name})'] = metrics['f1-score']
        
        # Add ROC AUC and PR AUC if available
        if 'roc_auc_macro' in result:
            row['ROC AUC (Macro)'] = result['roc_auc_macro']
            row['PR AUC (Macro)'] = result['pr_auc_macro']
        
        summary.append(row)
    
    # Convert to DataFrame and save
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "metrics", "model_comparison_summary.csv"), index=False)
    
    # Save as text table
    with open(os.path.join(RESULTS_DIR, "metrics", "model_comparison_summary.txt"), 'w') as f:
        f.write("Model Comparison Summary\n")
        f.write("=======================\n\n")
        
        # Format the table
        col_widths = {col: max(len(col), summary_df[col].astype(str).map(len).max()) 
                     for col in summary_df.columns}
        
        # Write header
        f.write("| ")
        for col in summary_df.columns:
            f.write(f"{col:{col_widths[col]}} | ")
        f.write("\n")
        
        # Write separator
        f.write("| ")
        for col in summary_df.columns:
            f.write(f"{'-' * col_widths[col]} | ")
        f.write("\n")
        
        # Write data
        for _, row in summary_df.iterrows():
            f.write("| ")
            for col in summary_df.columns:
                val = row[col]
                if isinstance(val, float):
                    f.write(f"{val:{col_widths[col]}.4f} | ")
                else:
                    f.write(f"{val:{col_widths[col]}} | ")
            f.write("\n")
    
    # Also save a pretty markdown table for the readme
    with open(os.path.join(RESULTS_DIR, "metrics", "model_comparison_summary.md"), 'w') as f:
        f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
    
    return summary_df

def compare_model_performance(all_results, metric='macro_f1'):
    """Create a comparison plot of models based on a specific metric"""
    models = [r['model_name'] for r in all_results]
    values = [r[metric] for r in all_results]
    
    # Sort by performance
    sorted_indices = np.argsort(values)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    # Convert metric name to display name
    metric_display = metric.replace('_', ' ').title()
    if 'Auc' in metric_display:
        metric_display = metric_display.replace('Auc', 'AUC')
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(sorted_models)), sorted_values, align='center')
    plt.yticks(range(len(sorted_models)), sorted_models)
    plt.xlim(0, max(sorted_values) * 1.1)
    
    # Add values to bars
    for i, v in enumerate(sorted_values):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    plt.xlabel(metric_display)
    plt.title(f'Model Comparison by {metric_display}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "plots", f"model_comparison_{metric}.png"))
    plt.close()
    
    # Also save as text file
    with open(os.path.join(RESULTS_DIR, "metrics", f"model_comparison_{metric}.txt"), 'w') as f:
        f.write(f"Model Comparison by {metric_display}\n")
        f.write("=================================\n\n")
        f.write("Ranked from highest to lowest:\n\n")
        for model, value in zip(sorted_models, sorted_values):
            f.write(f"{model}: {value:.4f}\n")

def main():
    # 1. Load data
    data, subject_map = load_data(CATCH22_DATA_DIR)
    
    # 2. Split data by subject
    X_train, X_val, X_test, y_train, y_val, y_test, split_info = split_by_subject(data, subject_map)
    
    # Save split info
    joblib.dump(split_info, os.path.join(RESULTS_DIR, "metrics", "data_split_info.pkl"))
    
    # 3. Plot class distribution
    class_dist = plot_class_distribution(y_train, y_val, y_test)
    
    # 4. Define class names
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    # 5. Data preprocessing
    # Keep track of feature names for interpretation
    feature_names = list(X_train.columns)
    
    # Check for NaN values
    nan_counts = {
        'X_train': X_train.isna().sum().sum(),
        'X_val': X_val.isna().sum().sum(),
        'X_test': X_test.isna().sum().sum()
    }
    
    # Log NaN values
    with open(os.path.join(RESULTS_DIR, "metrics", "preprocessing_stats.txt"), 'w') as f:
        f.write("Data Preprocessing Statistics\n")
        f.write("============================\n\n")
        f.write(f"NaN values in training set: {nan_counts['X_train']}\n")
        f.write(f"NaN values in validation set: {nan_counts['X_val']}\n")
        f.write(f"NaN values in test set: {nan_counts['X_test']}\n\n")
        
        if sum(nan_counts.values()) > 0:
            f.write("NaN handling strategy: Using SimpleImputer with median strategy\n")
    
    # Convert to numpy arrays
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values
    
    # 6. Define models to evaluate
    models = {
        "random_forest": {
            "name": "RandomForest",
            "pipeline": Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100, max_depth=15, min_samples_split=5,
                    random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1))
            ])
        },
        "hist_gradient_boosting": {
            "name": "HistGradientBoosting",
            "pipeline": Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', HistGradientBoostingClassifier(
                    max_iter=100, max_depth=5, learning_rate=0.1,
                    random_state=RANDOM_STATE, class_weight='balanced'))
            ])
        },
        "svm": {
            "name": "SVM",
            "pipeline": Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    C=1.0, kernel='rbf', gamma='scale',
                    class_weight='balanced', probability=True,
                    random_state=RANDOM_STATE))
            ])
        },
        "logistic_regression": {
            "name": "LogisticRegression",
            "pipeline": Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    multi_class='multinomial', solver='lbfgs', max_iter=1000,
                    class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))
            ])
        },
        "mlp": {
            "name": "MLP",
            "pipeline": Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                    alpha=0.0001, learning_rate='adaptive', max_iter=500,
                    random_state=RANDOM_STATE))
            ])
        },
        "smote_random_forest": {
            "name": "SMOTE_RandomForest",
            "pipeline": ImbPipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('classifier', RandomForestClassifier(
                    n_estimators=100, max_depth=15, min_samples_split=5,
                    random_state=RANDOM_STATE, n_jobs=-1))
            ])
        }
    }

    # Create a summary file for the analysis
    with open(os.path.join(RESULTS_DIR, "analysis_summary.txt"), 'w') as f:
        f.write("Sleep Stage Classification Analysis\n")
        f.write("================================\n\n")
        f.write(f"Data source: {CATCH22_DATA_DIR}\n")
        f.write(f"Results directory: {RESULTS_DIR}\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"- Total samples: {len(data)}\n")
        f.write(f"- Total subjects: {len(subject_map)}\n")
        f.write(f"- Train samples: {len(X_train)} from {len(split_info['train_subjects'])} subjects\n")
        f.write(f"- Validation samples: {len(X_val)} from {len(split_info['val_subjects'])} subjects\n")
        f.write(f"- Test samples: {len(X_test)} from {len(split_info['test_subjects'])} subjects\n\n")
        
        f.write("Models evaluated:\n")
        for model_key, model_info in models.items():
            f.write(f"- {model_info['name']}\n")
        
        f.write("\nAnalysis includes:\n")
        f.write("- Subject-based train/test splits\n")
        f.write("- Performance metrics (accuracy, F1 score, precision, recall)\n")
        f.write("- Confusion matrices\n")
        f.write("- ROC and Precision-Recall curves\n")
        f.write("- Feature importance analysis for tree-based models\n")
        f.write("- Temporal smoothing post-processing\n")
        f.write("- Class imbalance handling (balanced weights and SMOTE)\n\n")
        
        f.write("Check the metrics directory for detailed results on each model.\n")

    # 7. Train and evaluate each model
    all_results = []
    
    for model_key, model_info in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_info['name']}...")
        
        # Train model
        model = model_info['pipeline']
        
        try:
            model.fit(X_train, y_train)
            
            # Save model
            joblib.dump(model, os.path.join(RESULTS_DIR, "models", f"{model_info['name']}_model.pkl"))
            
            # Evaluate on validation set first
            print("\nValidation performance:")
            val_preds = model.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds, average='macro')
            print(f"Accuracy: {val_acc:.4f}, Macro F1: {val_f1:.4f}")
            
            # Save validation results
            with open(os.path.join(RESULTS_DIR, "metrics", f"{model_info['name']}_validation.txt"), 'w') as f:
                f.write(f"Validation Results for {model_info['name']}:\n")
                f.write(f"Accuracy: {val_acc:.4f}\n")
                f.write(f"Macro F1: {val_f1:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report(y_val, val_preds, target_names=class_names))
            
            # Evaluate on test set
            results = evaluate_model(model, X_test, y_test, model_info['name'], class_names)
            all_results.append(results)
            
            # Plot confusion matrices
            plot_confusion_matrix(results['confusion_matrix'], class_names, model_info['name'])
            plot_confusion_matrix(results['confusion_matrix'], class_names, model_info['name'], 
                                normalized=True, title="Normalized Confusion Matrix")
            
            plot_confusion_matrix(results['confusion_matrix_smoothed'], class_names, model_info['name'],
                                smoothed=True, title="Smoothed Confusion Matrix")
            plot_confusion_matrix(results['confusion_matrix_smoothed'], class_names, model_info['name'],
                                smoothed=True, normalized=True, title="Normalized Smoothed Confusion Matrix")
            
            # For models with probability estimates, plot ROC and PR curves
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
                plot_roc_curves(y_test, y_prob, class_names, model_info['name'])
                plot_precision_recall_curves(y_test, y_prob, class_names, model_info['name'])
            
            # For tree-based models, plot feature importance
            if model_key in ["random_forest", "hist_gradient_boosting", "smote_random_forest"]:
                # Get the classifier from the pipeline
                classifier = model.named_steps['classifier']
                plot_feature_importance(classifier, feature_names, model_info['name'])
                
        except Exception as e:
            print(f"Error training {model_info['name']}: {e}")
            
            # Log the error
            with open(os.path.join(RESULTS_DIR, "metrics", f"{model_info['name']}_error.txt"), 'w') as f:
                f.write(f"Error training {model_info['name']}:\n")
                f.write(str(e))
    
    # 8. Compare model performance
    if all_results:
        print("\n\nCreating summary tables and comparison plots...")
        save_summary_table(all_results)
        
        # Compare models on different metrics
        for metric in ['accuracy', 'macro_f1', 'weighted_f1', 'accuracy_smoothed', 'macro_f1_smoothed']:
            if all(metric in result for result in all_results):
                compare_model_performance(all_results, metric)
        
        # Also compare ROC AUC if available
        if all('roc_auc_macro' in result for result in all_results):
            compare_model_performance(all_results, 'roc_auc_macro')
    else:
        print("No models were successfully trained and evaluated.")
    
    print("\nAnalysis complete. All results saved to:", RESULTS_DIR)
    print("See the 'metrics' directory for detailed text reports of each model's performance.")

if __name__ == "__main__":
    main()