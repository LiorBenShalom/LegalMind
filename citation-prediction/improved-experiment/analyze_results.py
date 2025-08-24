#!/usr/bin/env python3
"""
Results Analyzer
================

Analyzes and compares results from improved model training.
Generates comprehensive statistical analysis and visualizations.

Usage:
    python analyze_results.py --results_dir improved_results
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results = []
        self.model_results = defaultdict(list)
        
    def load_all_results(self):
        """Load all results from the results directory"""
        logger.info("📂 Loading all results...")
        
        # Find all result files
        result_files = list(self.results_dir.glob("*/results.json"))
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    
                # Add file path for reference
                result['result_file'] = str(result_file)
                result['folder_name'] = result_file.parent.name
                
                self.results.append(result)
                
                # Group by model
                model_name = result['model']
                self.model_results[model_name].append(result)
                
            except Exception as e:
                logger.warning(f"❌ Failed to load {result_file}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(self.results)} results")
        logger.info(f"📊 Models found: {list(self.model_results.keys())}")
        
        # Log fold distribution
        for model_name, model_res in self.model_results.items():
            folds = [r['fold'] for r in model_res]
            logger.info(f"  {model_name}: {len(folds)} folds - {sorted(folds)}")
    
    def create_results_dataframe(self):
        """Create a comprehensive DataFrame with all results"""
        rows = []
        
        for result in self.results:
            # Extract main metrics
            row = {
                'model': result['model'],
                'fold': result['fold'],
                'test_f1': result['test_f1'],
                'val_f1': result['best_val_f1'],
                'threshold': result['best_threshold'],
                'test_accuracy': result['test_metrics']['accuracy'],
                'test_precision': result['test_metrics']['precision'],
                'test_recall': result['test_metrics']['recall']
            }
            
            # Extract confusion matrix if available
            if 'confusion_matrix' in result['test_metrics']:
                cm = result['test_metrics']['confusion_matrix']
                row.update({
                    'tp': cm['tp'], 'fp': cm['fp'],
                    'tn': cm['tn'], 'fn': cm['fn']
                })
            
            # Extract config if available
            if 'config' in result:
                config = result['config']
                row.update({
                    'learning_rate': config.get('learning_rate', np.nan),
                    'patience': config.get('patience', np.nan),
                    'class_weight_ratio': config.get('class_weight_ratio', np.nan),
                    'gradient_clip': config.get('gradient_clip', np.nan)
                })
            
            rows.append(row)
        
        self.df_results = pd.DataFrame(rows)
        logger.info(f"📊 Created results DataFrame: {self.df_results.shape}")
        
        return self.df_results
    
    def calculate_statistics(self):
        """Calculate comprehensive statistics for each model"""
        logger.info("📈 Calculating statistics...")
        
        stats_results = {}
        
        for model_name in self.model_results.keys():
            model_data = self.df_results[self.df_results['model'] == model_name]
            
            if len(model_data) == 0:
                continue
            
            # Calculate statistics for key metrics
            stats_results[model_name] = {
                'n_folds': len(model_data),
                'f1_mean': model_data['test_f1'].mean(),
                'f1_std': model_data['test_f1'].std(),
                'f1_min': model_data['test_f1'].min(),
                'f1_max': model_data['test_f1'].max(),
                'accuracy_mean': model_data['test_accuracy'].mean(),
                'accuracy_std': model_data['test_accuracy'].std(),
                'precision_mean': model_data['test_precision'].mean(),
                'precision_std': model_data['test_precision'].std(),
                'recall_mean': model_data['test_recall'].mean(),
                'recall_std': model_data['test_recall'].std(),
                'threshold_mean': model_data['threshold'].mean(),
                'threshold_std': model_data['threshold'].std()
            }
            
            # Add 95% confidence intervals
            n = len(model_data)
            if n > 1:
                alpha = 0.05
                dof = n - 1
                t_critical = stats.t.ppf(1 - alpha/2, dof)
                
                f1_se = stats.sem(model_data['test_f1'])
                acc_se = stats.sem(model_data['test_accuracy'])
                
                stats_results[model_name].update({
                    'f1_ci_lower': model_data['test_f1'].mean() - t_critical * f1_se,
                    'f1_ci_upper': model_data['test_f1'].mean() + t_critical * f1_se,
                    'accuracy_ci_lower': model_data['test_accuracy'].mean() - t_critical * acc_se,
                    'accuracy_ci_upper': model_data['test_accuracy'].mean() + t_critical * acc_se
                })
        
        self.stats_results = stats_results
        return stats_results
    
    def compare_models(self):
        """Perform statistical comparison between models"""
        logger.info("🔬 Performing statistical comparisons...")
        
        model_names = list(self.model_results.keys())
        
        if len(model_names) < 2:
            logger.warning("⚠️ Need at least 2 models for comparison")
            return {}
        
        comparisons = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                
                data1 = self.df_results[self.df_results['model'] == model1]
                data2 = self.df_results[self.df_results['model'] == model2]
                
                # Check if we have data for both models
                if len(data1) == 0 or len(data2) == 0:
                    continue
                
                comparison_key = f"{model1}_vs_{model2}"
                
                # Get metrics for comparison
                f1_1 = data1['test_f1'].values
                f1_2 = data2['test_f1'].values
                acc_1 = data1['test_accuracy'].values
                acc_2 = data2['test_accuracy'].values
                
                # Perform statistical tests
                comparisons[comparison_key] = {}
                
                # F1 Score comparison
                if len(f1_1) == len(f1_2):
                    # Paired t-test (same folds)
                    f1_t_stat, f1_p_value = stats.ttest_rel(f1_1, f1_2)
                    test_type = "paired"
                else:
                    # Independent t-test (different folds)
                    f1_t_stat, f1_p_value = stats.ttest_ind(f1_1, f1_2)
                    test_type = "independent"
                
                # Effect size (Cohen's d)
                if test_type == "paired":
                    f1_diff = f1_1 - f1_2
                    cohens_d_f1 = np.mean(f1_diff) / np.std(f1_diff) if np.std(f1_diff) > 0 else 0
                else:
                    pooled_std = np.sqrt(((len(f1_1) - 1) * np.var(f1_1) + (len(f1_2) - 1) * np.var(f1_2)) / (len(f1_1) + len(f1_2) - 2))
                    cohens_d_f1 = (np.mean(f1_1) - np.mean(f1_2)) / pooled_std if pooled_std > 0 else 0
                
                # Significance level
                if f1_p_value < 0.001:
                    significance_f1 = "***"
                elif f1_p_value < 0.01:
                    significance_f1 = "**"
                elif f1_p_value < 0.05:
                    significance_f1 = "*"
                else:
                    significance_f1 = "ns"
                
                comparisons[comparison_key]['f1'] = {
                    'model1_mean': np.mean(f1_1),
                    'model2_mean': np.mean(f1_2),
                    'difference': np.mean(f1_1) - np.mean(f1_2),
                    't_statistic': f1_t_stat,
                    'p_value': f1_p_value,
                    'cohens_d': cohens_d_f1,
                    'significance': significance_f1,
                    'test_type': test_type
                }
                
                # Accuracy comparison (similar process)
                if len(acc_1) == len(acc_2):
                    acc_t_stat, acc_p_value = stats.ttest_rel(acc_1, acc_2)
                else:
                    acc_t_stat, acc_p_value = stats.ttest_ind(acc_1, acc_2)
                
                if test_type == "paired":
                    acc_diff = acc_1 - acc_2
                    cohens_d_acc = np.mean(acc_diff) / np.std(acc_diff) if np.std(acc_diff) > 0 else 0
                else:
                    pooled_std = np.sqrt(((len(acc_1) - 1) * np.var(acc_1) + (len(acc_2) - 1) * np.var(acc_2)) / (len(acc_1) + len(acc_2) - 2))
                    cohens_d_acc = (np.mean(acc_1) - np.mean(acc_2)) / pooled_std if pooled_std > 0 else 0
                
                if acc_p_value < 0.001:
                    significance_acc = "***"
                elif acc_p_value < 0.01:
                    significance_acc = "**"
                elif acc_p_value < 0.05:
                    significance_acc = "*"
                else:
                    significance_acc = "ns"
                
                comparisons[comparison_key]['accuracy'] = {
                    'model1_mean': np.mean(acc_1),
                    'model2_mean': np.mean(acc_2),
                    'difference': np.mean(acc_1) - np.mean(acc_2),
                    't_statistic': acc_t_stat,
                    'p_value': acc_p_value,
                    'cohens_d': cohens_d_acc,
                    'significance': significance_acc,
                    'test_type': test_type
                }
        
        self.comparisons = comparisons
        return comparisons
    
    def create_visualizations(self, output_dir):
        """Create comprehensive visualizations"""
        logger.info("📊 Creating visualizations...")
        
        viz_dir = Path(output_dir) / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. F1 Score comparison
        plt.figure(figsize=(12, 8))
        
        # Box plot
        plt.subplot(2, 2, 1)
        sns.boxplot(data=self.df_results, x='model', y='test_f1')
        plt.title('F1 Score Distribution by Model')
        plt.xticks(rotation=45)
        
        # Violin plot  
        plt.subplot(2, 2, 2)
        sns.violinplot(data=self.df_results, x='model', y='test_f1')
        plt.title('F1 Score Density by Model')
        plt.xticks(rotation=45)
        
        # Bar plot with error bars
        plt.subplot(2, 2, 3)
        stats_df = pd.DataFrame(self.stats_results).T
        stats_df.plot(y='f1_mean', yerr='f1_std', kind='bar', ax=plt.gca(), capsize=4)
        plt.title('Mean F1 Score ± Std Dev')
        plt.xticks(rotation=45)
        plt.legend().remove()
        
        # Fold-by-fold comparison
        plt.subplot(2, 2, 4)
        for model in self.df_results['model'].unique():
            model_data = self.df_results[self.df_results['model'] == model]
            plt.plot(model_data['fold'], model_data['test_f1'], 'o-', label=model)
        plt.xlabel('Fold')
        plt.ylabel('Test F1 Score')
        plt.title('F1 Score by Fold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(viz_dir / "f1_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Accuracy comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=self.df_results, x='model', y='test_accuracy')
        plt.title('Accuracy Distribution by Model')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        for model in self.df_results['model'].unique():
            model_data = self.df_results[self.df_results['model'] == model]
            plt.plot(model_data['fold'], model_data['test_accuracy'], 'o-', label=model)
        plt.xlabel('Fold')
        plt.ylabel('Test Accuracy')
        plt.title('Accuracy by Fold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(viz_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Threshold analysis
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=self.df_results, x='model', y='threshold')
        plt.title('Threshold Distribution by Model')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(data=self.df_results, x='threshold', y='test_f1', hue='model', s=100)
        plt.title('F1 Score vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Test F1 Score')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "threshold_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Precision-Recall scatter
        plt.figure(figsize=(10, 8))
        
        for model in self.df_results['model'].unique():
            model_data = self.df_results[self.df_results['model'] == model]
            plt.scatter(model_data['test_recall'], model_data['test_precision'], 
                       label=model, s=100, alpha=0.7)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs Recall by Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add diagonal line for F1 reference
        x = np.linspace(0, 1, 100)
        for f1_level in [0.5, 0.6, 0.7, 0.8]:
            y = (f1_level * x) / (2 * x - f1_level)
            y = np.where(y > 0, y, np.nan)
            plt.plot(x, y, '--', alpha=0.3, color='gray')
            plt.text(0.9, f1_level * 0.9 / (2 * 0.9 - f1_level), f'F1={f1_level}', 
                    alpha=0.5, fontsize=8)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(viz_dir / "precision_recall.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Model configuration comparison (if available)
        config_cols = ['learning_rate', 'patience', 'class_weight_ratio', 'gradient_clip']
        available_configs = [col for col in config_cols if col in self.df_results.columns]
        
        if available_configs:
            n_configs = len(available_configs)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, config_col in enumerate(available_configs[:4]):
                if i < len(axes):
                    sns.boxplot(data=self.df_results, x='model', y=config_col, ax=axes[i])
                    axes[i].set_title(f'{config_col.replace("_", " ").title()} by Model')
                    axes[i].tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for i in range(n_configs, 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "config_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"📊 Visualizations saved to: {viz_dir}")
    
    def generate_report(self, output_dir):
        """Generate a comprehensive text report"""
        logger.info("📄 Generating comprehensive report...")
        
        report_file = Path(output_dir) / "analysis_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE MODEL ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # 1. Overview
            f.write("1. OVERVIEW\n")
            f.write("-"*40 + "\n")
            f.write(f"Total experiments: {len(self.results)}\n")
            f.write(f"Models analyzed: {list(self.model_results.keys())}\n")
            f.write(f"Folds per model: {[len(self.model_results[model]) for model in self.model_results.keys()]}\n")
            f.write("\n")
            
            # 2. Summary statistics
            f.write("2. SUMMARY STATISTICS\n")
            f.write("-"*40 + "\n")
            
            for model_name, stats in self.stats_results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  F1 Score: {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f} (range: {stats['f1_min']:.4f}-{stats['f1_max']:.4f})\n")
                f.write(f"  Accuracy: {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}\n")
                f.write(f"  Precision: {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}\n")
                f.write(f"  Recall: {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}\n")
                f.write(f"  Threshold: {stats['threshold_mean']:.4f} ± {stats['threshold_std']:.4f}\n")
                
                if 'f1_ci_lower' in stats:
                    f.write(f"  F1 95% CI: [{stats['f1_ci_lower']:.4f}, {stats['f1_ci_upper']:.4f}]\n")
                    f.write(f"  Accuracy 95% CI: [{stats['accuracy_ci_lower']:.4f}, {stats['accuracy_ci_upper']:.4f}]\n")
            
            # 3. Statistical comparisons
            if hasattr(self, 'comparisons') and self.comparisons:
                f.write("\n3. STATISTICAL COMPARISONS\n")
                f.write("-"*40 + "\n")
                
                for comparison_name, comparison_data in self.comparisons.items():
                    model1, model2 = comparison_name.split('_vs_')
                    f.write(f"\n{model1} vs {model2}:\n")
                    
                    # F1 comparison
                    f1_data = comparison_data['f1']
                    f.write(f"  F1 Score ({f1_data['test_type']} t-test):\n")
                    f.write(f"    {model1}: {f1_data['model1_mean']:.4f}\n")
                    f.write(f"    {model2}: {f1_data['model2_mean']:.4f}\n")
                    f.write(f"    Difference: {f1_data['difference']:.4f}\n")
                    f.write(f"    t-statistic: {f1_data['t_statistic']:.4f}\n")
                    f.write(f"    p-value: {f1_data['p_value']:.6f} {f1_data['significance']}\n")
                    f.write(f"    Cohen's d: {f1_data['cohens_d']:.4f}\n")
                    
                    # Accuracy comparison
                    acc_data = comparison_data['accuracy']
                    f.write(f"  Accuracy ({acc_data['test_type']} t-test):\n")
                    f.write(f"    {model1}: {acc_data['model1_mean']:.4f}\n")
                    f.write(f"    {model2}: {acc_data['model2_mean']:.4f}\n")
                    f.write(f"    Difference: {acc_data['difference']:.4f}\n")
                    f.write(f"    t-statistic: {acc_data['t_statistic']:.4f}\n")
                    f.write(f"    p-value: {acc_data['p_value']:.6f} {acc_data['significance']}\n")
                    f.write(f"    Cohen's d: {acc_data['cohens_d']:.4f}\n")
            
            # 4. Detailed results table
            f.write("\n4. DETAILED RESULTS BY FOLD\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Model':<20} {'Fold':<6} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'Thresh':<10}\n")
            f.write("-"*70 + "\n")
            
            for _, row in self.df_results.iterrows():
                f.write(f"{row['model']:<20} {row['fold']:<6} {row['test_f1']:<8.4f} "
                       f"{row['test_accuracy']:<8.4f} {row['test_precision']:<8.4f} "
                       f"{row['test_recall']:<8.4f} {row['threshold']:<10.4f}\n")
            
            # 5. Recommendations
            f.write("\n5. RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            
            # Find best performing model
            best_model = max(self.stats_results.keys(), 
                           key=lambda x: self.stats_results[x]['f1_mean'])
            best_f1 = self.stats_results[best_model]['f1_mean']
            
            f.write(f"Best performing model: {best_model} (F1: {best_f1:.4f})\n")
            
            # Check for significant differences
            if hasattr(self, 'comparisons'):
                significant_diffs = []
                for comp_name, comp_data in self.comparisons.items():
                    if comp_data['f1']['significance'] != 'ns':
                        significant_diffs.append((comp_name, comp_data['f1']['significance']))
                
                if significant_diffs:
                    f.write("\nStatistically significant differences found:\n")
                    for comp_name, significance in significant_diffs:
                        f.write(f"  - {comp_name}: {significance}\n")
                else:
                    f.write("\nNo statistically significant differences found.\n")
            
            # Stability analysis
            f.write("\nModel stability (coefficient of variation in F1):\n")
            for model_name, stats in self.stats_results.items():
                cv = stats['f1_std'] / stats['f1_mean'] if stats['f1_mean'] > 0 else 0
                stability = "High" if cv < 0.1 else "Medium" if cv < 0.2 else "Low"
                f.write(f"  - {model_name}: CV = {cv:.4f} ({stability} stability)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"📄 Report saved to: {report_file}")
    
    def save_processed_data(self, output_dir):
        """Save processed data for further analysis"""
        logger.info("💾 Saving processed data...")
        
        data_dir = Path(output_dir) / "processed_data"
        data_dir.mkdir(exist_ok=True)
        
        # Save main results DataFrame
        self.df_results.to_csv(data_dir / "results_summary.csv", index=False)
        
        # Save statistics
        stats_df = pd.DataFrame(self.stats_results).T
        stats_df.to_csv(data_dir / "model_statistics.csv")
        
        # Save comparisons
        if hasattr(self, 'comparisons'):
            comparisons_data = []
            for comp_name, comp_data in self.comparisons.items():
                row = {'comparison': comp_name}
                row.update({f"f1_{k}": v for k, v in comp_data['f1'].items()})
                row.update({f"accuracy_{k}": v for k, v in comp_data['accuracy'].items()})
                comparisons_data.append(row)
            
            pd.DataFrame(comparisons_data).to_csv(data_dir / "statistical_comparisons.csv", index=False)
        
        logger.info(f"💾 Processed data saved to: {data_dir}")
    
    def run_complete_analysis(self, output_dir):
        """Run the complete analysis pipeline"""
        logger.info("🚀 Starting complete analysis pipeline...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Step 1: Load results
        self.load_all_results()
        
        if not self.results:
            logger.error("❌ No results found! Check your results directory.")
            return
        
        # Step 2: Create DataFrame
        self.create_results_dataframe()
        
        # Step 3: Calculate statistics
        self.calculate_statistics()
        
        # Step 4: Compare models
        self.compare_models()
        
        # Step 5: Create visualizations
        self.create_visualizations(output_dir)
        
        # Step 6: Generate report
        self.generate_report(output_dir)
        
        # Step 7: Save processed data
        self.save_processed_data(output_dir)
        
        logger.info("✅ Complete analysis finished!")
        logger.info(f"📁 All outputs saved to: {output_path.absolute()}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        
        for model_name, stats in self.stats_results.items():
            print(f"{model_name}:")
            print(f"  Mean F1: {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}")
            print(f"  Mean Accuracy: {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
        
        if hasattr(self, 'comparisons'):
            print(f"\nStatistical comparisons performed: {len(self.comparisons)}")
            for comp_name, comp_data in self.comparisons.items():
                significance = comp_data['f1']['significance']
                if significance != 'ns':
                    print(f"  {comp_name}: {significance}")
        
        print(f"\n📁 Check {output_path} for detailed results!")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Analyze model training results')
    parser.add_argument('--results_dir', type=str, default='improved_results',
                       help='Directory containing training results')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                       help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not Path(args.results_dir).exists():
        logger.error(f"❌ Results directory not found: {args.results_dir}")
        return
    
    # Create analyzer
    analyzer = ResultsAnalyzer(args.results_dir)
    
    # Run complete analysis
    analyzer.run_complete_analysis(args.output_dir)

if __name__ == "__main__":
    main()