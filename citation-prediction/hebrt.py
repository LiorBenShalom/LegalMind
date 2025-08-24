

# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoModel, AutoTokenizer
# from torch.optim import Adam
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
# from scipy.stats import ttest_rel, t
# from tqdm import tqdm
# import warnings
# import random
# warnings.filterwarnings('ignore')

# def set_seed(seed=42):
#     """Set all random seeds for reproducibility"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # --- Dataset and Model Classes (unchanged from your original) ---
# class CrossEncoderVerdictDataset(Dataset):
#     def __init__(self, df, tokenizer, max_len=512):
#         self.df = df.reset_index(drop=True)
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         text = f"[CLS] {row['gpt_facts_a']} [SEP] {row['gpt_facts_b']} [SEP]"
#         enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
#         return {
#             'input_ids': enc['input_ids'].squeeze(),
#             'attention_mask': enc['attention_mask'].squeeze(),
#             'label': torch.tensor(row['label'], dtype=torch.float)
#         }

# class CrossEncoderHeBERT(nn.Module):
#     def __init__(self, model_name):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(model_name)
#         hidden = self.encoder.config.hidden_size
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1)
#         )

#     def forward(self, input_ids, attention_mask):
#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
#         return self.classifier(pooled).squeeze(-1)

# def train_model_fold(model, train_loader, val_loader, optimizer, device, epochs=25, patience=5, verbose=False):
#     """Train model for one fold with early stopping and better stability"""
#     criterion = nn.BCEWithLogitsLoss()
#     best_auc = 0
#     no_improve = 0
    
#     # Add learning rate scheduler for stability
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='max', factor=0.5, patience=2, verbose=verbose
#     )
    
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
        
#         for batch in train_loader:
#             for key in batch:
#                 batch[key] = batch[key].to(device)

#             logits = model(batch['input_ids'], batch['attention_mask'])
#             loss = criterion(logits, batch['label'])

#             optimizer.zero_grad()
#             loss.backward()
            
#             # Add gradient clipping for stability
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
#             total_loss += loss.item()

#         # Validation
#         model.eval()
#         val_probs, val_labels = [], []
#         with torch.no_grad():
#             for batch in val_loader:
#                 for key in batch:
#                     batch[key] = batch[key].to(device)
#                 logits = model(batch['input_ids'], batch['attention_mask'])
#                 prob = torch.sigmoid(logits).cpu().numpy()
#                 label = batch['label'].cpu().numpy()
#                 val_probs.extend(prob)
#                 val_labels.extend(label)
        
#         val_auc = roc_auc_score(val_labels, val_probs)
#         scheduler.step(val_auc)  # Update learning rate
        
#         if verbose:
#             current_lr = optimizer.param_groups[0]['lr']
#             print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}, LR: {current_lr:.2e}")
        
#         # Early stopping with minimum improvement threshold
#         if val_auc > best_auc + 1e-4:  # Require minimum improvement
#             best_auc = val_auc
#             no_improve = 0
#             best_state = model.state_dict().copy()
#         else:
#             no_improve += 1
#             if no_improve >= patience:
#                 if verbose:
#                     print("Early stopping triggered")
#                 break
    
#     # Load best model
#     model.load_state_dict(best_state)
#     return model, best_auc

# def evaluate_model_fold(model, test_loader, device):
#     """Evaluate model on test set and return AUC"""
#     model.eval()
#     probs, targets = [], []
    
#     with torch.no_grad():
#         for batch in test_loader:
#             for key in batch:
#                 batch[key] = batch[key].to(device)
#             logits = model(batch['input_ids'], batch['attention_mask'])
#             prob = torch.sigmoid(logits).cpu().numpy()
#             label = batch['label'].cpu().numpy()
#             probs.extend(prob)
#             targets.extend(label)
    
#     auc = roc_auc_score(targets, probs)
#     return auc, probs, targets

# def run_kfold_comparison_fixed(df_full, baseline_model_name, finetuned_model_name, k=5, epochs=25, patience=5, batch_size=8, random_state=42):
#     """
#     Fixed k-fold cross-validation with better stability and reproducibility
#     """
    
#     # Set seed at the beginning
#     set_seed(random_state)
    
#     # Initialize tokenizers
#     baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)
#     finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)
    
#     # Set up k-fold with fixed random state
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
#     baseline_aucs = []
#     finetuned_aucs = []
#     fold_results = []
    
#     print(f"🚀 Starting {k}-Fold Cross-Validation (Fixed Version)")
#     print(f"Baseline Model: {baseline_model_name}")
#     print(f"Fine-tuned Model: {finetuned_model_name}")
#     print("=" * 80)
    
#     for fold, (train_idx, test_idx) in enumerate(skf.split(df_full, df_full['label'])):
#         print(f"\n📁 FOLD {fold + 1}/{k}")
#         print("-" * 40)
        
#         # Reset seed for each fold for consistency
#         set_seed(random_state + fold)
        
#         # Split data
#         df_train_fold = df_full.iloc[train_idx].reset_index(drop=True)
#         df_test_fold = df_full.iloc[test_idx].reset_index(drop=True)
        
#         # Further split training into train/val (80/20)
#         train_size = int(0.8 * len(df_train_fold))
#         df_train = df_train_fold[:train_size]
#         df_val = df_train_fold[train_size:]
        
#         print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test_fold)}")
        
#         # === BASELINE MODEL ===
#         print("\n🔵 Training Baseline Model...")
        
#         # Create datasets and loaders for baseline
#         train_dataset_base = CrossEncoderVerdictDataset(df_train, baseline_tokenizer)
#         val_dataset_base = CrossEncoderVerdictDataset(df_val, baseline_tokenizer)
#         test_dataset_base = CrossEncoderVerdictDataset(df_test_fold, baseline_tokenizer)
        
#         train_loader_base = DataLoader(train_dataset_base, batch_size=batch_size, shuffle=True)
#         val_loader_base = DataLoader(val_dataset_base, batch_size=batch_size)
#         test_loader_base = DataLoader(test_dataset_base, batch_size=batch_size)
        
#         # Initialize and train baseline model
#         baseline_model = CrossEncoderHeBERT(baseline_model_name).to(device)
#         baseline_optimizer = Adam(baseline_model.parameters(), lr=1e-5, weight_decay=1e-4)  # Lower LR, add weight decay
        
#         baseline_model, baseline_val_auc = train_model_fold(
#             baseline_model, train_loader_base, val_loader_base, 
#             baseline_optimizer, device, epochs, patience, verbose=False
#         )
        
#         # Evaluate baseline
#         baseline_auc, _, _ = evaluate_model_fold(baseline_model, test_loader_base, device)
#         baseline_aucs.append(baseline_auc)
#         print(f"Baseline - Val AUC: {baseline_val_auc:.4f}, Test AUC: {baseline_auc:.4f}")
        
#         # Clean up baseline model
#         del baseline_model, baseline_optimizer
#         torch.cuda.empty_cache()
        
#         # Reset seed again for fine-tuned model
#         set_seed(random_state + fold)
        
#         # === FINE-TUNED MODEL ===
#         print("\n🟢 Training Fine-tuned Model...")
        
#         # Create datasets and loaders for fine-tuned model
#         train_dataset_ft = CrossEncoderVerdictDataset(df_train, finetuned_tokenizer)
#         val_dataset_ft = CrossEncoderVerdictDataset(df_val, finetuned_tokenizer)
#         test_dataset_ft = CrossEncoderVerdictDataset(df_test_fold, finetuned_tokenizer)
        
#         train_loader_ft = DataLoader(train_dataset_ft, batch_size=batch_size, shuffle=True)
#         val_loader_ft = DataLoader(val_dataset_ft, batch_size=batch_size)
#         test_loader_ft = DataLoader(test_dataset_ft, batch_size=batch_size)
        
#         # Initialize and train fine-tuned model
#         finetuned_model = CrossEncoderHeBERT(finetuned_model_name).to(device)
#         finetuned_optimizer = Adam(finetuned_model.parameters(), lr=1e-5, weight_decay=1e-4)  # Same params
        
#         finetuned_model, finetuned_val_auc = train_model_fold(
#             finetuned_model, train_loader_ft, val_loader_ft, 
#             finetuned_optimizer, device, epochs, patience, verbose=False
#         )
        
#         # Evaluate fine-tuned
#         finetuned_auc, _, _ = evaluate_model_fold(finetuned_model, test_loader_ft, device)
#         finetuned_aucs.append(finetuned_auc)
#         print(f"Fine-tuned - Val AUC: {finetuned_val_auc:.4f}, Test AUC: {finetuned_auc:.4f}")
        
#         # Calculate improvement
#         improvement = finetuned_auc - baseline_auc
#         print(f"Improvement: {improvement:+.4f}")
        
#         # Store fold results
#         fold_results.append({
#             'fold': fold + 1,
#             'baseline_auc': baseline_auc,
#             'finetuned_auc': finetuned_auc,
#             'improvement': improvement,
#             'baseline_val_auc': baseline_val_auc,
#             'finetuned_val_auc': finetuned_val_auc
#         })
        
#         # Clean up fine-tuned model
#         del finetuned_model, finetuned_optimizer
#         torch.cuda.empty_cache()
    
#     # === STATISTICAL ANALYSIS ===
#     print("\n" + "=" * 80)
#     print("📊 STATISTICAL ANALYSIS")
#     print("=" * 80)
    
#     # Convert to numpy arrays
#     baseline_aucs = np.array(baseline_aucs)
#     finetuned_aucs = np.array(finetuned_aucs)
#     improvements = finetuned_aucs - baseline_aucs
    
#     # Summary statistics
#     print(f"\n📈 Summary Statistics:")
#     print(f"Baseline AUC:    {baseline_aucs.mean():.4f} ± {baseline_aucs.std():.4f}")
#     print(f"Fine-tuned AUC:  {finetuned_aucs.mean():.4f} ± {finetuned_aucs.std():.4f}")
#     print(f"Mean Improvement: {improvements.mean():+.4f} ± {improvements.std():.4f}")
    
#     # Paired t-test
#     t_stat, p_value = ttest_rel(finetuned_aucs, baseline_aucs)
    
#     print(f"\n🧪 Paired T-Test Results:")
#     print(f"t-statistic: {t_stat:.4f}")
#     print(f"p-value: {p_value:.2e}")
    
#     # Significance interpretation
#     alpha = 0.05
#     if p_value < alpha:
#         significance = "✅ SIGNIFICANT"
#         interpretation = f"The improvement is statistically significant (p < {alpha})"
#     else:
#         significance = "❌ NOT SIGNIFICANT"
#         interpretation = f"The improvement is not statistically significant (p ≥ {alpha})"
    
#     print(f"Result: {significance}")
#     print(f"Interpretation: {interpretation}")
    
#     # Effect size (Cohen's d)
#     pooled_std = np.sqrt((baseline_aucs.var() + finetuned_aucs.var()) / 2)
#     cohens_d = improvements.mean() / pooled_std
#     print(f"Effect Size (Cohen's d): {cohens_d:.4f}")
    
#     # Effect size interpretation
#     if abs(cohens_d) < 0.2:
#         effect_size_interp = "negligible"
#     elif abs(cohens_d) < 0.5:
#         effect_size_interp = "small"
#     elif abs(cohens_d) < 0.8:
#         effect_size_interp = "medium"
#     else:
#         effect_size_interp = "large"
    
#     print(f"Effect Size Interpretation: {effect_size_interp}")
    
#     # Fold-wise results table
#     print(f"\n📋 Fold-wise Results:")
#     print("Fold | Baseline | Fine-tuned | Improvement | Base Val | FT Val")
#     print("-" * 70)
#     for result in fold_results:
#         print(f"{result['fold']:4d} | {result['baseline_auc']:8.4f} | {result['finetuned_auc']:10.4f} | {result['improvement']:+10.4f} | {result['baseline_val_auc']:8.4f} | {result['finetuned_val_auc']:6.4f}")
    
#     # Confidence interval for mean improvement
#     confidence_level = 0.95
#     df_ci = len(improvements) - 1
#     t_critical = t.ppf((1 + confidence_level) / 2, df_ci)
#     margin_error = t_critical * (improvements.std() / np.sqrt(len(improvements)))
#     ci_lower = improvements.mean() - margin_error
#     ci_upper = improvements.mean() + margin_error
    
#     print(f"\n🎯 {confidence_level*100}% Confidence Interval for Mean Improvement:")
#     print(f"[{ci_lower:+.4f}, {ci_upper:+.4f}]")
    
#     return {
#         'baseline_aucs': baseline_aucs,
#         'finetuned_aucs': finetuned_aucs,
#         'improvements': improvements,
#         't_statistic': t_stat,
#         'p_value': p_value,
#         'cohens_d': cohens_d,
#         'mean_improvement': improvements.mean(),
#         'std_improvement': improvements.std(),
#         'confidence_interval': (ci_lower, ci_upper),
#         'fold_results': fold_results
#     }

# # === MAIN EXECUTION ===
# if __name__ == "__main__":
#     # Set seed at the very beginning
#     set_seed(42)
    
#     # Load your full dataset
#     print("📂 Loading datasets...")
#     df_train = pd.read_csv("/home/liorkob/M.Sc/thesis/citation-prediction/data_splits/crossencoder_train.csv")
#     df_val = pd.read_csv("/home/liorkob/M.Sc/thesis/citation-prediction/data_splits/crossencoder_val.csv")
#     df_test = pd.read_csv("/home/liorkob/M.Sc/thesis/citation-prediction/data_splits/crossencoder_test.csv")
    
#     # Combine all splits for k-fold CV
#     df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
#     print(f"Total dataset size: {len(df_full)} samples")
#     print(f"Label distribution: {df_full['label'].value_counts().to_dict()}")
    
#     # Define experiments - run one at a time for debugging
#     experiments = [
#         {
#             "name": "HeBERT MLM vs Baseline",
#             "baseline": "avichr/heBERT",
#             "finetuned": "/home/liorkob/M.Sc/thesis/pre-train/models/hebert-mlm-3k-drugs/final"
#         },
#         {
#             "name": "Legal-HeBERT MLM vs Baseline",
#             "baseline": "avichr/Legal-heBERT",
#             "finetuned": "/home/liorkob/M.Sc/thesis/pre-train/models/Legal-heBERT-mlm-3k-drugs/final"
#         },
#         {
#             "name": "mBERT MLM vs Baseline",
#             "baseline": "bert-base-multilingual-cased",
#             "finetuned": "/home/liorkob/M.Sc/thesis/pre-train/models/mBERT-mlm-3k-drugs/final"
#         }
#     ]
    
#     # Run experiments one by one
#     all_results = {}
    
#     for exp in experiments:
#         print(f"\n\n🚨 Running Experiment: {exp['name']}")
#         print("=" * 100)
        
#         results = run_kfold_comparison_fixed(
#             df_full=df_full,
#             baseline_model_name=exp["baseline"],
#             finetuned_model_name=exp["finetuned"],
#             k=5,
#             epochs=25,  # Increased epochs
#             patience=5,  # Increased patience
#             batch_size=8,
#             random_state=42
#         )
        
#         all_results[exp['name']] = results
        
#         print(f"\n📌 {exp['name']} SUMMARY:")
#         print(f"Mean AUC improvement: {results['mean_improvement']:+.4f}")
#         print(f"P-value: {results['p_value']:.2e}")
#         if results["p_value"] < 0.05:
#             print("✅ Statistically significant improvement!")
#         else:
#             print("❌ Not statistically significant.")
        
#         print("\n" + "="*50 + " END OF EXPERIMENT " + "="*50)
    
#     # Final summary of all experiments
#     print("\n\n🏆 FINAL SUMMARY OF ALL EXPERIMENTS")
#     print("=" * 80)
#     for name, results in all_results.items():
#         status = "✅ SIGNIFICANT" if results['p_value'] < 0.05 else "❌ NOT SIGNIFICANT"
#         print(f"{name}: {results['mean_improvement']:+.4f} AUC improvement, p={results['p_value']:.2e} - {status}")
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel, t
from tqdm import tqdm
import warnings
import random
import pickle
import os
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_and_save_folds(df_full, output_dir, random_state=42):
    """Generate 10-fold splits and save to CSV files"""
    
    print("🔄 Generating 10-fold cross-validation splits...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate stratified k-fold splits
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    
    fold_info = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(df_full, df_full['label'])):
        fold_num = fold + 1
        fold_dir = os.path.join(output_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        
        print(f"📁 Generating Fold {fold_num}/10...")
        
        # Get train and test data for this fold
        df_train_fold = df_full.iloc[train_idx].reset_index(drop=True)
        df_test_fold = df_full.iloc[test_idx].reset_index(drop=True)
        
        # Split training data into train/val (80/20)
        train_size = int(0.8 * len(df_train_fold))
        df_train = df_train_fold[:train_size].reset_index(drop=True)
        df_val = df_train_fold[train_size:].reset_index(drop=True)
        
        # Save CSV files
        train_file = os.path.join(fold_dir, "train.csv")
        val_file = os.path.join(fold_dir, "val.csv")
        test_file = os.path.join(fold_dir, "test.csv")
        
        df_train.to_csv(train_file, index=False)
        df_val.to_csv(val_file, index=False)
        df_test_fold.to_csv(test_file, index=False)
        
        # Store fold information
        fold_info.append({
            'fold': fold_num,
            'train_samples': len(df_train),
            'val_samples': len(df_val),
            'test_samples': len(df_test_fold),
            'train_pos': df_train['label'].sum(),
            'val_pos': df_val['label'].sum(),
            'test_pos': df_test_fold['label'].sum(),
            'train_file': train_file,
            'val_file': val_file,
            'test_file': test_file
        })
        
        print(f"   Train: {len(df_train)} ({df_train['label'].sum()} pos)")
        print(f"   Val: {len(df_val)} ({df_val['label'].sum()} pos)")
        print(f"   Test: {len(df_test_fold)} ({df_test_fold['label'].sum()} pos)")
    
    # Save fold information
    fold_info_file = os.path.join(output_dir, "fold_info.csv")
    pd.DataFrame(fold_info).to_csv(fold_info_file, index=False)
    
    print(f"\n✅ All 10 folds saved to: {output_dir}")
    print(f"📊 Fold information saved to: {fold_info_file}")
    
    return fold_info

def load_fold_data(fold_dir):
    """Load train, val, test data for a specific fold"""
    train_file = os.path.join(fold_dir, "train.csv")
    val_file = os.path.join(fold_dir, "val.csv")
    test_file = os.path.join(fold_dir, "test.csv")
    
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)
    
    return df_train, df_val, df_test

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Dataset and Model Classes ---
class CrossEncoderVerdictDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"[CLS] {row['gpt_facts_a']} [SEP] {row['gpt_facts_b']} [SEP]"
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label': torch.tensor(row['label'], dtype=torch.float)
        }

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(logits, targets)

class CrossEncoderHeBERT(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(pooled).squeeze(-1)

def train_model_fold(model, train_loader, val_loader, optimizer, device, train_df, epochs=25, patience=5, verbose=False):
    """Train model for one fold with weighted loss and early stopping"""
    # Calculate class weights from training data
    num_pos = train_df['label'].sum()
    num_neg = len(train_df) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float).to(device)
    criterion = WeightedBCELoss(pos_weight=pos_weight)
    
    if verbose:
        print(f"Class balance - Positive: {num_pos}, Negative: {num_neg}, Weight: {pos_weight.item():.3f}")
    
    best_auc = 0
    no_improve = 0
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=verbose
    )
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            for key in batch:
                batch[key] = batch[key].to(device)

            logits = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(logits, batch['label'])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                for key in batch:
                    batch[key] = batch[key].to(device)
                logits = model(batch['input_ids'], batch['attention_mask'])
                prob = torch.sigmoid(logits).cpu().numpy()
                label = batch['label'].cpu().numpy()
                val_probs.extend(prob)
                val_labels.extend(label)
        
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_auc)
        
        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}, LR: {current_lr:.2e}")
        
        # Early stopping
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            no_improve = 0
            best_state = model.state_dict().copy()
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print("Early stopping triggered")
                break
    
    model.load_state_dict(best_state)
    return model, best_auc

def evaluate_model_fold(model, test_loader, device):
    """Evaluate model on test set and return AUC"""
    model.eval()
    probs, targets = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            for key in batch:
                batch[key] = batch[key].to(device)
            logits = model(batch['input_ids'], batch['attention_mask'])
            prob = torch.sigmoid(logits).cpu().numpy()
            label = batch['label'].cpu().numpy()
            probs.extend(prob)
            targets.extend(label)
    
    auc = roc_auc_score(targets, probs)
    return auc

def evaluate_non_trained_model(model_name, tokenizer, test_df, device, batch_size=8):
    """Evaluate a non-trained model (only with classifier head initialized)"""
    test_dataset = CrossEncoderVerdictDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = CrossEncoderHeBERT(model_name).to(device)
    auc = evaluate_model_fold(model, test_loader, device)
    
    del model
    torch.cuda.empty_cache()
    return auc

def save_checkpoint(filename, data):
    """Save checkpoint to file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ Checkpoint saved: {filename}")
    except:
        print(f"❌ Failed to save checkpoint: {filename}")

def load_checkpoint(filename):
    """Load checkpoint from file"""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except:
            print(f"❌ Failed to load checkpoint: {filename}")
    return None

def run_10fold_comparison_from_splits(folds_dir, baseline_model_name, finetuned_model_name, checkpoint_file, epochs=25, patience=5, batch_size=8, random_state=42):
    """Run 10-fold cross-validation using pre-generated fold splits"""
    
    # Try to load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        print(f"📂 Resuming from checkpoint: {checkpoint['completed_folds']}/10 folds completed")
        start_fold = checkpoint['completed_folds']
        baseline_trained_aucs = checkpoint['baseline_trained_aucs']
        finetuned_trained_aucs = checkpoint['finetuned_trained_aucs']
        baseline_non_trained_aucs = checkpoint['baseline_non_trained_aucs']
        finetuned_non_trained_aucs = checkpoint['finetuned_non_trained_aucs']
    else:
        print(f"🚀 Starting new 10-fold experiment using pre-generated splits")
        start_fold = 0
        baseline_trained_aucs = []
        finetuned_trained_aucs = []
        baseline_non_trained_aucs = []
        finetuned_non_trained_aucs = []
    
    set_seed(random_state)
    
    # Initialize tokenizers
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)
    finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)
    
    for fold in range(start_fold, 10):
        fold_num = fold + 1
        fold_dir = os.path.join(folds_dir, f"fold_{fold_num}")
        
        print(f"\n📁 FOLD {fold_num}/10")
        print("-" * 40)
        
        set_seed(random_state + fold)
        
        # Load pre-generated fold data
        df_train, df_val, df_test = load_fold_data(fold_dir)
        
        print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
        
        # === NON-TRAINED EVALUATIONS ===
        print("🔵 Evaluating Non-Trained Models...")
        baseline_non_trained_auc = evaluate_non_trained_model(
            baseline_model_name, baseline_tokenizer, df_test, device, batch_size
        )
        finetuned_non_trained_auc = evaluate_non_trained_model(
            finetuned_model_name, finetuned_tokenizer, df_test, device, batch_size
        )
        
        baseline_non_trained_aucs.append(baseline_non_trained_auc)
        finetuned_non_trained_aucs.append(finetuned_non_trained_auc)
        
        print(f"Baseline (Non-Trained): {baseline_non_trained_auc:.4f}")
        print(f"Fine-tuned (Non-Trained): {finetuned_non_trained_auc:.4f}")
        
        # === TRAINED BASELINE MODEL ===
        print("🔵 Training Baseline Model...")
        
        train_dataset_base = CrossEncoderVerdictDataset(df_train, baseline_tokenizer)
        val_dataset_base = CrossEncoderVerdictDataset(df_val, baseline_tokenizer)
        test_dataset_base = CrossEncoderVerdictDataset(df_test, baseline_tokenizer)
        
        train_loader_base = DataLoader(train_dataset_base, batch_size=batch_size, shuffle=True)
        val_loader_base = DataLoader(val_dataset_base, batch_size=batch_size)
        test_loader_base = DataLoader(test_dataset_base, batch_size=batch_size)
        
        baseline_model = CrossEncoderHeBERT(baseline_model_name).to(device)
        baseline_optimizer = Adam(baseline_model.parameters(), lr=1e-5, weight_decay=1e-4)
        
        baseline_model, baseline_val_auc = train_model_fold(
            baseline_model, train_loader_base, val_loader_base, 
            baseline_optimizer, device, df_train, epochs, patience, verbose=False
        )
        
        baseline_trained_auc = evaluate_model_fold(baseline_model, test_loader_base, device)
        baseline_trained_aucs.append(baseline_trained_auc)
        print(f"Baseline (Trained): {baseline_trained_auc:.4f}")
        
        del baseline_model, baseline_optimizer
        torch.cuda.empty_cache()
        
        # === TRAINED FINE-TUNED MODEL ===
        print("🟢 Training Fine-tuned Model...")
        set_seed(random_state + fold)
        
        train_dataset_ft = CrossEncoderVerdictDataset(df_train, finetuned_tokenizer)
        val_dataset_ft = CrossEncoderVerdictDataset(df_val, finetuned_tokenizer)
        test_dataset_ft = CrossEncoderVerdictDataset(df_test, finetuned_tokenizer)
        
        train_loader_ft = DataLoader(train_dataset_ft, batch_size=batch_size, shuffle=True)
        val_loader_ft = DataLoader(val_dataset_ft, batch_size=batch_size)
        test_loader_ft = DataLoader(test_dataset_ft, batch_size=batch_size)
        
        finetuned_model = CrossEncoderHeBERT(finetuned_model_name).to(device)
        finetuned_optimizer = Adam(finetuned_model.parameters(), lr=1e-5, weight_decay=1e-4)
        
        finetuned_model, finetuned_val_auc = train_model_fold(
            finetuned_model, train_loader_ft, val_loader_ft, 
            finetuned_optimizer, device, df_train, epochs, patience, verbose=False
        )
        
        finetuned_trained_auc = evaluate_model_fold(finetuned_model, test_loader_ft, device)
        finetuned_trained_aucs.append(finetuned_trained_auc)
        print(f"Fine-tuned (Trained): {finetuned_trained_auc:.4f}")
        
        del finetuned_model, finetuned_optimizer
        torch.cuda.empty_cache()
        
        # Save checkpoint after each fold
        checkpoint_data = {
            'completed_folds': fold + 1,
            'baseline_trained_aucs': baseline_trained_aucs,
            'finetuned_trained_aucs': finetuned_trained_aucs,
            'baseline_non_trained_aucs': baseline_non_trained_aucs,
            'finetuned_non_trained_aucs': finetuned_non_trained_aucs
        }
        save_checkpoint(checkpoint_file, checkpoint_data)
    
    # Convert to numpy arrays and calculate final statistics
    baseline_trained_aucs = np.array(baseline_trained_aucs)
    finetuned_trained_aucs = np.array(finetuned_trained_aucs)
    baseline_non_trained_aucs = np.array(baseline_non_trained_aucs)
    finetuned_non_trained_aucs = np.array(finetuned_non_trained_aucs)
    
    trained_improvements = finetuned_trained_aucs - baseline_trained_aucs
    non_trained_improvements = finetuned_non_trained_aucs - baseline_non_trained_aucs
    
    # Statistical tests
    t_stat_trained, p_value_trained = ttest_rel(finetuned_trained_aucs, baseline_trained_aucs)
    t_stat_non_trained, p_value_non_trained = ttest_rel(finetuned_non_trained_aucs, baseline_non_trained_aucs)
    
    # Calculate Cohen's d effect sizes
    def cohens_d(x, y):
        pooled_std = np.sqrt((x.var() + y.var()) / 2)
        return (x.mean() - y.mean()) / pooled_std
    
    cohens_d_trained = cohens_d(finetuned_trained_aucs, baseline_trained_aucs)
    cohens_d_non_trained = cohens_d(finetuned_non_trained_aucs, baseline_non_trained_aucs)
    
    return {
        'baseline_trained_aucs': baseline_trained_aucs,
        'finetuned_trained_aucs': finetuned_trained_aucs,
        'baseline_non_trained_aucs': baseline_non_trained_aucs,
        'finetuned_non_trained_aucs': finetuned_non_trained_aucs,
        'trained_improvements': trained_improvements,
        'non_trained_improvements': non_trained_improvements,
        'p_values': {
            'trained': p_value_trained,
            'non_trained': p_value_non_trained
        },
        'cohens_d': {
            'trained': cohens_d_trained,
            'non_trained': cohens_d_non_trained
        }
    }

def display_results_summary(results_dict):
    """Display results in the requested format with Cohen's d"""
    print("\n" + "="*80)
    print("📊 10-FOLD CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    print("\n1. 📋 BASELINE MODEL PERFORMANCE (No MLM, No Citation Training)")
    print("-" * 60)
    
    for model_name, results in results_dict.items():
        baseline_mean = results['baseline_non_trained_aucs'].mean()
        baseline_std = results['baseline_non_trained_aucs'].std()
        print(f"{model_name:15}: {baseline_mean:.4f} ± {baseline_std:.4f}")
    
    print("\n2. 📊 IMPACT OF CITATION PREDICTION TASK TRAINING")
    print("-" * 100)
    print(f"{'Model':<15} {'Condition':<12} {'No Training':<12} {'After Training':<15} {'Training Impact':<15} {'P-value':<10} {'Cohens d':<10}")
    print("-" * 100)
    
    for model_name, results in results_dict.items():
        # Baseline row
        baseline_no_train = results['baseline_non_trained_aucs'].mean()
        baseline_after_train = results['baseline_trained_aucs'].mean()
        baseline_impact = baseline_after_train - baseline_no_train
        
        print(f"{model_name:<15} {'Baseline':<12} {baseline_no_train:<12.4f} {baseline_after_train:<15.4f} {baseline_impact:<15.4f} {'N/A':<10} {'N/A':<10}")
        
        # MLM row
        mlm_no_train = results['finetuned_non_trained_aucs'].mean()
        mlm_after_train = results['finetuned_trained_aucs'].mean()
        mlm_impact = mlm_after_train - mlm_no_train
        
        print(f"{'':<15} {'+MLM':<12} {mlm_no_train:<12.4f} {mlm_after_train:<15.4f} {mlm_impact:<15.4f} {'N/A':<10} {'N/A':<10}")
        
        # MLM Benefit row
        mlm_benefit_no_train = results['non_trained_improvements'].mean()
        mlm_benefit_after_train = results['trained_improvements'].mean()
        mlm_net_benefit = mlm_benefit_after_train - mlm_benefit_no_train
        p_val_trained = results['p_values']['trained']
        cohens_d_trained = results['cohens_d']['trained']
        
        significance = "✅" if p_val_trained < 0.05 else "❌"
        
        # Interpret Cohen's d effect size
        if abs(cohens_d_trained) < 0.2:
            effect_size = "Small"
        elif abs(cohens_d_trained) < 0.5:
            effect_size = "Small"
        elif abs(cohens_d_trained) < 0.8:
            effect_size = "Medium"
        else:
            effect_size = "Large"
        
        print(f"{'':<15} {'MLM Benefit':<12} {mlm_benefit_no_train:<12.4f} {mlm_benefit_after_train:<15.4f} {mlm_net_benefit:<15.4f} {p_val_trained:<.3f} {significance:<3} {cohens_d_trained:<6.3f} ({effect_size})")
        print("-" * 100)
    
    print("\n3. 📏 EFFECT SIZE INTERPRETATION:")
    print("   Cohen's d: |0.2| = Small, |0.5| = Medium, |0.8| = Large effect")
    print("   Positive d = MLM model better, Negative d = Baseline model better")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    set_seed(42)
    
    # Configuration
    data_dir = "/home/liorkob/M.Sc/thesis/citation-prediction/data_splits"
    folds_output_dir = "/home/liorkob/M.Sc/thesis/citation-prediction/data_splits_10fold"
    
    # Step 1: Generate folds (run this once)
    generate_folds = True  # Set to False if folds already generated
    
    if generate_folds:
        print("📂 Loading original datasets...")
        df_train = pd.read_csv(f"{data_dir}/crossencoder_train.csv")
        df_val = pd.read_csv(f"{data_dir}/crossencoder_val.csv")
        df_test = pd.read_csv(f"{data_dir}/crossencoder_test.csv")
        
        df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)
        print(f"Total dataset size: {len(df_full)} samples")
        
        # Generate and save 10 folds
        fold_info = generate_and_save_folds(df_full, folds_output_dir, random_state=42)
    else:
        print(f"📂 Using existing folds from: {folds_output_dir}")
    
    # Step 2: Run experiments using saved folds
    print("\n" + "="*80)
    print("🚀 STARTING 10-FOLD EXPERIMENTS")
    print("="*80)
    
    # Define experiments
    experiments = [
        ("HeBERT", "avichr/heBERT", "/home/liorkob/M.Sc/thesis/pre-train/models/hebert-mlm-3k-drugs/final"),
        ("Legal-HeBERT", "avichr/Legal-heBERT", "/home/liorkob/M.Sc/thesis/pre-train/models/Legal-heBERT-mlm-3k-drugs/final"),
        ("mBERT", "bert-base-multilingual-cased", "/home/liorkob/M.Sc/thesis/pre-train/models/mBERT-mlm-3k-drugs/final")
    ]
    
    # Run 10-fold experiments using pre-generated splits
    results_dict = {}
    for model_name, baseline_model, finetuned_model in experiments:
        print(f"\n🚀 Running 10-Fold Experiment: {model_name}")
        checkpoint_file = f"{model_name}_10fold_checkpoint.pkl"
        
        results = run_10fold_comparison_from_splits(
            folds_dir=folds_output_dir,
            baseline_model_name=baseline_model,
            finetuned_model_name=finetuned_model,
            checkpoint_file=checkpoint_file,
            epochs=25,
            patience=5,
            batch_size=8,
            random_state=42
        )
        results_dict[model_name] = results
    
    # Display results in requested format
    display_results_summary(results_dict)
    
    print(f"\n🏁 10-FOLD EXPERIMENTS COMPLETED!")
    print(f"📁 Fold splits saved in: {folds_output_dir}")