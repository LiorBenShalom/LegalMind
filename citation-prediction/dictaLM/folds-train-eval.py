import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.stats import ttest_rel
from tqdm import tqdm
import warnings
import random
import pickle
import os
import json
print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')

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
        
        print(f"  Train: {len(df_train)} ({df_train['label'].sum()} pos)")
        print(f"  Val: {len(df_val)} ({df_val['label'].sum()} pos)")
        print(f"  Test: {len(df_test_fold)} ({df_test_fold['label'].sum()} pos)")
    
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

# --- Dataset Class for DictaLM2 ---
class DictaLM2CitationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=2048, is_training=True):
        self.df = df.reset_index(drop=True).dropna()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_training = is_training
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label'])
        answer = "כן" if label == 1 else "לא"  # Hebrew yes/no for DictaLM2
        
        # Truncate text to avoid OOM
        facts_a = str(row['gpt_facts_a'])
        facts_b = str(row['gpt_facts_b'])
        
        prompt = f"""מערכת חיזוי ציטוטים משפטיים מתמחה
תחום: דין פלילי - מדיניות ענישה
מטרה: חיזוי ציטוטים בין פסקי דין על בסיס דמיון בעובדות כתב האישום
קריטריונים: ציטוט רלוונטי אם הוא תומך בהחלטת טווח העונש (לא הליכים, הגדרות, או פסקי דין לא קשורים)
פסק דין א' יצטט פסק דין ב' אם יש דמיון בעבירות ובנסיבות העוולות המוצגות בכתבי האישום.
שאלה: בהתבסס על עובדות כתב האישום, האם צפוי שפסק דין א' יצטט פסק דין ב' לתמיכה בטווח הענישה?
        

עובדות כתב אישום - פסק דין א':
{facts_a}

עובדות כתב אישום - פסק דין ב':
{facts_b}

על בסיס דמיון העבירות והנסיבות, האם פסק דין א' יצטט פסק דין ב' לתמיכה במדיניות הענישה?

תשובה:
"""
        
        if self.is_training:
            full_text = prompt + answer
            tokens = self.tokenizer.encode(full_text, max_length=self.max_len, truncation=True, add_special_tokens=True)
            prompt_tokens = self.tokenizer.encode(prompt, max_length=self.max_len-10, truncation=True, add_special_tokens=True)
            
            # Ensure we have space for the answer
            if len(prompt_tokens) >= self.max_len - 5:
                prompt_tokens = prompt_tokens[:self.max_len-5]
            
            answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
            full_tokens = prompt_tokens + answer_tokens
            
            input_ids = full_tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(full_tokens))
            attention_mask = [1] * len(full_tokens) + [0] * (self.max_len - len(full_tokens))
            
            # Only train on answer tokens
            labels = [-100] * len(prompt_tokens) + answer_tokens + [-100] * (self.max_len - len(full_tokens))
            
            return {
                "input_ids": torch.tensor(input_ids[:self.max_len]),
                "attention_mask": torch.tensor(attention_mask[:self.max_len]),
                "labels": torch.tensor(labels[:self.max_len])
            }
        else:
            tokens = self.tokenizer.encode(prompt, max_length=self.max_len, truncation=True, add_special_tokens=True)
            input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * (self.max_len - len(tokens))
            
            return {
                "input_ids": torch.tensor(input_ids[:self.max_len]),
                "attention_mask": torch.tensor(attention_mask[:self.max_len]),
                "true_label": torch.tensor(label),
                "prompt_length": len(tokens)
            }

# --- DictaLM2 Model Setup ---
def setup_dictalm2_model(model_path):
    """Setup DictaLM2 model without LoRA - full model training"""
    print(f"Loading DictaLM2 model from: {model_path}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    return model, tokenizer

def train_dictalm2_model_fold(model, train_loader, val_loader, optimizer, device,
                              epochs=2, patience=2, verbose=False, tokenizer=None):
    """Train DictaLM2 model for one fold"""
    best_auc = 0.0
    best_sd = None
    no_improve = 0

    model.train()
    for epoch in range(epochs):
        if verbose:
            print(f"🔍 Training Epoch {epoch+1}...")

        total_loss, step = 0.0, 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            try:
                in_dev = model.get_input_embeddings().weight.device
                input_ids = batch["input_ids"].to(in_dev)
                attention_mask = batch["attention_mask"].to(in_dev)
                labels = batch["labels"].to(in_dev)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                if torch.isnan(loss):
                    if verbose:
                        print("NaN loss—skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += float(loss.item())
                step += 1
                
                if step % 200 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if verbose:
                    print(f"Training error: {e}, skipping batch...")
                continue
        
        # Validation
        with torch.no_grad():
            val_metrics = evaluate_dictalm2_model_fold(model, val_loader, device, tokenizer=tokenizer)
        val_auc = float(val_metrics.get('auc', 0.0))
        
        if verbose:
            print(f"Epoch {epoch+1} | Loss: {total_loss/max(step, 1):.4f}, Val AUC: {val_auc:.4f}")
        
        # Early stopping + snapshot של המצב הטוב
        if val_auc > best_auc + 1e-3:
            best_auc = val_auc
            no_improve = 0
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print("Early stopping triggered")
                break
    
    if best_sd is not None:
        model.load_state_dict(best_sd, strict=False)
    return model, best_auc

def evaluate_dictalm2_model_fold(model, test_loader, device, tokenizer=None, max_samples=None):
    """Evaluate DictaLM2 model on test set and return metrics including AUC"""
    assert tokenizer is not None, "tokenizer must be provided to evaluate_dictalm2_model_fold"
    model.eval()
    predictions, true_labels, probabilities = [], [], []
    
    # Build robust target sequences (may be multi-token)
    yes_ids = tokenizer.encode(" כן", add_special_tokens=False)
    no_ids  = tokenizer.encode(" לא", add_special_tokens=False)

    def seq_logprob(model, input_ids, attention_mask, start_pos, target_ids):
        logp = 0.0
        cur_ids = input_ids.clone()
        cur_mask = attention_mask.clone()
        for t in target_ids:
            out = model(input_ids=cur_ids, attention_mask=cur_mask)
            step_logits = out.logits[0, start_pos-1, :]
            step_logp = torch.log_softmax(step_logits, dim=-1)[t]
            logp += step_logp.item()
            cur_ids[0, start_pos] = t
            cur_mask[0, start_pos] = 1
            start_pos += 1
        return logp
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if max_samples is not None and i >= max_samples:
                break
                
            try:
                in_dev = model.get_input_embeddings().weight.device
                input_ids = batch["input_ids"].to(in_dev)
                attention_mask = batch["attention_mask"].to(in_dev)
                true_label = batch["true_label"].item()
                prompt_length = batch["prompt_length"].item()
                
                yes_lp = seq_logprob(model, input_ids, attention_mask, prompt_length, yes_ids)
                no_lp  = seq_logprob(model, input_ids, attention_mask, prompt_length,  no_ids)
                
                m = max(yes_lp, no_lp)
                yes_prob = np.exp(yes_lp - m) / (np.exp(yes_lp - m) + np.exp(no_lp - m))
                
                pred = 1 if yes_prob > 0.5 else 0
                
                predictions.append(pred)
                true_labels.append(true_label)
                probabilities.append(yes_prob)
                
            except Exception:
                predictions.append(0)
                true_labels.append(true_label)
                probabilities.append(0.5)
                continue
    
    if not predictions:
        model.train()
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'auc': 0.5}
    
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'auc': roc_auc_score(true_labels, probabilities) if len(set(true_labels)) > 1 else 0.5
    }
    
    model.train()
    return metrics

def evaluate_non_trained_dictalm2_model(model_path, test_df, device, batch_size=1):
    """Evaluate a non-trained DictaLM2 model"""
    model, tokenizer = setup_dictalm2_model(model_path)
    test_dataset = DictaLM2CitationDataset(test_df, tokenizer, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    metrics = evaluate_dictalm2_model_fold(model, test_loader, device, tokenizer=tokenizer)
    
    del model
    torch.cuda.empty_cache()
    return metrics

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

def run_dictalm2_10fold_comparison_from_splits(folds_dir, baseline_model_path, finetuned_model_path, checkpoint_file, epochs=2, patience=2, batch_size=1, random_state=42):
    """Run 10-fold cross-validation for DictaLM2 models using pre-generated fold splits"""
    
    # Try to load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        print(f"📂 Resuming from checkpoint: {checkpoint['completed_folds']}/10 folds completed")
        start_fold = checkpoint['completed_folds']
        baseline_trained_metrics = checkpoint['baseline_trained_metrics']
        finetuned_trained_metrics = checkpoint['finetuned_trained_metrics']
        baseline_non_trained_metrics = checkpoint['baseline_non_trained_metrics']
        finetuned_non_trained_metrics = checkpoint['finetuned_non_trained_metrics']
    else:
        print(f"🚀 Starting new 10-fold DictaLM2 experiment using pre-generated splits")
        start_fold = 0
        baseline_trained_metrics = []
        finetuned_trained_metrics = []
        baseline_non_trained_metrics = []
        finetuned_non_trained_metrics = []
    
    set_seed(random_state)
    
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
        baseline_non_trained = evaluate_non_trained_dictalm2_model(baseline_model_path, df_test, device, batch_size)
        finetuned_non_trained = evaluate_non_trained_dictalm2_model(finetuned_model_path, df_test, device, batch_size)
        
        baseline_non_trained_metrics.append(baseline_non_trained)
        finetuned_non_trained_metrics.append(finetuned_non_trained)
        
        print(f"Baseline (Non-Trained) AUC: {baseline_non_trained['auc']:.4f}")
        print(f"Fine-tuned (Non-Trained) AUC: {finetuned_non_trained['auc']:.4f}")
        
        # === TRAINED BASELINE MODEL ===
        print("🔵 Training Baseline Model...")
        baseline_model, baseline_tokenizer = setup_dictalm2_model(baseline_model_path)
        
        train_dataset_base = DictaLM2CitationDataset(df_train, baseline_tokenizer, is_training=True)
        val_dataset_base = DictaLM2CitationDataset(df_val, baseline_tokenizer, is_training=False)
        test_dataset_base = DictaLM2CitationDataset(df_test, baseline_tokenizer, is_training=False)
        
        train_loader_base = DataLoader(train_dataset_base, batch_size=batch_size, shuffle=True)
        val_loader_base = DataLoader(val_dataset_base, batch_size=batch_size)
        test_loader_base = DataLoader(test_dataset_base, batch_size=batch_size)
        
        baseline_optimizer = AdamW(baseline_model.parameters(), lr=1e-3, weight_decay=1e-4)
                
        # Baseline
        baseline_model, baseline_val_auc = train_dictalm2_model_fold(
            baseline_model, train_loader_base, val_loader_base, baseline_optimizer, device,
            epochs, patience, verbose=False, tokenizer=baseline_tokenizer
        )

        
        baseline_trained = evaluate_dictalm2_model_fold(baseline_model, test_loader_base, device, tokenizer=baseline_tokenizer)
        baseline_trained_metrics.append(baseline_trained)
        
        print(f"Baseline (Trained) AUC: {baseline_trained['auc']:.4f}")
        
        del baseline_model, baseline_optimizer
        torch.cuda.empty_cache()
        
        # === TRAINED FINE-TUNED MODEL ===
        print("🟢 Training Fine-tuned Model...")
        set_seed(random_state + fold)
        
        finetuned_model, finetuned_tokenizer = setup_dictalm2_model(finetuned_model_path)
        
        train_dataset_ft = DictaLM2CitationDataset(df_train, finetuned_tokenizer, is_training=True)
        val_dataset_ft = DictaLM2CitationDataset(df_val, finetuned_tokenizer, is_training=False)
        test_dataset_ft = DictaLM2CitationDataset(df_test, finetuned_tokenizer, is_training=False)
        
        train_loader_ft = DataLoader(train_dataset_ft, batch_size=batch_size, shuffle=True)
        val_loader_ft = DataLoader(val_dataset_ft, batch_size=batch_size)
        test_loader_ft = DataLoader(test_dataset_ft, batch_size=batch_size)
        
        finetuned_optimizer = AdamW(finetuned_model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Fine-tuned
        finetuned_model, finetuned_val_auc = train_dictalm2_model_fold(
            finetuned_model, train_loader_ft, val_loader_ft, finetuned_optimizer, device,
            epochs, patience, verbose=False, tokenizer=finetuned_tokenizer
        )
        
        finetuned_trained = evaluate_dictalm2_model_fold(finetuned_model, test_loader_ft, device, tokenizer=finetuned_tokenizer)
        finetuned_trained_metrics.append(finetuned_trained)
        
        print(f"Fine-tuned (Trained) AUC: {finetuned_trained['auc']:.4f}")
        
        del finetuned_model, finetuned_optimizer
        torch.cuda.empty_cache()
        
        # Save checkpoint after each fold
        checkpoint_data = {
            'completed_folds': fold + 1,
            'baseline_trained_metrics': baseline_trained_metrics,
            'finetuned_trained_metrics': finetuned_trained_metrics,
            'baseline_non_trained_metrics': baseline_non_trained_metrics,
            'finetuned_non_trained_metrics': finetuned_non_trained_metrics
        }
        save_checkpoint(checkpoint_file, checkpoint_data)
    
    # Extract AUC scores for statistical tests
    baseline_trained_aucs = np.array([m['auc'] for m in baseline_trained_metrics])
    finetuned_trained_aucs = np.array([m['auc'] for m in finetuned_trained_metrics])
    baseline_non_trained_aucs = np.array([m['auc'] for m in baseline_non_trained_metrics])
    finetuned_non_trained_aucs = np.array([m['auc'] for m in finetuned_non_trained_metrics])
    
    trained_improvements = finetuned_trained_aucs - baseline_trained_aucs
    non_trained_improvements = finetuned_non_trained_aucs - baseline_non_trained_aucs
    
    # Statistical tests
    t_stat_trained, p_value_trained = ttest_rel(finetuned_trained_aucs, baseline_trained_aucs)
    t_stat_non_trained, p_value_non_trained = ttest_rel(finetuned_non_trained_aucs, baseline_non_trained_aucs)
    
    # Paired Cohen's d
    def cohens_d_paired(x, y):
        diff = x - y
        return diff.mean() / (diff.std(ddof=1) + 1e-12)
    
    cohens_d_trained = cohens_d_paired(finetuned_trained_aucs, baseline_trained_aucs)
    cohens_d_non_trained = cohens_d_paired(finetuned_non_trained_aucs, baseline_non_trained_aucs)
    
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

def display_dictalm2_results_summary(results_dict):
    """Display results in the requested format with Cohen's d - AUC as main metric"""
    print("\n" + "="*80)
    print("📊 DICTALM2 10-FOLD CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    print("\n1. 📋 BASELINE MODEL PERFORMANCE (No Citation Training)")
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
        
        # Fine-tuned row
        ft_no_train = results['finetuned_non_trained_aucs'].mean()
        ft_after_train = results['finetuned_trained_aucs'].mean()
        ft_impact = ft_after_train - ft_no_train
        print(f"{'':<15} {'+Fine-tuned':<12} {ft_no_train:<12.4f} {ft_after_train:<15.4f} {ft_impact:<15.4f} {'N/A':<10} {'N/A':<10}")
        
        # Fine-tuned Benefit row
        ft_benefit_no_train = results['non_trained_improvements'].mean()
        ft_benefit_after_train = results['trained_improvements'].mean()
        ft_net_benefit = ft_benefit_after_train - ft_benefit_no_train
        p_val_trained = results['p_values']['trained']
        cohens_d_trained = results['cohens_d']['trained']
        significance = "✅" if p_val_trained < 0.05 else "❌"
        
        if abs(cohens_d_trained) < 0.2:
            effect_size = "Small"
        elif abs(cohens_d_trained) < 0.5:
            effect_size = "Small"
        elif abs(cohens_d_trained) < 0.8:
            effect_size = "Medium"
        else:
            effect_size = "Large"
        
        print(f"{'':<15} {'FT Benefit':<12} {ft_benefit_no_train:<12.4f} {ft_benefit_after_train:<15.4f} {ft_net_benefit:<15.4f} {p_val_trained:<.3f} {significance:<3} {cohens_d_trained:<6.3f} ({effect_size})")
    
    print("-" * 100)
    print("\n3. 📏 EFFECT SIZE INTERPRETATION:")
    print("   Cohen's d: |0.2| = Small, |0.5| = Medium, |0.8| = Large effect")
    print("   Positive d = Fine-tuned model better, Negative d = Baseline model better")
    print("   🎯 Main Metric: AUC (Area Under the ROC Curve)")
    print("   AUC Range: 0.5 (random) to 1.0 (perfect classification)")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    set_seed(42)
    
    # Configuration - Update these paths for your DictaLM2 models
    BASE_MODEL = "dicta-il/dictalm2.0-instruct"
    FT_MODEL = "/home/liorkob/extensive-clm-final"  # Update this path
    
    # Use existing splits directory
    folds_output_dir = "/home/liorkob/M.Sc/thesis/citation-prediction/data_splits_10fold"
    
    # Step 1: Skip fold generation since we're using existing splits
    print(f"📂 Using existing folds from: {folds_output_dir}")
    
    # Verify folds exist
    if not os.path.exists(folds_output_dir):
        raise FileNotFoundError(f"Folds directory not found: {folds_output_dir}")
    
    for fold_num in range(1, 11):
        fold_dir = os.path.join(folds_output_dir, f"fold_{fold_num}")
        if not os.path.exists(fold_dir):
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
        for file_name in ["train.csv", "val.csv", "test.csv"]:
            file_path = os.path.join(fold_dir, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
    
    print("✅ All 10 folds verified successfully")
    
    # Step 2: Run experiments using saved folds
    print("\n" + "="*80)
    print("🚀 STARTING DICTALM2 10-FOLD EXPERIMENTS")
    print("="*80)
    
    # Define experiment
    print(f"\n🚀 Running 10-Fold Experiment: DictaLM2 Citation Prediction")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Fine-tuned Model: {FT_MODEL}")
    
    checkpoint_file = "dictalm2_citation_10fold_checkpoint_new.pkl"
    
    results = run_dictalm2_10fold_comparison_from_splits(
        folds_dir=folds_output_dir,
        baseline_model_path=BASE_MODEL,
        finetuned_model_path=FT_MODEL,
        checkpoint_file=checkpoint_file,
        epochs=5,  # Increase from 2
        patience=3,  # Increase patience
        batch_size=1,  # Small batch size due to memory constraints
        random_state=42
    )
    
    results_dict = {"DictaLM2.0": results}
    
    # Display results in requested format
    display_dictalm2_results_summary(results_dict)
    
    # Save detailed results
    results_file = "dictalm2_10fold_detailed_results.json"
    detailed_results = {
        'baseline_trained_aucs': results['baseline_trained_aucs'].tolist(),
        'finetuned_trained_aucs': results['finetuned_trained_aucs'].tolist(),
        'baseline_non_trained_aucs': results['baseline_non_trained_aucs'].tolist(),
        'finetuned_non_trained_aucs': results['finetuned_non_trained_aucs'].tolist(),
        'trained_improvements': results['trained_improvements'].tolist(),
        'non_trained_improvements': results['non_trained_improvements'].tolist(),
        'p_values': results['p_values'],
        'cohens_d': results['cohens_d'],
        'summary_stats': {
            'baseline_trained_mean': float(results['baseline_trained_aucs'].mean()),
            'baseline_trained_std': float(results['baseline_trained_aucs'].std()),
            'finetuned_trained_mean': float(results['finetuned_trained_aucs'].mean()),
            'finetuned_trained_std': float(results['finetuned_trained_aucs'].std()),
            'baseline_non_trained_mean': float(results['baseline_non_trained_aucs'].mean()),
            'baseline_non_trained_std': float(results['baseline_non_trained_aucs'].std()),
            'finetuned_non_trained_mean': float(results['finetuned_non_trained_aucs'].mean()),
            'finetuned_non_trained_std': float(results['finetuned_non_trained_aucs'].std())
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n🏁 DICTALM2 10-FOLD EXPERIMENTS COMPLETED!")
    print(f"📁 Fold splits saved in: {folds_output_dir}")
    print(f"📊 Detailed results saved in: {results_file}")
    print(f"💾 Checkpoint saved in: {checkpoint_file}")
    
    # Print final summary
    print(f"\n📈 FINAL SUMMARY (AUC - Main Metric):")
    print(f"Baseline Model (No Training): {results['baseline_non_trained_aucs'].mean():.4f} ± {results['baseline_non_trained_aucs'].std():.4f}")
    print(f"Fine-tuned Model (No Training): {results['finetuned_non_trained_aucs'].mean():.4f} ± {results['finetuned_non_trained_aucs'].std():.4f}")
    print(f"Baseline Model (After Training): {results['baseline_trained_aucs'].mean():.4f} ± {results['baseline_trained_aucs'].std():.4f}")
    print(f"Fine-tuned Model (After Training): {results['finetuned_trained_aucs'].mean():.4f} ± {results['finetuned_trained_aucs'].std():.4f}")
    print(f"P-value (Trained): {results['p_values']['trained']:.4f}")
    print(f"Cohen's d (Trained): {results['cohens_d']['trained']:.4f}")
    
    significance_trained = "✅ Significant" if results['p_values']['trained'] < 0.05 else "❌ Not Significant"
    print(f"Statistical Significance: {significance_trained}")
    
    if abs(results['cohens_d']['trained']) >= 0.8:
        effect_interpretation = "Large effect size"
    elif abs(results['cohens_d']['trained']) >= 0.5:
        effect_interpretation = "Medium effect size"
    elif abs(results['cohens_d']['trained']) >= 0.2:
        effect_interpretation = "Small effect size"
    else:
        effect_interpretation = "Negligible effect size"
    
    print(f"Effect Size: {effect_interpretation}")
    
    if results['cohens_d']['trained'] > 0:
        print("🎯 Fine-tuned model performs better than baseline model")
    else:
        print("🎯 Baseline model performs better than fine-tuned model")
