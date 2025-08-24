#!/usr/bin/env python3
"""
Improved Model Training Script
=============================

Uses pre-split data folds and model-specific configurations
to address the issues found in mt5-base training.

Usage:
    python train_models_improved.py --fold 1 --model mt5-base
    python train_models_improved.py --fold all --model all
"""

import os
import json
import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import copy
import logging

# Configure GPU memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========================
# 🔧 MODEL-SPECIFIC CONFIGURATIONS
# ========================
MODEL_CONFIGS = {
    'mt5-base': {
        'learning_rate': 5e-6,           # ⚡ נמוך יותר עבור מודל כללי
        'warmup_ratio': 0.15,            # 🔥 יותר warmup
        'patience': 5,                   # 🕐 יותר סבלנות
        'weight_decay': 0.01,            # 🛡️ regularization חזק
        'gradient_clip': 0.5,            # ✂️ clipping חזק יותר
        'class_weight_ratio': 2.5,      # ⚖️ משקל גבוה לציטוטים
        'threshold_epochs': [0, 2, 4],  # 🎯 עוד נקודות כיוון
        'min_epochs': 3,                 # 📊 מינימום epochs
        'dropout_rate': 0.3,             # 🚫 dropout גבוה
        'label_smoothing': 0.1           # 🎭 label smoothing
    },
    'mt5-mlm-final': {
        'learning_rate': 2e-5,           # ⚡ כמו במקור
        'warmup_ratio': 0.1,             # 🔥 warmup סטנדרטי
        'patience': 3,                   # 🕐 כמו במקור
        'weight_decay': 0.001,           # 🛡️ regularization קל
        'gradient_clip': 1.0,            # ✂️ כמו במקור
        'class_weight_ratio': 1.5,      # ⚖️ משקל קל לציטוטים
        'threshold_epochs': [0, 2],     # 🎯 כמו במקור
        'min_epochs': 2,                 # 📊 מינימום נמוך
        'dropout_rate': 0.1,             # 🚫 dropout נמוך
        'label_smoothing': 0.0           # 🎭 ללא label smoothing
    }
}

# ========================
# 📊 IMPROVED DATASET WITH CLASS WEIGHTS
# ========================
class ImprovedLegalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=1024, class_weights=None):
        # Same legal prompt as before
        prompt = """מערכת חיזוי ציטוטים משפטיים מתמחה
תחום: דין פלילי - מדיניות ענישה
מטרה: חיזוי ציטוטים בין פסקי דין על בסיס דמיון בעובדות כתב האישום
קריטריונים: ציטוט רלוונטי אם הוא תומך בהחלטת טווח העונש (לא הליכים, הגדרות, או פסקי דין לא קשורים)
פסק דין א' יצטט פסק דין ב' אם יש דמיון בעבירות ובנסיבות העוולות המוצגות בכתבי האישום.
שאלה: בהתבסס על עובדות כתב האישום, האם צפוי שפסק דין א' יצטט פסק דין ב' לתמיכה בטווח הענישה?
"""
        
        self.inputs = []
        for idx, row in df.iterrows():
            legal_input = f"""{prompt}

עובדות כתב אישום - פסק דין א':
{row['gpt_facts_a']}

עובדות כתב אישום - פסק דין ב':
{row['gpt_facts_b']}

על בסיס דמיון העבירות והנסיבות, האם פסק דין א' יצטט פסק דין ב' לתמיכה במדיניות הענישה?"""
            
            self.inputs.append(legal_input)
        
        self.targets = df["label"].apply(lambda l: "כן" if l == 1 else "לא").tolist()
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.class_weights = class_weights
        
        logger.info(f"Dataset created: {len(self.inputs)} samples")
        logger.info(f"Label distribution: {np.bincount(self.labels)}")
        
        if class_weights:
            logger.info(f"Class weights: {class_weights}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]
        
        input_enc = self.tokenizer(
            input_text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors="pt"
        )
        
        target_enc = self.tokenizer(
            target_text, 
            padding='max_length', 
            truncation=True, 
            max_length=5,
            return_tensors="pt"
        )
        
        labels = target_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Add sample weight if class weights provided
        sample_weight = 1.0
        if self.class_weights is not None:
            sample_weight = self.class_weights[self.labels[idx]]
        
        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
            "numeric_label": self.labels[idx],
            "sample_weight": sample_weight
        }

# ========================
# 🎯 IMPROVED CLASSIFICATION WITH BETTER THRESHOLD SEARCH
# ========================
def classify_with_threshold(model, tokenizer, input_ids, attention_mask, threshold=0.0):
    """Improved classification with more stable scoring"""
    with torch.no_grad():
        batch_size = input_ids.shape[0]
        
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
        decoder_input_ids[:, 0] = tokenizer.pad_token_id
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
        logits = outputs.logits[:, -1, :]
        
        # Improved token selection
        yes_tokens = [259, 1903]  # כן
        no_tokens = [1124]        # לא
        
        predictions = []
        scores = []
        
        for batch_idx in range(batch_size):
            batch_logits = logits[batch_idx]
            
            # Apply softmax for better probability interpretation
            probs = torch.softmax(batch_logits, dim=0)
            
            yes_score = torch.mean(probs[yes_tokens]).item()
            no_score = torch.mean(probs[no_tokens]).item()
            
            score_diff = yes_score - no_score
            
            prediction = 1 if score_diff > threshold else 0
            predicted_text = "כן" if prediction == 1 else "לא"
            
            predictions.append(prediction)
            scores.append({
                'prediction': prediction,
                'predicted_text': predicted_text,
                'score_diff': score_diff,
                'yes_score': yes_score,
                'no_score': no_score,
                'confidence': abs(score_diff)
            })
        
        return predictions, scores

def find_optimal_threshold(model, tokenizer, dataloader, device, true_labels, n_thresholds=100):
    """Improved threshold search with more granular search"""
    logger.info("🔍 Finding optimal threshold...")
    
    all_score_diffs = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting scores", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            _, scores = classify_with_threshold(
                model, tokenizer,
                batch["input_ids"],
                batch["attention_mask"],
                threshold=0.0
            )
            
            for score in scores:
                all_score_diffs.append(score['score_diff'])
    
    # More comprehensive threshold search
    score_min, score_max = min(all_score_diffs), max(all_score_diffs)
    
    # Linear search
    linear_thresholds = np.linspace(score_min, score_max, n_thresholds)
    
    # Percentile-based search
    percentile_thresholds = [np.percentile(all_score_diffs, p) for p in range(5, 96, 5)]
    
    # Combine and deduplicate
    all_thresholds = np.unique(np.concatenate([linear_thresholds, percentile_thresholds]))
    
    best_threshold = 0.0
    best_f1 = 0.0
    best_metrics = {}
    
    for threshold in all_thresholds:
        predictions = []
        
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            batch_preds, _ = classify_with_threshold(
                model, tokenizer,
                batch["input_ids"],
                batch["attention_mask"],
                threshold=threshold
            )
            
            predictions.extend(batch_preds)
        
        predictions = np.array(predictions)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'f1': f1,
                'precision': precision_score(true_labels, predictions, zero_division=0),
                'recall': recall_score(true_labels, predictions, zero_division=0),
                'accuracy': np.mean(predictions == true_labels)
            }
    
    logger.info(f"Best threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
    return best_threshold, best_metrics

# ========================
# 🏋️ IMPROVED TRAINING FUNCTION
# ========================
def train_model_improved(model, model_name, train_loader, val_loader, config, tokenizer, device, epochs=7):
    """Improved training with model-specific configurations"""
    
    # Calculate class weights
    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch["numeric_label"].numpy())
    train_labels = np.array(train_labels)
    
    # Compute class weights
    classes = np.unique(train_labels)
    sklearn_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    
    # Apply model-specific weight ratio
    class_weights = {
        0: 1.0,
        1: config['class_weight_ratio']
    }
    
    logger.info(f"Using class weights: {class_weights}")
    
    # Create optimizer with model-specific settings
    optimizer = AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create learning rate scheduler
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(config['warmup_ratio'] * num_training_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Training setup: LR={config['learning_rate']}, Warmup={num_warmup_steps}/{num_training_steps} steps")
    
    # Training state
    best_val_f1 = 0
    best_val_threshold = 0.0
    best_model_state = None
    epochs_no_improve = 0
    
    # Get validation labels
    val_labels = []
    for batch in val_loader:
        val_labels.extend(batch["numeric_label"].numpy())
    val_labels = np.array(val_labels)
    
    # Initial threshold calibration
    logger.info("📏 Initial threshold calibration")
    initial_threshold, _ = find_optimal_threshold(model, tokenizer, val_loader, device, val_labels)
    best_val_threshold = initial_threshold
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            
            # Apply class weighting to loss
            if 'sample_weight' in batch:
                # Weight the loss by sample weights
                sample_weights = batch['sample_weight'].to(device)
                loss = loss * sample_weights.mean()
            
            loss.backward()
            
            # Model-specific gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation with fixed threshold
        val_f1, val_metrics = evaluate_model_improved(
            model, val_loader, tokenizer, device, 
            fix_threshold=best_val_threshold
        )
        
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val F1={val_f1:.4f}, Best={best_val_f1:.4f}")
        
        # Threshold recalibration at specific epochs
        if epoch in config['threshold_epochs']:
            logger.info(f"📏 Recalibrating threshold at epoch {epoch+1}")
            new_threshold, _ = find_optimal_threshold(model, tokenizer, val_loader, device, val_labels)
            best_val_threshold = new_threshold
            logger.info(f"📌 New threshold: {best_val_threshold:.4f}")
        
        # Check for improvement (but not before min_epochs)
        if epoch >= config['min_epochs'] - 1:
            if val_f1 > best_val_f1 or best_model_state is None:
                best_val_f1 = val_f1
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                logger.info(f"✅ Best model updated (F1: {best_val_f1:.4f})")
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement for {epochs_no_improve} epochs.")
                
                if epochs_no_improve >= config['patience']:
                    logger.info(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                    break
        else:
            # Always save in first epochs before min_epochs
            if val_f1 > best_val_f1 or best_model_state is None:
                best_val_f1 = val_f1
                best_model_state = copy.deepcopy(model.state_dict())
                logger.info(f"✅ Model saved (epoch {epoch+1} < min_epochs)")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_val_threshold, best_val_f1

# ========================
# 📊 IMPROVED EVALUATION FUNCTION
# ========================
def evaluate_model_improved(model, dataloader, tokenizer, device, fix_threshold=None, tune_threshold=False):
    """Improved evaluation with better metrics"""
    model.eval()
    
    # Collect true labels
    true_labels = []
    for batch in dataloader:
        true_labels.extend(batch["numeric_label"].numpy())
    true_labels = np.array(true_labels)
    
    all_predictions = []
    all_scores = []
    
    if fix_threshold is not None:
        # Use fixed threshold
        threshold = fix_threshold
        logger.info(f"Using fixed threshold: {threshold:.4f}")
    elif tune_threshold:
        # Tune threshold on this data (avoid on test set!)
        threshold, _ = find_optimal_threshold(model, tokenizer, dataloader, device, true_labels)
    else:
        # Use default threshold
        threshold = 0.0
    
    # Collect predictions
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            predictions, scores = classify_with_threshold(
                model, tokenizer,
                batch["input_ids"],
                batch["attention_mask"],
                threshold=threshold
            )
            
            all_predictions.extend(predictions)
            all_scores.extend(scores)
    
    predictions = np.array(all_predictions)
    
    # Calculate comprehensive metrics
    f1 = f1_score(true_labels, predictions, zero_division=0)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    accuracy = np.mean(predictions == true_labels)
    
    # Additional metrics
    true_pos = np.sum((predictions == 1) & (true_labels == 1))
    false_pos = np.sum((predictions == 1) & (true_labels == 0))
    true_neg = np.sum((predictions == 0) & (true_labels == 0))
    false_neg = np.sum((predictions == 0) & (true_labels == 1))
    
    logger.info(f"\n📊 EVALUATION RESULTS:")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Threshold: {threshold:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"TP: {true_pos}, FP: {false_pos}")
    logger.info(f"FN: {false_neg}, TN: {true_neg}")
    
    logger.info(f"\nPrediction Distribution: {np.bincount(predictions)}")
    logger.info(f"True Label Distribution: {np.bincount(true_labels)}")
    
    # Classification report
    if len(np.unique(predictions)) > 1:
        logger.info(f"\nClassification Report:")
        logger.info(classification_report(true_labels, predictions))
    
    return f1, {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'threshold': threshold,
        'predictions': predictions,
        'true_labels': true_labels,
        'scores': all_scores,
        'confusion_matrix': {
            'tp': true_pos, 'fp': false_pos,
            'tn': true_neg, 'fn': false_neg
        }
    }

# ========================
# 📂 DATA LOADING FUNCTIONS
# ========================
def load_fold_data(data_dir, fold_num):
    """Load data for a specific fold"""
    fold_dir = Path(data_dir) / f"fold_{fold_num}"
    
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
    
    train_df = pd.read_csv(fold_dir / "train.csv")
    val_df = pd.read_csv(fold_dir / "val.csv")
    test_df = pd.read_csv(fold_dir / "test.csv")
    
    logger.info(f"Loaded fold {fold_num}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    
    return train_df, val_df, test_df

def get_available_folds(data_dir):
    """Get list of available fold numbers"""
    data_path = Path(data_dir)
    fold_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('fold_')]
    fold_nums = [int(d.name.split('_')[1]) for d in fold_dirs]
    return sorted(fold_nums)

# ========================
# 🚀 MAIN EXECUTION FUNCTIONS
# ========================
def train_single_fold(model_path, fold_num, data_dir, output_dir, max_len=1024, batch_size=4, epochs=7):
    """Train a single model on a single fold"""
    
    model_name = model_path.split('/')[-1]
    config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['mt5-mlm-final'])
    
    logger.info(f"🚀 Training {model_name} on fold {fold_num}")
    logger.info(f"Config: {config}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    train_df, val_df, test_df = load_fold_data(data_dir, fold_num)
    
    # Create datasets
    train_dataset = ImprovedLegalDataset(train_df, tokenizer, max_len)
    val_dataset = ImprovedLegalDataset(val_df, tokenizer, max_len)
    test_dataset = ImprovedLegalDataset(test_df, tokenizer, max_len)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    
    # Train model
    best_threshold, best_val_f1 = train_model_improved(
        model, model_name, train_loader, val_loader, 
        config, tokenizer, device, epochs
    )
    
    # Final evaluation on test set
    logger.info("🧪 Final test evaluation...")
    test_f1, test_metrics = evaluate_model_improved(
        model, test_loader, tokenizer, device,
        fix_threshold=best_threshold
    )
    
    # Save results
    results = {
        'model': model_name,
        'fold': fold_num,
        'config': config,
        'best_val_f1': best_val_f1,
        'best_threshold': best_threshold,
        'test_f1': test_f1,
        'test_metrics': test_metrics
    }
    
    # Save model and results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    fold_output_dir = output_path / f"{model_name}_fold_{fold_num}"
    fold_output_dir.mkdir(exist_ok=True)
    
    # Save model
    model.save_pretrained(fold_output_dir / "model")
    tokenizer.save_pretrained(fold_output_dir / "tokenizer")
    
    # Save results
    import json
    with open(fold_output_dir / "results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = copy.deepcopy(results)
        for key in ['predictions', 'true_labels']:
            if key in json_results['test_metrics']:
                json_results['test_metrics'][key] = json_results['test_metrics'][key].tolist()
        json.dump(json_results, f, indent=2, default=str)
    
    logger.info(f"✅ Fold {fold_num} complete: Test F1 = {test_f1:.4f}")
    logger.info(f"📁 Results saved to: {fold_output_dir}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results

def train_all_folds(model_paths, data_dir, output_dir, max_len=1024, batch_size=4, epochs=7):
    """Train all models on all available folds"""
    
    # Get available folds
    available_folds = get_available_folds(data_dir)
    logger.info(f"Available folds: {available_folds}")
    
    all_results = []
    
    for model_path in model_paths:
        model_name = model_path.split('/')[-1]
        logger.info(f"\n🎯 Starting training for model: {model_name}")
        
        model_results = []
        
        for fold_num in available_folds:
            try:
                result = train_single_fold(
                    model_path, fold_num, data_dir, output_dir,
                    max_len, batch_size, epochs
                )
                model_results.append(result)
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"❌ Error training {model_name} fold {fold_num}: {e}")
                continue
        
        # Calculate model summary
        if model_results:
            test_f1_scores = [r['test_f1'] for r in model_results]
            logger.info(f"\n📊 {model_name} Summary:")
            logger.info(f"Test F1 scores: {test_f1_scores}")
            logger.info(f"Mean F1: {np.mean(test_f1_scores):.4f} ± {np.std(test_f1_scores):.4f}")
    
    return all_results

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Train models with improved configurations')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['all', 'mt5-base', 'mt5-mlm-final'],
                       help='Model to train')
    parser.add_argument('--fold', type=str, default='all',
                       help='Fold number to train (or "all" for all folds)')
    parser.add_argument('--data_dir', type=str, default='data_splits_5fold',
                       help='Directory containing fold data')
    parser.add_argument('--output_dir', type=str, default='improved_results',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=7,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--max_len', type=int, default=1024,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Define model paths
    model_paths = {
        'mt5-base': 'google/mt5-base',
        'mt5-mlm-final': '/home/liorkob/M.Sc/thesis/t5/mt5-mlm-final'
    }
    
    # Select models to train
    if args.model == 'all':
        selected_models = list(model_paths.values())
    else:
        selected_models = [model_paths[args.model]]
    
    logger.info(f"🎯 Training models: {[p.split('/')[-1] for p in selected_models]}")
    
    # Train models
    if args.fold == 'all':
        # Train all folds
        results = train_all_folds(
            selected_models, args.data_dir, args.output_dir,
            args.max_len, args.batch_size, args.epochs
        )
    else:
        # Train single fold
        fold_num = int(args.fold)
        results = []
        
        for model_path in selected_models:
            result = train_single_fold(
                model_path, fold_num, args.data_dir, args.output_dir,
                args.max_len, args.batch_size, args.epochs
            )
            results.append(result)
    
    logger.info("🎉 Training complete!")
    logger.info(f"📁 All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()