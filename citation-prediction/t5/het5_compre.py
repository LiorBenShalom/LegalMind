
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import shutil
checkpoint_dir = "checkpoints_legal_citation_auc_focused"
print("📁 Checking if checkpoint dir exists...")
if os.path.exists(checkpoint_dir):
    print("🧹 Deleting checkpoint folder...")
    shutil.rmtree(checkpoint_dir)
    print("✅ Checkpoint folder deleted")
else:
    print("ℹ️ No checkpoint folder to delete")

print("✅ Imports and cleanup done, moving to tokenizer loading")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print("📦 Transformers import OK")


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
import numpy as np

from scipy import stats
from collections import defaultdict
import copy
import os
import pickle
import json
from datetime import datetime

# ========================
# 🔧 SAFE MEMORY MANAGEMENT & CHECKPOINTING
# ========================

import os
import json
import pickle
from datetime import datetime
from collections import defaultdict

def nested_defaultdict_list():
    return defaultdict(list)

class SafeExperimentManager:
    def __init__(self, experiment_name="legal_citation_experiment_t5"):
        self.experiment_name = experiment_name
        self.checkpoint_dir = f"checkpoints_{experiment_name}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Results storage without lambda
        self.results = {
            'baseline_results': {},
            'fold_results': defaultdict(nested_defaultdict_list),
            'fold_f1': defaultdict(nested_defaultdict_list),
            'fold_accuracy': defaultdict(nested_defaultdict_list),
            'experiment_status': {
                'baseline_completed': [],
                'folds_completed': defaultdict(list)
            }
        }

        self.load_checkpoint()

    def save_checkpoint(self):
        """Save current experiment state"""
        checkpoint_path = os.path.join(self.checkpoint_dir, "experiment_checkpoint.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.results, f)

        # Also save as JSON for readability
        json_path = os.path.join(self.checkpoint_dir, "experiment_status.json")
        status_data = {
            'timestamp': datetime.now().isoformat(),
            'baseline_completed': self.results['experiment_status']['baseline_completed'],
            'folds_completed': dict(self.results['experiment_status']['folds_completed'])
        }
        with open(json_path, 'w') as f:
            json.dump(status_data, f, indent=2)

        print(f"💾 Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self):
        """Load previous experiment state if exists"""
        checkpoint_path = os.path.join(self.checkpoint_dir, "experiment_checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                self.results = pickle.load(f)
            print(f"🔄 Loaded checkpoint from {checkpoint_path}")
            self.print_resume_status()
        else:
            print("🆕 Starting new experiment")

    def print_resume_status(self):
        """Print what has been completed so far"""
        print("\n📊 EXPERIMENT RESUME STATUS:")
        print("="*50)

        baseline_done = self.results['experiment_status']['baseline_completed']
        if baseline_done:
            print(f"✅ Baseline completed for: {baseline_done}")
        else:
            print("❌ Baseline not completed")

        folds_done = self.results['experiment_status']['folds_completed']
        for k_fold, models in folds_done.items():
            if models:
                print(f"✅ {k_fold}-fold completed for: {models}")
            else:
                print(f"❌ {k_fold}-fold not started")
        print("="*50)

    def is_baseline_completed(self, model_name):
        return model_name in self.results['experiment_status']['baseline_completed']

    def is_fold_completed(self, k_fold, model_name, fold_num):
        completed_folds = self.results['experiment_status']['folds_completed'].get(f"{k_fold}fold_{model_name}", [])
        return fold_num in completed_folds

    def mark_baseline_completed(self, model_name, results):
        self.results['baseline_results'][model_name] = results
        if model_name not in self.results['experiment_status']['baseline_completed']:
            self.results['experiment_status']['baseline_completed'].append(model_name)
        self.save_checkpoint()

    def mark_fold_completed(self, k_fold, model_name, fold_num, auc, f1, accuracy):
        self.results['fold_results'][k_fold][model_name].append(auc)
        self.results['fold_f1'][k_fold][model_name].append(f1)
        self.results['fold_accuracy'][k_fold][model_name].append(accuracy)

        fold_key = f"{k_fold}fold_{model_name}"
        if fold_key not in self.results['experiment_status']['folds_completed']:
            self.results['experiment_status']['folds_completed'][fold_key] = []
        if fold_num not in self.results['experiment_status']['folds_completed'][fold_key]:
            self.results['experiment_status']['folds_completed'][fold_key].append(fold_num)

        self.save_checkpoint()

# Initialize safe experiment manager
safe_manager = SafeExperimentManager("legal_citation_auc_focused")

# ========================
# 🔧 CONFIG
# ========================
train_file = "/home/liorkob/M.Sc/thesis/citation-prediction/data_splits/crossencoder_train.csv"
val_file = "/home/liorkob/M.Sc/thesis/citation-prediction/data_splits/crossencoder_val.csv"
test_file = "/home/liorkob/M.Sc/thesis/citation-prediction/data_splits/crossencoder_test.csv"
batch_size = 4
max_len = 1024
epochs = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ========================
# 🆕 SAFE GPU MEMORY MANAGEMENT
# ========================
def safe_gpu_cleanup():
    """Safely clean up GPU memory"""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def safe_model_load(model_path, device):
    """Safely load model with error handling"""
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        return model
    except Exception as e:
        print(f"❌ Error loading model {model_path}: {e}")
        safe_gpu_cleanup()
        raise e

# ========================
# 🆕 WEIGHTED LOSS FUNCTION (unchanged)
# ========================
class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, pos_weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets, attention_mask=None):
        """Custom weighted cross-entropy loss for sequence-to-sequence models"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_targets = shift_labels.view(-1)
        
        mask = flat_targets != -100
        
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
        
        valid_logits = flat_logits[mask]
        valid_targets = flat_targets[mask]
        
        if self.pos_weight is not None and len(valid_targets) > 0:
            weights = torch.ones(valid_targets.size(0), device=logits.device)
            
            yes_token_ids = [259, 1903]  # כן tokens
            for yes_token in yes_token_ids:
                yes_mask = valid_targets == yes_token
                weights[yes_mask] = self.pos_weight
            
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fn(valid_logits, valid_targets)
            weighted_loss = (losses * weights).mean()
            return weighted_loss
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
            return loss_fn(valid_logits, valid_targets)

# ========================
# 🆕 ENHANCED CLASSIFICATION WITH PROBABILITY SCORES (unchanged)
# ========================
def classify_with_probabilities(model, tokenizer, input_ids, attention_mask, threshold=0.0):
    """Classify using threshold-based method with probability scores for AUC calculation"""
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
        
        yes_tokens = [259, 1903]  # כן
        no_tokens = [1124]        # לא
        
        predictions = []
        probabilities = []
        scores = []
        
        for batch_idx in range(batch_size):
            batch_logits = logits[batch_idx]
            
            yes_score = torch.mean(batch_logits[yes_tokens]).item()
            no_score = torch.mean(batch_logits[no_tokens]).item()
            
            score_diff = yes_score - no_score
            probability = torch.sigmoid(torch.tensor(score_diff)).item()
            
            if score_diff > threshold:
                prediction = 1
                predicted_text = "כן"
            else:
                prediction = 0
                predicted_text = "לא"
            
            predictions.append(prediction)
            probabilities.append(probability)
            scores.append({
                'prediction': prediction,
                'predicted_text': predicted_text,
                'score_diff': score_diff,
                'probability': probability,
                'yes_score': yes_score,
                'no_score': no_score
            })
        
        return predictions, probabilities, scores

def find_best_threshold_for_auc(model, tokenizer, dataloader, device, true_labels):
    """Find optimal threshold for maximizing AUC"""
    print("🔍 Finding best threshold for AUC optimization...")
    
    all_probabilities = []
    
    model.eval()
    print("📊 Collecting probabilities from batches...")
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            _, probabilities, _ = classify_with_probabilities(
                model, tokenizer,
                batch["input_ids"],
                batch["attention_mask"],
                threshold=0.0
            )
            
            all_probabilities.extend(probabilities)
    
    auc_score = roc_auc_score(true_labels, all_probabilities)
    print(f"Current AUC with probabilities: {auc_score:.4f}")
    
    print("🎯 Testing thresholds for optimal F1...")
    all_score_diffs = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _, _, scores = classify_with_probabilities(
            model, tokenizer,
            batch["input_ids"],
            batch["attention_mask"],
            threshold=0.0
        )
        for score in scores:
            all_score_diffs.append(score['score_diff'])
    
    thresholds = np.linspace(min(all_score_diffs), max(all_score_diffs), 50)
    best_threshold = 0.0
    best_f1 = 0.0
    
    for threshold in thresholds:
        predictions = []
        
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            batch_preds, _, _ = classify_with_probabilities(
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
    
    print(f"Best threshold: {best_threshold:.4f} (F1: {best_f1:.4f}, AUC: {auc_score:.4f})")
    return best_threshold, auc_score

# ========================
# 🧠 IMPROVED Dataset with Specific Legal Prompts (unchanged)
# ========================
class LegalSentencingCitationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        """Dataset with legally-specific prompts for sentencing citation prediction"""
        
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
        
        print(f"Legal Dataset created: {len(self.inputs)} samples")
        print(f"Label distribution: {np.bincount(self.labels)}")

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
        
        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
            "numeric_label": self.labels[idx]
        }

# ========================
# 🆕 SAFE EVALUATION FUNCTION
# ========================
def evaluate_legal_model_safely(model, dataloader, tokenizer, device, use_threshold_tuning=False, fix_threshold=None):
    """Safely evaluate the legal citation model with error handling"""
    try:
        model.eval()
        
        true_labels = []
        for batch in dataloader:
            true_labels.extend(batch["numeric_label"].numpy())
        true_labels = np.array(true_labels)
        
        all_predictions = []
        all_probabilities = []
        all_confidence_scores = []

        if fix_threshold is not None:
            print(f"🔧 Using fixed threshold: {fix_threshold}")
            best_threshold = fix_threshold
            print("📊 Evaluating with fixed threshold...")
            
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}

                    predictions, probabilities, confidence_scores = classify_with_probabilities(
                        model, tokenizer,
                        batch["input_ids"],
                        batch["attention_mask"],
                        threshold=best_threshold
                    )

                    all_predictions.extend(predictions)
                    all_probabilities.extend(probabilities)
                    all_confidence_scores.extend(confidence_scores)
                    
        elif use_threshold_tuning:
            print("⚙️ Tuning threshold for optimal performance...")
            best_threshold, auc_during_tuning = find_best_threshold_for_auc(model, tokenizer, dataloader, device, true_labels)
            print("📊 Evaluating with tuned threshold...")

            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}

                    predictions, probabilities, confidence_scores = classify_with_probabilities(
                        model, tokenizer,
                        batch["input_ids"],
                        batch["attention_mask"],
                        threshold=best_threshold
                    )

                    all_predictions.extend(predictions)
                    all_probabilities.extend(probabilities)
                    all_confidence_scores.extend(confidence_scores)
        else:
            print("🎯 Evaluating using text generation...")
            best_threshold = None
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    generated = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=5
                    )
                    decoded_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
                    predictions = [1 if p.strip() == "כן" else 0 for p in decoded_preds]

                    all_predictions.extend(predictions)
                    all_probabilities.extend([float(p) for p in predictions])
                    all_confidence_scores.extend([{} for _ in predictions])

        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)

        f1 = f1_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        accuracy = np.mean(predictions == true_labels)
        
        auc = roc_auc_score(true_labels, probabilities) if len(np.unique(true_labels)) > 1 else 0.0

        print(f"\n📊 LEGAL CITATION PREDICTION RESULTS:")
        print(f"🎯 AUC Score (PRIMARY): {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        print(f"\nPrediction Distribution: {np.bincount(predictions)}")
        print(f"True Label Distribution: {np.bincount(true_labels)}")

        return auc, {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'threshold': best_threshold,
            'scores': all_confidence_scores
        }
    
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        safe_gpu_cleanup()
        raise e

# ========================
# 🆕 SAFE NON-TRAINED MODEL EVALUATION
# ========================
def evaluate_non_trained_models_safely(model_paths, test_dataset, tokenizer_dict):
    """Safely evaluate non-trained (base) models"""
    print("\n" + "="*80)
    print("🔍 EVALUATING NON-TRAINED (BASE) MODELS")
    print("="*80)
    
    for model_path in model_paths:
        model_name = model_path.split('/')[-1]
        
        # Skip if already completed
        if safe_manager.is_baseline_completed(model_name):
            print(f"✅ Baseline for {model_name} already completed, skipping...")
            continue
        
        print(f"\n📋 Evaluating base model: {model_name}")
        
        try:
            tokenizer = tokenizer_dict[model_path]
            model = safe_model_load(model_path, device)
            
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            test_auc, test_metrics = evaluate_legal_model_safely(
                model, test_loader, tokenizer, device,
                use_threshold_tuning=True
            )
            
            # Save results immediately
            baseline_results = {
                'auc': test_auc,
                'f1': test_metrics['f1'],
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'threshold': test_metrics['threshold']
            }
            
            safe_manager.mark_baseline_completed(model_name, baseline_results)
            
            print(f"✅ {model_name} baseline results:")
            print(f"   AUC: {test_auc:.4f}")
            print(f"   F1: {test_metrics['f1']:.4f}")
            print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
        finally:
            # Always clean up
            if 'model' in locals():
                del model
            safe_gpu_cleanup()

def calculate_class_weights(dataset):
    """Calculate class weights for weighted loss"""
    labels = [dataset[i]['numeric_label'] for i in range(len(dataset))]
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    total_samples = len(labels)
    weights = {}
    for label, count in zip(unique_labels, counts):
        weights[label] = total_samples / (len(unique_labels) * count)
    
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")
    print(f"Class weights: {weights}")
    
    pos_weight = weights.get(1, 1.0) / weights.get(0, 1.0)
    print(f"Positive class weight: {pos_weight:.4f}")
    
    return pos_weight

# ========================
# 📥 MAIN EXECUTION WITH SAFE MEMORY MANAGEMENT
# ========================

# K-fold cross-validation setup
K_FOLDS_OPTIONS = [10]  # Updated to 10-fold as per your data structure
RANDOM_SEED = 42

print("Loading and combining datasets for k-fold cross-validation...")

model_paths = ["/home/liorkob/M.Sc/thesis/t5/het5-mlm-final", "imvladikon/het5-base"]

# Pre-load tokenizers
tokenizer_dict = {}
for model_path in model_paths:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer_dict[model_path] = tokenizer
    except Exception as e:
        print(f"❌ Error loading tokenizer for {model_path}: {e}")
        continue

# ========================
# 🆕 1. SAFE BASELINE EVALUATION
# ========================
df_test = pd.read_csv(test_file)
test_dataset = LegalSentencingCitationDataset(df_test, list(tokenizer_dict.values())[0], max_len=max_len)
evaluate_non_trained_models_safely(model_paths, test_dataset, tokenizer_dict)

# ========================
# 🔄 2. SAFE K-FOLD EVALUATION
# ========================
for K_FOLDS in K_FOLDS_OPTIONS:
    print(f"\n" + "="*80)
    print(f"🔄 STARTING {K_FOLDS}-FOLD CROSS-VALIDATION WITH AUC OPTIMIZATION")
    print(f"="*80)
    
    for model_idx, model_path in enumerate(model_paths):
        model_name = model_path.split('/')[-1]
        print(f"\n{'='*80}")
        print(f"🔄 STARTING {K_FOLDS}-FOLD EVALUATION FOR: {model_name}")
        print(f"{'='*80}")
        
        tokenizer = tokenizer_dict[model_path]
        
        for fold in range(1, K_FOLDS + 1):
            # Skip if already completed
            if safe_manager.is_fold_completed(K_FOLDS, model_name, fold):
                print(f"✅ Fold {fold} for {model_name} already completed, skipping...")
                continue
            
            print(f"\n📁 FOLD {fold}/{K_FOLDS}")
            
            try:
                fold_dir = f"/home/liorkob/M.Sc/thesis/citation-prediction/data_splits_10fold/fold_{fold}"

                df_train = pd.read_csv(os.path.join(fold_dir, "train.csv"))
                df_val = pd.read_csv(os.path.join(fold_dir, "val.csv"))
                df_test = pd.read_csv(os.path.join(fold_dir, "test.csv"))
                
                train_dataset = LegalSentencingCitationDataset(df_train, tokenizer, max_len=max_len)
                val_dataset = LegalSentencingCitationDataset(df_val, tokenizer, max_len=max_len)
                test_dataset = LegalSentencingCitationDataset(df_test, tokenizer, max_len=max_len)

                pos_weight = calculate_class_weights(train_dataset)
                weighted_loss_fn = WeightedCrossEntropyLoss(pos_weight=pos_weight)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)
                
                print(f"Loading fresh model for fold {fold}...")
                model = safe_model_load(model_path, device)
                optimizer = AdamW(model.parameters(), lr=2e-5)
                
                best_val_auc = 0
                best_val_threshold = 0.5
                best_model_state = None
                patience = 3
                epochs_no_improve = 0

                print("📏 Initial threshold calibration (epoch 0)")
                _, init_metrics = evaluate_legal_model_safely(
                    model, val_loader, tokenizer, device,
                    use_threshold_tuning=True
                )
                best_val_threshold = init_metrics["threshold"]
                print(f"Initial threshold set to: {best_val_threshold:.4f}")

                val_true_labels = np.array([val_loader.dataset[i]['numeric_label'] for i in range(len(val_loader.dataset))])

                # Training loop
                for epoch in range(epochs):
                    model.train()
                    total_loss = 0
                    num_batches = len(train_loader)

                    for step, batch in enumerate(train_loader):
                        try:
                            batch = {k: v.to(device) for k, v in batch.items()}

                            decoder_input_ids = torch.zeros_like(batch["labels"])
                            decoder_input_ids[:, 1:] = batch["labels"][:, :-1].clone()
                            decoder_input_ids[:, 0] = tokenizer.pad_token_id
                            decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id

                            outputs = model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                decoder_input_ids=decoder_input_ids,
                                labels=batch["labels"]
                            )

                            loss = weighted_loss_fn(outputs.logits, batch["labels"])
                            
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            optimizer.zero_grad()

                            total_loss += loss.item()
                        
                        except Exception as e:
                            print(f"❌ Error in training step: {e}")
                            safe_gpu_cleanup()
                            continue
                    
                    # Simple progress indicator
                    print(f"📈 Fold {fold}, Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/num_batches:.4f}")

                    # Validation
                    val_auc, val_metrics = evaluate_legal_model_safely(
                        model, val_loader, tokenizer, device,
                        use_threshold_tuning=False,
                        fix_threshold=best_val_threshold
                    )

                    print(f"   Val AUC: {val_auc:.4f} (Best: {best_val_auc:.4f})")

                    # Recalculate threshold only at epoch 0 and 3
                    if epoch in [0, 2]:
                        print(f"📏 Re-calculating threshold at epoch {epoch+1}")
                        best_val_threshold, _ = find_best_threshold_for_auc(model, tokenizer, val_loader, device, val_true_labels)
                        print(f"📌 New threshold: {best_val_threshold:.4f}")

                    # Save best model if AUC improved
                    if val_auc > best_val_auc or best_model_state is None:
                        best_val_auc = val_auc
                        best_model_state = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                        print(f"✅ Best model updated")
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience:
                            print(f"⏹️ Early stopping at epoch {epoch+1}")
                            break

                # Load best model
                model.load_state_dict(best_model_state)

                # Final Test Evaluation
                test_auc, test_metrics = evaluate_legal_model_safely(
                    model, test_loader, tokenizer, device,
                    use_threshold_tuning=False,
                    fix_threshold=best_val_threshold
                )

                # Save fold results immediately
                safe_manager.mark_fold_completed(K_FOLDS, model_name, fold, test_auc, test_metrics['f1'], test_metrics['accuracy'])

                print(f"✅ {K_FOLDS}-fold, Fold {fold} Results: AUC = {test_auc:.4f}, F1 = {test_metrics['f1']:.4f}, Accuracy = {test_metrics['accuracy']:.4f}")
                
            except Exception as e:
                print(f"❌ Error in fold {fold} for {model_name}: {e}")
                safe_gpu_cleanup()
                continue
            finally:
                # Always clean up
                if 'model' in locals():
                    del model
                safe_gpu_cleanup()
        
        # Print summary for this model
        if model_name in safe_manager.results['fold_results'][K_FOLDS]:
            fold_auc_scores = safe_manager.results['fold_results'][K_FOLDS][model_name]
            fold_f1_scores = safe_manager.results['fold_f1'][K_FOLDS][model_name]
            fold_accuracies = safe_manager.results['fold_accuracy'][K_FOLDS][model_name]
            
            print(f"\n📊 {model_name} - {K_FOLDS}-Fold Summary:")
            print(f"AUC Scores: {fold_auc_scores}")
            print(f"Mean AUC: {np.mean(fold_auc_scores):.4f} ± {np.std(fold_auc_scores):.4f}")
            print(f"F1 Scores: {fold_f1_scores}")
            print(f"Mean F1: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
            print(f"Accuracy: {fold_accuracies}")
            print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")

# ========================
# 📊 FINAL RESULTS PRESENTATION
# ========================
def print_final_results():
    """Print the final results in the requested format"""
    print("\n" + "="*100)
    print("🎯 FINAL RESULTS SUMMARY")
    print("="*100)
    
    # 1. Baseline Results (No MLM, No Citation Training) - AUC Only
    print("\n1️⃣ BASELINE RESULTS (No MLM, No Citation Training) - AUC Only")
    print("="*70)
    
    baseline_results = safe_manager.results['baseline_results']
    if baseline_results:
        print(f"{'Model':<25} {'AUC':<10}")
        print("-" * 35)
        for model_name, results in baseline_results.items():
            print(f"{model_name:<25} {results['auc']:<10.4f}")
    else:
        print("❌ No baseline results available")
    
    # 2. Comparison Table: Impact of Citation Prediction Task
    print("\n2️⃣ IMPACT OF MLM AND CITATION PREDICTION TRAINING")
    print("="*70)
    
    model_comparison = []
    
    # Extract model names
    model_names = list(baseline_results.keys()) if baseline_results else []
    
    if len(model_names) >= 2:
        print(f"{'Metric':<15} {'het5-mlm-final':<20} {'het5-base':<20} {'Difference':<15} {'p-value':<10} {'Cohens d':<10}")
        print("-" * 95)
        
        # Get baseline AUC
        mlm_baseline_auc = baseline_results.get('het5-mlm-final', {}).get('auc', 0)
        base_baseline_auc = baseline_results.get('het5-base', {}).get('auc', 0)
        
        print(f"{'Baseline AUC':<15} {mlm_baseline_auc:<20.4f} {base_baseline_auc:<20.4f} {mlm_baseline_auc - base_baseline_auc:<15.4f} {'N/A':<10} {'N/A':<10}")
        
        # Get trained results for 10-fold
        K_FOLDS = 10
        if ('het5-mlm-final' in safe_manager.results['fold_results'][K_FOLDS] and 
            'het5-base' in safe_manager.results['fold_results'][K_FOLDS]):
            
            mlm_auc_scores = safe_manager.results['fold_results'][K_FOLDS]['het5-mlm-final']
            base_auc_scores = safe_manager.results['fold_results'][K_FOLDS]['het5-base']
            mlm_f1_scores = safe_manager.results['fold_f1'][K_FOLDS]['het5-mlm-final']
            base_f1_scores = safe_manager.results['fold_f1'][K_FOLDS]['het5-base']
            mlm_acc_scores = safe_manager.results['fold_accuracy'][K_FOLDS]['het5-mlm-final']
            base_acc_scores = safe_manager.results['fold_accuracy'][K_FOLDS]['het5-base']
            
            if len(mlm_auc_scores) == len(base_auc_scores) and len(mlm_auc_scores) > 1:
                # Statistical tests
                auc_t_stat, auc_p_value = stats.ttest_rel(mlm_auc_scores, base_auc_scores)
                f1_t_stat, f1_p_value = stats.ttest_rel(mlm_f1_scores, base_f1_scores)
                acc_t_stat, acc_p_value = stats.ttest_rel(mlm_acc_scores, base_acc_scores)
                
                # Cohen's d
                auc_diff = np.array(mlm_auc_scores) - np.array(base_auc_scores)
                f1_diff = np.array(mlm_f1_scores) - np.array(base_f1_scores)
                acc_diff = np.array(mlm_acc_scores) - np.array(base_acc_scores)
                
                auc_cohens_d = np.mean(auc_diff) / np.std(auc_diff) if np.std(auc_diff) > 0 else 0
                f1_cohens_d = np.mean(f1_diff) / np.std(f1_diff) if np.std(f1_diff) > 0 else 0
                acc_cohens_d = np.mean(acc_diff) / np.std(acc_diff) if np.std(acc_diff) > 0 else 0
                
                # Format significance
                def format_significance(p_val):
                    if p_val < 0.001:
                        return "***"
                    elif p_val < 0.01:
                        return "**"
                    elif p_val < 0.05:
                        return "*"
                    else:
                        return "ns"
                
                mlm_auc_mean = np.mean(mlm_auc_scores)
                base_auc_mean = np.mean(base_auc_scores)
                mlm_f1_mean = np.mean(mlm_f1_scores)
                base_f1_mean = np.mean(base_f1_scores)
                mlm_acc_mean = np.mean(mlm_acc_scores)
                base_acc_mean = np.mean(base_acc_scores)
                
                print(f"{'Trained AUC':<15} {mlm_auc_mean:<20.4f} {base_auc_mean:<20.4f} {mlm_auc_mean - base_auc_mean:<15.4f} {format_significance(auc_p_value):<10} {auc_cohens_d:<10.2f}")
                print(f"{'Trained F1':<15} {mlm_f1_mean:<20.4f} {base_f1_mean:<20.4f} {mlm_f1_mean - base_f1_mean:<15.4f} {format_significance(f1_p_value):<10} {f1_cohens_d:<10.2f}")
                print(f"{'Trained Acc':<15} {mlm_acc_mean:<20.4f} {base_acc_mean:<20.4f} {mlm_acc_mean - base_acc_mean:<15.4f} {format_significance(acc_p_value):<10} {acc_cohens_d:<10.2f}")
                
                # Improvement from baseline
                mlm_auc_improvement = mlm_auc_mean - mlm_baseline_auc
                base_auc_improvement = base_auc_mean - base_baseline_auc
                
                print(f"\n{'AUC Improvement from Baseline:'}")
                print(f"{'het5-mlm-final':<25} {mlm_auc_improvement:<10.4f}")
                print(f"{'het5-base':<25} {base_auc_improvement:<10.4f}")
                
                print(f"\n📊 STATISTICAL SUMMARY:")
                print(f"• AUC difference: {mlm_auc_mean - base_auc_mean:.4f} (p={auc_p_value:.3f}, d={auc_cohens_d:.2f})")
                print(f"• F1 difference: {mlm_f1_mean - base_f1_mean:.4f} (p={f1_p_value:.3f}, d={f1_cohens_d:.2f})")
                print(f"• Accuracy difference: {mlm_acc_mean - base_acc_mean:.4f} (p={acc_p_value:.3f}, d={acc_cohens_d:.2f})")
                
        else:
            print("❌ Insufficient trained results for comparison")
    else:
        print("❌ Insufficient baseline results for comparison")
    
    print("\n📝 INTERPRETATION:")
    print("* p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant")
    print("Effect size: |d| < 0.2 (small), 0.2-0.8 (medium), > 0.8 (large)")
    
    # 3. Save final results to CSV
    print("\n💾 SAVING FINAL RESULTS...")
    
    # Create final summary DataFrame
    final_results = []
    
    # Add baseline results
    for model_name, results in baseline_results.items():
        final_results.append({
            'model': model_name,
            'condition': 'baseline',
            'auc_mean': results['auc'],
            'auc_std': 0.0,
            'f1_mean': results['f1'],
            'f1_std': 0.0,
            'accuracy_mean': results['accuracy'],
            'accuracy_std': 0.0
        })
    
    # Add trained results
    K_FOLDS = 10
    for model_name in safe_manager.results['fold_results'][K_FOLDS]:
        auc_scores = safe_manager.results['fold_results'][K_FOLDS][model_name]
        f1_scores = safe_manager.results['fold_f1'][K_FOLDS][model_name]
        acc_scores = safe_manager.results['fold_accuracy'][K_FOLDS][model_name]
        
        if auc_scores:  # Only if we have results
            final_results.append({
                'model': model_name,
                'condition': f'{K_FOLDS}fold_trained',
                'auc_mean': np.mean(auc_scores),
                'auc_std': np.std(auc_scores),
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'accuracy_mean': np.mean(acc_scores),
                'accuracy_std': np.std(acc_scores)
            })
    
    if final_results:
        final_df = pd.DataFrame(final_results)
        final_df.to_csv('final_legal_citation_results.csv', index=False)
        print("✅ Results saved to 'final_legal_citation_results.csv'")
        
        # Also save detailed comparison
        if len(model_names) >= 2:
            comparison_data = {
                'metric': ['Baseline_AUC', 'Trained_AUC', 'Trained_F1', 'Trained_Accuracy'],
                'het5_mlm_final': [
                    mlm_baseline_auc,
                    mlm_auc_mean if 'mlm_auc_mean' in locals() else 0,
                    mlm_f1_mean if 'mlm_f1_mean' in locals() else 0,
                    mlm_acc_mean if 'mlm_acc_mean' in locals() else 0
                ],
                'het5_base': [
                    base_baseline_auc,
                    base_auc_mean if 'base_auc_mean' in locals() else 0,
                    base_f1_mean if 'base_f1_mean' in locals() else 0,
                    base_acc_mean if 'base_acc_mean' in locals() else 0
                ],
                'p_value': [
                    'N/A',
                    auc_p_value if 'auc_p_value' in locals() else 'N/A',
                    f1_p_value if 'f1_p_value' in locals() else 'N/A',
                    acc_p_value if 'acc_p_value' in locals() else 'N/A'
                ],
                'cohens_d': [
                    'N/A',
                    auc_cohens_d if 'auc_cohens_d' in locals() else 'N/A',
                    f1_cohens_d if 'f1_cohens_d' in locals() else 'N/A',
                    acc_cohens_d if 'acc_cohens_d' in locals() else 'N/A'
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv('model_comparison_table.csv', index=False)
            print("✅ Comparison table saved to 'model_comparison_table.csv'")
    
    print("\n🎯 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*100)

# Execute the main experiment
print("🚀 STARTING SAFE LEGAL CITATION PREDICTION EXPERIMENT")
print("="*80)

# Print final results at the end
print_final_results()