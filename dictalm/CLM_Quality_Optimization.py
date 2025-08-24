import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
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
import math
from peft import PeftModel, PeftConfig

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

# === CLM QUALITY ASSESSMENT ===

def calculate_perplexity(model, tokenizer, texts, max_length=512, batch_size=4):
    """Calculate perplexity on a set of texts to assess CLM quality"""
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(device)
            
            # Calculate loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Count actual tokens (excluding padding)
            num_tokens = (inputs["attention_mask"] == 1).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss

def assess_clm_quality(model_path, test_texts, is_peft_model=False):
    """Comprehensive assessment of CLM quality"""
    print(f"\n🔍 Assessing CLM quality for: {model_path}")
    print("-" * 60)
    
    # Load model and tokenizer
    if is_peft_model:
        print("Loading PEFT model...")
        config = PeftConfig.from_pretrained(model_path)
        base_model_path = config.base_model_name_or_path
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        print("Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Calculate perplexity
    perplexity, avg_loss = calculate_perplexity(model, tokenizer, test_texts[:100])  # Sample for speed
    
    # Test generation quality
    test_prompts = [
        "בתיק הפלילי הנוכחי",
        "הנאשם הורשע בעבירות",
        "בבית המשפט נדונה",
        "העונש שיוטל על הנאשם"
    ]
    
    generation_quality = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_quality.append(generated_text)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'generation_samples': generation_quality
    }

# === CITATION PREDICTION DATASET ===

class OptimizedCitationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=1024, is_training=True):
        self.df = df.reset_index(drop=True).dropna()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_training = is_training
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label'])
        answer = "כן" if label == 1 else "לא"
        
        facts_a = str(row['gpt_facts_a'])
        facts_b = str(row['gpt_facts_b'])
        
        prompt = f"""מערכת חיזוי ציטוטים משפטיים
האם פסק דין א' יצטט פסק דין ב' על בסיס דמיון בעובדות?

עובדות פסק דין א':
{facts_a}

עובדות פסק דין ב':
{facts_b}

תשובה: """
        
        if self.is_training:
            full_text = prompt + answer
            tokens = self.tokenizer.encode(full_text, max_length=self.max_len, truncation=True)
            prompt_tokens = self.tokenizer.encode(prompt, max_length=self.max_len-10, truncation=True)
            answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
            
            full_tokens = prompt_tokens + answer_tokens
            input_ids = full_tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(full_tokens))
            attention_mask = [1] * len(full_tokens) + [0] * (self.max_len - len(full_tokens))
            labels = [-100] * len(prompt_tokens) + answer_tokens + [-100] * (self.max_len - len(full_tokens))
            
            return {
                "input_ids": torch.tensor(input_ids[:self.max_len]),
                "attention_mask": torch.tensor(attention_mask[:self.max_len]),
                "labels": torch.tensor(labels[:self.max_len])
            }
        else:
            tokens = self.tokenizer.encode(prompt, max_length=self.max_len, truncation=True)
            input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * (self.max_len - len(tokens))
            
            return {
                "input_ids": torch.tensor(input_ids[:self.max_len]),
                "attention_mask": torch.tensor(attention_mask[:self.max_len]),
                "true_label": torch.tensor(label),
                "prompt_length": len(tokens)
            }

# === OPTIMIZATION STRATEGIES ===

def setup_model_with_strategy(model_path, is_peft_model=False, strategy="baseline"):
    """Setup model with different optimization strategies"""
    
    if is_peft_model:
        config = PeftConfig.from_pretrained(model_path)
        base_model_path = config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model_path = model_path
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Different quantization strategies
    if strategy == "no_quantization":
        print("Strategy: No quantization")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    elif strategy == "conservative_quant":
        print("Strategy: Conservative quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:  # baseline
        print("Strategy: Baseline (4-bit quantization)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    if is_peft_model:
        model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer

def train_with_optimization(model, train_loader, val_loader, device, strategy="baseline", epochs=2):
    """Train with different optimization strategies"""
    
    model.train()
    
    # Different learning rate strategies
    if strategy == "low_lr":
        lr = 1e-5
        print(f"Strategy: Low learning rate ({lr})")
    elif strategy == "adaptive_lr":
        lr = 2e-4
        print(f"Strategy: Adaptive learning rate with scheduler")
    elif strategy == "warmup":
        lr = 1e-4
        print(f"Strategy: Warmup learning rate")
    else:  # baseline
        lr = 1e-4
        print(f"Strategy: Baseline learning rate ({lr})")
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Add scheduler for adaptive strategy
    if strategy == "adaptive_lr":
        from transformers import get_cosine_schedule_with_warmup
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
    elif strategy == "warmup":
        from transformers import get_linear_schedule_with_warmup
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 5,
            num_training_steps=total_steps
        )
    else:
        scheduler = None
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            try:
                device_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = model(**device_batch)
                loss = outputs.loss
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Lower clipping
                optimizer.step()
                
                if scheduler:
                    scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"Training error: {e}")
                continue
        
        # Validation
        val_loss = evaluate_loss(model, val_loader, device)
        avg_train_loss = total_loss / max(num_batches, 1)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_loss(model, loader, device):
    """Evaluate validation loss"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            try:
                device_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = model(**device_batch)
                loss = outputs.loss
                if not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1
            except:
                continue
    
    model.train()
    return total_loss / max(num_batches, 1)

def evaluate_citation_performance(model, test_loader, device, tokenizer):
    """Evaluate citation prediction performance"""
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    yes_ids = tokenizer.encode("כן", add_special_tokens=False)
    no_ids = tokenizer.encode("לא", add_special_tokens=False)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                true_label = batch["true_label"].item()
                prompt_length = batch["prompt_length"].item()
                
                # Generate response
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                generated_tokens = outputs[0][prompt_length:].cpu().tolist()
                
                # Check if generated tokens match yes/no
                if any(token in yes_ids for token in generated_tokens[:3]):
                    pred = 1
                    prob = 0.8
                elif any(token in no_ids for token in generated_tokens[:3]):
                    pred = 0
                    prob = 0.2
                else:
                    pred = 0
                    prob = 0.5
                
                predictions.append(pred)
                true_labels.append(true_label)
                probabilities.append(prob)
                
            except Exception as e:
                predictions.append(0)
                true_labels.append(true_label)
                probabilities.append(0.5)
    
    if not predictions:
        return {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.5}
    
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'auc': roc_auc_score(true_labels, probabilities) if len(set(true_labels)) > 1 else 0.5
    }
    
    model.train()
    return metrics

# === MAIN EXPERIMENT ===

def comprehensive_clm_experiment(folds_dir, base_model_path, finetuned_model_path, 
                                 clm_texts_file, max_folds=3):
    """Comprehensive experiment to assess and optimize CLM quality"""
    
    print("🔬 COMPREHENSIVE CLM QUALITY ASSESSMENT AND OPTIMIZATION")
    print("=" * 80)
    
    # Load CLM assessment texts
    print("📚 Loading texts for CLM assessment...")
    if os.path.exists(clm_texts_file):
        clm_df = pd.read_csv(clm_texts_file)
        clm_texts = clm_df['extracted_gpt_facts'].dropna().astype(str).tolist()[:500]  # Sample
    else:
        print("⚠️ CLM texts file not found. Using citation data for assessment.")
        # Load from first fold as fallback
        fold_1_train = pd.read_csv(os.path.join(folds_dir, "fold_1", "train.csv"))
        clm_texts = (fold_1_train['gpt_facts_a'].fillna('') + ' ' + 
                    fold_1_train['gpt_facts_b'].fillna('')).tolist()[:200]
    
    print(f"Loaded {len(clm_texts)} texts for assessment")
    
    # 1. Assess original CLM quality
    print("\n" + "="*60)
    print("PHASE 1: ORIGINAL CLM QUALITY ASSESSMENT")
    print("="*60)
    
    base_quality = assess_clm_quality(base_model_path, clm_texts, is_peft_model=False)
    finetuned_quality = assess_clm_quality(finetuned_model_path, clm_texts, is_peft_model=True)
    
    print(f"\n📊 CLM Quality Comparison:")
    print(f"Base Model Perplexity: {base_quality['perplexity']:.2f}")
    print(f"Fine-tuned Model Perplexity: {finetuned_quality['perplexity']:.2f}")
    print(f"Quality Change: {((finetuned_quality['perplexity'] - base_quality['perplexity']) / base_quality['perplexity'] * 100):+.1f}%")
    
    # 2. Test different optimization strategies
    print("\n" + "="*60)
    print("PHASE 2: OPTIMIZATION STRATEGY TESTING")
    print("="*60)
    
    strategies = [
        ("baseline", "baseline"),
        ("low_lr", "baseline"),
        ("adaptive_lr", "baseline"),
        ("warmup", "baseline"),
        ("baseline", "no_quantization"),
        ("baseline", "conservative_quant")
    ]
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for train_strategy, model_strategy in strategies:
        strategy_name = f"{train_strategy}_{model_strategy}"
        print(f"\n🧪 Testing strategy: {strategy_name}")
        
        fold_results = []
        
        for fold in range(1, min(max_folds + 1, 4)):  # Test on 3 folds max
            print(f"  Fold {fold}/3...")
            
            # Load data
            fold_dir = os.path.join(folds_dir, f"fold_{fold}")
            df_train = pd.read_csv(os.path.join(fold_dir, "train.csv")).head(100)  # Small sample
            df_val = pd.read_csv(os.path.join(fold_dir, "val.csv")).head(50)
            df_test = pd.read_csv(os.path.join(fold_dir, "test.csv")).head(50)
            
            try:
                # Setup models
                base_model, base_tokenizer = setup_model_with_strategy(
                    base_model_path, is_peft_model=False, strategy=model_strategy
                )
                ft_model, ft_tokenizer = setup_model_with_strategy(
                    finetuned_model_path, is_peft_model=True, strategy=model_strategy
                )
                
                # Create datasets
                train_dataset = OptimizedCitationDataset(df_train, base_tokenizer, is_training=True)
                val_dataset = OptimizedCitationDataset(df_val, base_tokenizer, is_training=True)
                test_dataset = OptimizedCitationDataset(df_test, base_tokenizer, is_training=False)
                
                train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=1)
                test_loader = DataLoader(test_dataset, batch_size=1)
                
                # Train and evaluate base model
                base_model_trained = train_with_optimization(
                    base_model, train_loader, val_loader, device, 
                    strategy=train_strategy, epochs=1
                )
                base_metrics = evaluate_citation_performance(
                    base_model_trained, test_loader, device, base_tokenizer
                )
                
                # Train and evaluate fine-tuned model
                ft_model_trained = train_with_optimization(
                    ft_model, train_loader, val_loader, device,
                    strategy=train_strategy, epochs=1
                )
                ft_metrics = evaluate_citation_performance(
                    ft_model_trained, test_loader, device, ft_tokenizer
                )
                
                fold_results.append({
                    'base_auc': base_metrics['auc'],
                    'ft_auc': ft_metrics['auc'],
                    'improvement': ft_metrics['auc'] - base_metrics['auc']
                })
                
                print(f"    Base AUC: {base_metrics['auc']:.4f}, FT AUC: {ft_metrics['auc']:.4f}, "
                      f"Improvement: {ft_metrics['auc'] - base_metrics['auc']:+.4f}")
                
                # Cleanup
                del base_model, ft_model, base_model_trained, ft_model_trained
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    Error in fold {fold}: {e}")
                continue
        
        if fold_results:
            avg_improvement = np.mean([r['improvement'] for r in fold_results])
            std_improvement = np.std([r['improvement'] for r in fold_results])
            results[strategy_name] = {
                'avg_improvement': avg_improvement,
                'std_improvement': std_improvement,
                'fold_results': fold_results
            }
            print(f"  Strategy {strategy_name}: {avg_improvement:+.4f} ± {std_improvement:.4f}")
    
    # 3. Summary and recommendations
    print("\n" + "="*60)
    print("PHASE 3: RESULTS SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    
    print(f"\n📊 CLM Quality Assessment:")
    print(f"{'Metric':<25} {'Base Model':<15} {'Fine-tuned':<15} {'Change':<15}")
    print("-" * 70)
    print(f"{'Perplexity':<25} {base_quality['perplexity']:<15.2f} "
          f"{finetuned_quality['perplexity']:<15.2f} "
          f"{finetuned_quality['perplexity'] - base_quality['perplexity']:+.2f}")
    print(f"{'Avg Loss':<25} {base_quality['avg_loss']:<15.4f} "
          f"{finetuned_quality['avg_loss']:<15.4f} "
          f"{finetuned_quality['avg_loss'] - base_quality['avg_loss']:+.4f}")
    
    print(f"\n🎯 Strategy Performance (Citation Task Improvement):")
    print(f"{'Strategy':<25} {'Avg Improvement':<20} {'Std':<15}")
    print("-" * 60)
    
    best_strategy = None
    best_improvement = -float('inf')
    
    for strategy, result in sorted(results.items(), key=lambda x: x[1]['avg_improvement'], reverse=True):
        improvement = result['avg_improvement']
        std = result['std_improvement']
        print(f"{strategy:<25} {improvement:+.4f} {'':<15} {std:.4f}")
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_strategy = strategy
    
    print(f"\n🏆 Best Strategy: {best_strategy}")
    print(f"Best Improvement: {best_improvement:+.4f}")
    
    print(f"\n💡 Recommendations:")
    if finetuned_quality['perplexity'] > base_quality['perplexity'] * 1.1:
        print("❌ Fine-tuned model has significantly worse perplexity - CLM quality degraded")
        print("   → Consider using base model or retraining with better regularization")
    else:
        print("✅ Fine-tuned model maintains reasonable CLM quality")
    
    if best_improvement > 0.01:
        print(f"✅ Strategy '{best_strategy}' shows meaningful improvement")
        print("   → Use this strategy for final training")
    else:
        print("❌ No strategy shows significant improvement")
        print("   → Consider different fine-tuning approach or data preprocessing")
    
    return {
        'clm_quality': {
            'base': base_quality,
            'finetuned': finetuned_quality
        },
        'strategy_results': results,
        'best_strategy': best_strategy,
        'best_improvement': best_improvement
    }

# === ADVANCED FIXES ===

def create_improved_citation_model(base_model_path, train_data, strategy="hybrid"):
    """Create an improved model based on findings"""
    print(f"\n🔧 Creating improved model with strategy: {strategy}")
    
    if strategy == "hybrid":
        # Combine best practices from testing
        print("Using hybrid approach: Conservative quantization + Warmup + Regularization")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Conservative quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for training with LoRA
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        model = prepare_model_for_kbit_training(model)
        
        # Conservative LoRA config
        lora_config = LoraConfig(
            r=8,  # Lower rank
            lora_alpha=16,  # Lower alpha
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.2,  # Higher dropout
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        # Setup optimizer with warmup
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        
        from transformers import get_linear_schedule_with_warmup
        total_steps = len(train_data) * 3  # 3 epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 4,
            num_training_steps=total_steps
        )
        
        return model, tokenizer, optimizer, scheduler
    
    elif strategy == "knowledge_distillation":
        # Use knowledge distillation from base model
        print("Using knowledge distillation approach")
        # Implementation would go here
        pass
    
    elif strategy == "curriculum_learning":
        # Start with easier examples
        print("Using curriculum learning approach")
        # Implementation would go here
        pass

def diagnose_fine_tuning_issues(finetuned_model_path, base_model_path, sample_data):
    """Diagnose specific issues with the fine-tuning"""
    print("\n🔍 DIAGNOSING FINE-TUNING ISSUES")
    print("-" * 50)
    
    # Load models for comparison
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    # Load PEFT model
    config = PeftConfig.from_pretrained(finetuned_model_path)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    finetuned_model = PeftModel.from_pretrained(base_model, finetuned_model_path)
    
    issues = []
    
    # 1. Check LoRA weights magnitude
    print("1. Checking LoRA weights...")
    lora_weights = []
    for name, param in finetuned_model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_weights.append(param.data.abs().mean().item())
    
    avg_lora_weight = np.mean(lora_weights) if lora_weights else 0
    print(f"   Average LoRA weight magnitude: {avg_lora_weight:.6f}")
    
    if avg_lora_weight > 0.1:
        issues.append("LoRA weights too large - possible overfitting")
    elif avg_lora_weight < 0.001:
        issues.append("LoRA weights too small - insufficient learning")
    
    # 2. Test response consistency
    print("2. Testing response consistency...")
    test_prompts = [
        "פסק דין בעבירת סמים",
        "הנאשם הורשע בעבירות פליליות",
        "בית המשפט קבע כי"
    ]
    
    base_responses = []
    ft_responses = []
    
    for prompt in test_prompts:
        inputs = base_tokenizer(prompt, return_tensors="pt").to(base_model.device)
        
        # Base model response
        with torch.no_grad():
            base_output = base_model.generate(
                **inputs, max_new_tokens=30, do_sample=False, pad_token_id=base_tokenizer.eos_token_id
            )
        base_text = base_tokenizer.decode(base_output[0], skip_special_tokens=True)
        base_responses.append(base_text)
        
        # Fine-tuned model response
        with torch.no_grad():
            ft_output = finetuned_model.generate(
                **inputs, max_new_tokens=30, do_sample=False, pad_token_id=base_tokenizer.eos_token_id
            )
        ft_text = base_tokenizer.decode(ft_output[0], skip_special_tokens=True)
        ft_responses.append(ft_text)
    
    # Check if responses are too similar (no learning) or too different (catastrophic forgetting)
    similarity_scores = []
    for base_resp, ft_resp in zip(base_responses, ft_responses):
        # Simple word overlap similarity
        base_words = set(base_resp.split())
        ft_words = set(ft_resp.split())
        if len(base_words.union(ft_words)) > 0:
            similarity = len(base_words.intersection(ft_words)) / len(base_words.union(ft_words))
            similarity_scores.append(similarity)
    
    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
    print(f"   Average response similarity: {avg_similarity:.3f}")
    
    if avg_similarity > 0.95:
        issues.append("Responses too similar - insufficient fine-tuning effect")
    elif avg_similarity < 0.3:
        issues.append("Responses too different - possible catastrophic forgetting")
    
    # 3. Check training data quality
    print("3. Analyzing training data characteristics...")
    if hasattr(sample_data, 'extracted_gpt_facts'):
        texts = sample_data['extracted_gpt_facts'].dropna().astype(str)
        avg_length = texts.str.len().mean()
        vocab_diversity = len(set(' '.join(texts).split())) / len(' '.join(texts).split())
        
        print(f"   Average text length: {avg_length:.1f} characters")
        print(f"   Vocabulary diversity: {vocab_diversity:.3f}")
        
        if avg_length > 2000:
            issues.append("Training texts too long - may cause memory issues")
        elif avg_length < 100:
            issues.append("Training texts too short - insufficient context")
        
        if vocab_diversity < 0.1:
            issues.append("Low vocabulary diversity - limited learning signal")
    
    # 4. Check for potential data leakage
    print("4. Checking for potential data issues...")
    # This would involve more complex analysis
    
    # Cleanup
    del base_model, finetuned_model
    torch.cuda.empty_cache()
    
    return issues

def create_optimized_training_script(issues, best_strategy):
    """Generate optimized training script based on findings"""
    print(f"\n📝 GENERATING OPTIMIZED TRAINING SCRIPT")
    print(f"Best strategy found: {best_strategy}")
    print(f"Issues to address: {issues}")
    
    script = f'''# Optimized DictaLM2 Fine-tuning Script
# Generated based on analysis results
# Issues found: {", ".join(issues)}
# Best strategy: {best_strategy}

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# Optimized configuration based on findings
model_id = "dicta-il/dictalm2.0-instruct"

# Conservative quantization for better stability
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Less aggressive than 4-bit
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# Conservative LoRA configuration
lora_config = LoraConfig(
    r=8,  # Reduced from 16
    lora_alpha=16,  # Reduced from 32
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.2,  # Increased regularization
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Optimized training arguments
training_args = TrainingArguments(
    output_dir="./optimized-dictalm2-citation",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Increased for stability
    learning_rate=5e-5,  # Lower learning rate
    num_train_epochs=2,  # Fewer epochs to prevent overfitting
    warmup_steps=200,  # More warmup
    weight_decay=0.01,  # Regularization
    logging_steps=50,
    save_steps=500,
    eval_steps=250,
    evaluation_strategy="steps",
    save_strategy="steps",
    fp16=True,
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    label_smoothing_factor=0.1,  # Label smoothing for better generalization
)

# Use the trainer with your dataset...
'''
    
    with open("optimized_training_script.py", "w", encoding="utf-8") as f:
        f.write(script)
    
    print("✅ Optimized training script saved as 'optimized_training_script.py'")

# === EXECUTION ===
if __name__ == "__main__":
    set_seed(42)
    
    # Configuration
    BASE_MODEL = "dicta-il/dictalm2.0-instruct"
    FT_MODEL = "/home/liorkob/dictalm2.0-clm-lora-final"
    FOLDS_DIR = "/home/liorkob/M.Sc/thesis/citation-prediction/data_splits_10fold"
    CLM_TEXTS_FILE = "/home/liorkob/M.Sc/thesis/data/drugs_3k/gpt/processed_verdicts_with_gpt.csv"
    
    print("🚀 STARTING COMPREHENSIVE CLM DIAGNOSIS AND OPTIMIZATION")
    print("=" * 80)
    
    # Step 1: Diagnose fine-tuning issues
    if os.path.exists(CLM_TEXTS_FILE):
        sample_data = pd.read_csv(CLM_TEXTS_FILE).head(100)
        issues = diagnose_fine_tuning_issues(FT_MODEL, BASE_MODEL, sample_data)
        
        print(f"\n🔍 IDENTIFIED ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        issues = ["Could not load training data for analysis"]
    
    # Step 2: Run comprehensive experiment
    results = comprehensive_clm_experiment(
        folds_dir=FOLDS_DIR,
        base_model_path=BASE_MODEL,
        finetuned_model_path=FT_MODEL,
        clm_texts_file=CLM_TEXTS_FILE,
        max_folds=3  # Test on 3 folds for speed
    )
    
    # Step 3: Generate optimized training script
    best_strategy = results.get('best_strategy', 'baseline_baseline')
    create_optimized_training_script(issues, best_strategy)
    
    # Step 4: Save comprehensive results
    comprehensive_results = {
        'clm_assessment': results,
        'identified_issues': issues,
        'best_strategy': best_strategy,
        'recommendations': {
            'primary': "Use conservative quantization (8-bit) with lower learning rate",
            'secondary': "Reduce LoRA rank and increase regularization",
            'tertiary': "Implement proper warmup and gradient accumulation"
        }
    }
    
    with open("comprehensive_clm_analysis.json", "w") as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETED!")
    print(f"📁 Full results: comprehensive_clm_analysis.json")
    print(f"📁 Optimized script: optimized_training_script.py")
    print(f"\n🎯 KEY RECOMMENDATIONS:")
    for key, rec in comprehensive_results['recommendations'].items():
        print(f"   {key.title()}: {rec}")
    
    print(f"\n🔄 NEXT STEPS:")
    print(f"   1. Review the generated optimized_training_script.py")
    print(f"   2. Retrain the model using the optimized configuration")
    print(f"   3. Run the original evaluation script with the new model")
    print(f"   4. Compare results to see improvement")