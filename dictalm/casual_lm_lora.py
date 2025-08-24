# import pandas as pd
# import torch
# import random
# import numpy as np
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForLanguageModeling,
#     BitsAndBytesConfig
# )
# from datasets import Dataset
# from typing import List
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# import os

# def split_to_windows(text: str, tokenizer, max_length: int = 256, stride: int = 128) -> List[str]:
#     """
#     Split text into overlapping windows based on token count
#     """
#     tokens = tokenizer.tokenize(text)

#     if len(tokens) <= max_length:
#         return [text]

#     windows = []
#     start = 0

#     while start < len(tokens):
#         end = min(start + max_length, len(tokens))
#         window_tokens = tokens[start:end]
#         window_text = tokenizer.convert_tokens_to_string(window_tokens)
#         windows.append(window_text)
#         start += stride
#         if end >= len(tokens):
#             break
#     return windows


# def prepare_dataset(df: pd.DataFrame, tokenizer, text_column: str = 'extracted_gpt_facts', max_length: int = 256) -> Dataset:
#     """
#     Prepare dataset from DataFrame with windowing and tokenization for CLM
#     """
#     all_tokenized_inputs = []

#     print(f"Processing {len(df)} texts...")

#     for idx, row in df.iterrows():
#         text = str(row[text_column])

#         if not text or text.strip() == '':
#             continue

#         windows = split_to_windows(text, tokenizer, max_length=max_length, stride=max_length // 2)
#         for window in windows:
#             tokenized_window = tokenizer(
#                 window,
#                 truncation=True,
#                 max_length=max_length
#             )
#             all_tokenized_inputs.append(tokenized_window)

#         if (idx + 1) % 100 == 0:
#             print(f"Processed {idx + 1} texts, generated {len(all_tokenized_inputs)} tokenized windows")

#     print(f"Total tokenized windows generated: {len(all_tokenized_inputs)}")

#     # Create dataset from tokenized inputs
#     dataset = Dataset.from_list(all_tokenized_inputs)
#     return dataset


# def main():
#     # Set random seeds for reproducibility
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)

#     # Load your data
#     print("Loading data...")
#     df = pd.read_csv("/home/liorkob/M.Sc/thesis/data/drugs_3k/gpt/processed_verdicts_with_gpt.csv")
#     print(f"Loaded {len(df)} rows")

#     # Initialize model and tokenizer for DictaLM 2.0 Instruct
#     model_id = "dicta-il/dictalm2.0-instruct"
#     print(f"Loading model and tokenizer: {model_id}")

#     tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

#     # DictaLM2 should have a pad token, but check anyway
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         print("Set pad_token to eos_token")

#     # Configure quantization
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         llm_int8_enable_fp32_cpu_offload=True  # Allow CPU offload for insufficient GPU memory
#     )

#     # Load model with proper quantization config
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         quantization_config=bnb_config,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#         max_memory={0: "10GiB", "cpu": "30GiB"}  # Adjust based on your GPU memory
#     )

#     # Prepare model for k-bit training
#     model = prepare_model_for_kbit_training(model)

#     # Configure LoRA for DictaLM2
#     # DictaLM2 is based on Mistral architecture, so we target similar modules
#     lora_config = LoraConfig(
#         r=16,  # Slightly higher rank for better performance
#         lora_alpha=32,
#         target_modules=[
#             "q_proj", 
#             "k_proj", 
#             "v_proj", 
#             "o_proj",
#             "gate_proj", 
#             "up_proj", 
#             "down_proj"
#         ],  # Common modules for Mistral-based models
#         lora_dropout=0.1,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )

#     # Apply LoRA to the model
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()

#     # Prepare dataset with tokenization for CLM
#     # DictaLM2 supports up to 8K context length, but use smaller chunks for fine-tuning
#     max_sequence_length = 1024
#     dataset = prepare_dataset(df, tokenizer, text_column='extracted_gpt_facts', max_length=max_sequence_length)

#     # Split dataset (80% train, 20% validation)
#     train_size = int(0.8 * len(dataset))
#     eval_size = len(dataset) - train_size

#     train_dataset = dataset.select(range(train_size))
#     eval_dataset = dataset.select(range(train_size, train_size + eval_size))

#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Eval dataset size: {len(eval_dataset)}")

#     # Data collator for Causal Language Modeling
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False
#     )

#     # Training arguments - adjusted for DictaLM2
#     training_args = TrainingArguments(
#         output_dir="./dictalm2.0-clm-lora-trained",
#         remove_unused_columns=False,
#         per_device_train_batch_size=2,  # Can potentially use larger batch size than Gemma 12B
#         gradient_accumulation_steps=4,
#         fp16=True,
#         eval_strategy="steps",
#         eval_steps=500,
#         per_device_eval_batch_size=2,
#         gradient_checkpointing=True,
#         num_train_epochs=3,
#         logging_dir="./logs",
#         logging_steps=100,
#         disable_tqdm=False,
#         log_level='info',
#         save_strategy="steps",
#         save_steps=500,
#         learning_rate=2e-4,
#         warmup_steps=100,  # Added warmup for better training stability
#         weight_decay=0.01,  # Added weight decay for regularization
#         report_to="tensorboard",
#         dataloader_pin_memory=False,  # May help with memory issues
#     )

#     # Initialize trainer
#     print("Initializing trainer...")
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#     )

#     # Train the model
#     print("Starting training...")
#     trainer.train()

#     # Save the final model (LoRA adapters)
#     final_output_dir = "./dictalm2.0-clm-lora-final"
#     print(f"Saving final LoRA adapters to {final_output_dir}")
#     trainer.model.save_pretrained(final_output_dir)
#     tokenizer.save_pretrained(final_output_dir)

#     print("Training completed! LoRA adapters saved.")

#     # To load the LoRA fine-tuned model for inference:
#     # from peft import PeftModel, PeftConfig
#     # config = PeftConfig.from_pretrained(final_output_dir)
#     # bnb_config = BitsAndBytesConfig(
#     #     load_in_4bit=True,
#     #     bnb_4bit_use_double_quant=True,
#     #     bnb_4bit_quant_type="nf4",
#     #     bnb_4bit_compute_dtype=torch.bfloat16,
#     #     llm_int8_enable_fp32_cpu_offload=True
#     # )
#     # model = AutoModelForCausalLM.from_pretrained(
#     #     config.base_model_name_or_path, 
#     #     quantization_config=bnb_config,
#     #     torch_dtype=torch.bfloat16, 
#     #     trust_remote_code=True,
#     #     device_map="auto"
#     # )
#     # model = PeftModel.from_pretrained(model, final_output_dir)
#     # model.eval()

# if __name__ == "__main__":
#     main()
import pandas as pd
import torch
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import math
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_perplexity(model, tokenizer, texts):
    """Calculate perplexity"""
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts[:50], desc="Calculating perplexity"):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                if not torch.isnan(loss):
                    num_tokens = inputs["attention_mask"].sum().item()
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
            except:
                continue
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

def load_train_data():
    """Load training data from gpt_facts.txt"""
    train_file = "/home/liorkob/M.Sc/thesis/pre-train/data_splits/gpt_facts.txt"
    
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(texts)} texts from gpt_facts.txt")
        return texts
    except:
        print("❌ Could not load gpt_facts.txt")
        return []

def load_test_data():
    """Load test data from test.txt"""
    test_file = "/home/liorkob/M.Sc/thesis/pre-train/data_splits/test.txt"
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(texts)} texts from test.txt")
        return texts
    except:
        print("❌ Could not load test.txt")
        return []

def setup_model():
    """Setup model for training"""
    model_id = "dicta-il/dictalm2.0-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_extensive_clm():
    """Main training function"""
    set_seed(42)
    
    print("🔥 EXTENSIVE CLM TRAINING")
    print("=" * 40)
    
    # Load data
    train_texts = load_train_data()
    test_texts = load_test_data()
    
    if not train_texts:
        print("❌ No training data available")
        return
    
    print(f"📊 Training on {len(train_texts)} texts")
    
    # Setup model
    model, tokenizer = setup_model()
    
    # Create dataset
    print("🔄 Creating dataset...")
    dataset = Dataset.from_dict({"text": train_texts})
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_special_tokens_mask=True,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Split 95/5
    train_size = int(0.95 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    print(f"📊 Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Calculate baseline perplexity
    if test_texts:
        print("📏 Calculating baseline perplexity...")
        baseline_perplexity = calculate_perplexity(model, tokenizer, test_texts)
        print(f"Baseline perplexity: {baseline_perplexity:.2f}")
    else:
        baseline_perplexity = None
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./extensive-clm",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=8e-5,
        num_train_epochs=5,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=1000,
        fp16=True,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    # Train
    print("🚀 Starting training...")
    trainer.train()
    
    # Calculate final perplexity
    if test_texts:
        print("📏 Calculating final perplexity...")
        final_perplexity = calculate_perplexity(model, tokenizer, test_texts)
        print(f"Final perplexity: {final_perplexity:.2f}")
        
        if baseline_perplexity:
            improvement = baseline_perplexity - final_perplexity
            print(f"Improvement: {improvement:.2f}")
    
    # Save model
    output_dir = "./extensive-clm-final"
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model saved to {output_dir}")
    
    # Test generation on real test data
    if test_texts:
        print("\n🧪 Testing generation on REAL data:")
        print("-" * 50)
        
        model.eval()
        for i, test_text in enumerate(test_texts[:5]):
            words = test_text.split()
            if len(words) < 10:
                continue
                
            prompt = " ".join(words[:6])
            expected = " ".join(words[6:12])
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = generated[len(prompt):].strip()
            
            print(f"{i+1}. PROMPT: {prompt}")
            print(f"   EXPECTED: {expected}")
            print(f"   GENERATED: {continuation}")
            print()
    
    print("🎉 Training completed!")

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train_extensive_clm()