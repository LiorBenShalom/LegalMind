# import pandas as pd
# import torch
# import random
# import numpy as np
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM, # Changed for CLM
#     Trainer,
#     TrainingArguments,
#     DataCollatorForLanguageModeling # Changed for CLM
# )
# from datasets import Dataset
# from typing import List
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training # Added for LoRA
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

#         windows = split_to_windows(text, tokenizer, max_length=max_length, stride=max_length // 2) # Use max_length for stride or similar
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
#     # The DataCollatorForLanguageModeling will handle creating input_ids and labels
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

#     # Initialize model and tokenizer for Gemma 3 12B
#     model_id = "google/gemma-3-12b-it" 
#     print(f"Loading model and tokenizer: {model_id}")

#     tokenizer = AutoTokenizer.from_pretrained(model_id)

#     # Add a padding token if the tokenizer doesn't have one (common for CLM models like Gemma)
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#         # You might also want to increase the model's embedding size if you add tokens
#         # model.resize_token_embeddings(len(tokenizer)) # Do this AFTER loading the model in 4-bit

#     # Load model in 4-bit for memory efficiency
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=torch.bfloat16, # Recommended for Gemma, check your GPU support
#         load_in_4bit=True, # Enable 4-bit quantization
#         device_map="auto" # Distribute model across available GPUs
#     )

#     # Resize token embeddings if padding token was added
#     if tokenizer.pad_token_id is not None:
#         current_vocab_size = model.get_input_embeddings().num_embeddings
#         if len(tokenizer) > current_vocab_size:
#             model.resize_token_embeddings(len(tokenizer))

#     # Prepare model for k-bit training (important for LoRA with quantization)
#     model = prepare_model_for_kbit_training(model)

#     # Configure LoRA
#     # Target modules are typically 'q_proj', 'v_proj' for attention layers
#     # For Gemma, you might also consider 'o_proj' or 'gate_proj' for better performance
#     lora_config = LoraConfig(
#         r=8, # LoRA attention dimension
#         lora_alpha=16, # Alpha parameter for LoRA scaling
#         target_modules=["q_proj", "v_proj"], # Layers to apply LoRA to
#         lora_dropout=0.05, # Dropout probability for LoRA layers
#         bias="none", # Do not train bias terms
#         task_type="CAUSAL_LM", # Specify task type
#     )

#     # Apply LoRA to the model
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters() # See how many parameters are now trainable

#     # Prepare dataset with tokenization for CLM
#     # Max length for Gemma 3 12B is 128K, but for fine-tuning smaller chunks are common
#     # You might want to adjust max_length based on your GPU memory and data characteristics
#     max_sequence_length = 1024 # A reasonable chunk size for fine-tuning, adjust as needed
#     dataset = prepare_dataset(df, tokenizer, text_column='extracted_gpt_facts', max_length=max_sequence_length)

#     # Split dataset (80% train, 20% validation)
#     train_size = int(0.8 * len(dataset))
#     eval_size = len(dataset) - train_size

#     # Ensure shuffle is False for consistent split with select
#     train_dataset = dataset.select(range(train_size))
#     eval_dataset = dataset.select(range(train_size, train_size + eval_size))

#     print(f"Train dataset size: {len(train_dataset)}")
#     print(f"Eval dataset size: {len(eval_dataset)}")

#     # Data collator for Causal Language Modeling
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False # Set to False for CLM
#     )

#     # Training arguments
#     # Adjust per_device_train_batch_size and gradient_accumulation_steps based on your GPU memory
#     training_args = TrainingArguments(
#         output_dir="./gemma-clm-lora-trained",
#         remove_unused_columns=False,
#         per_device_train_batch_size=1, # Start small, increase if possible
#         gradient_accumulation_steps=8, # Accumulate gradients over 8 steps
#         fp16=True, # Use fp16 for faster training and less memory (if GPU supports it)
#         eval_strategy="steps",
#         eval_steps=500,
#         per_device_eval_batch_size=1,
#         gradient_checkpointing=True, # Saves memory, but slows down training
#         num_train_epochs=3, # Start with a few epochs, adjust based on convergence
#         logging_dir="./logs",
#         logging_steps=100,
#         disable_tqdm=False,
#         log_level='info',
#         save_strategy="no", # Save checkpoints periodically
#         save_steps=500,
#         learning_rate=2e-4, # Common learning rate for LoRA
#         report_to="tensorboard", # For visualization
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
#     final_output_dir = "./gemma-clm-lora-final"
#     print(f"Saving final LoRA adapters to {final_output_dir}")
#     trainer.model.save_pretrained(final_output_dir)
#     tokenizer.save_pretrained(final_output_dir)

#     print("Training completed! LoRA adapters saved.")

#     # To load the LoRA fine-tuned model for inference:
#     # from peft import PeftModel, PeftConfig
#     # config = PeftConfig.from_pretrained(final_output_dir)
#     # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16, load_in_4bit=True)
#     # model = PeftModel.from_pretrained(model, final_output_dir)
#     # model.eval() # Set model to evaluation mode

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
    model_id = "google/gemma-3-12b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # <- keep bf16 compute
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,             # <- bf16 base
        device_map="auto",
        trust_remote_code=True,
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
    model.to(dtype=torch.bfloat16)
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
        output_dir="./gemma-clm",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=8e-5,
        num_train_epochs=5,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=1000,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        fp16=False,         # <- off
        bf16=True,          # <- on (your RTX 6000 Ada supports bf16)

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
    output_dir = "./gemma-clm-final"
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
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
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