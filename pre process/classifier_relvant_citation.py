import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import classification_report
import os
from tqdm import tqdm
from collections import Counter

# Enable MPS for GPU usage on Mac
os.environ["TF_ENABLE_MPS"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


df = pd.read_csv("/home/liorkob/thesis/lcp/global_citation_paragraphs - global_citation_paragraphs (2).csv")  # Replace with your CSV file path

def clean_texts(texts):
    return [str(text).strip().replace("\t", " ").replace("\n", " ") for text in texts]

# Clean data: Remove rows with missing or empty text
df = df[df['text'].notnull()]  # Remove null values
df = df[df['text'] != '']      # Remove empty strings
df = df[df['label'].notnull()]  # Remove rows with NaN in 'label'
df['label'] = df['label'].astype(int)  # Ensure labels are integers

# Extract texts and labels
texts = df['text'].tolist()
labels = df['label'].tolist()

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
train_texts=clean_texts(train_texts)
test_texts=clean_texts(test_texts)

# # Check class balance
# train_class_counts = Counter(train_labels)
# test_class_counts = Counter(test_labels)

# print("Train Class Distribution:", train_class_counts)
# print("Test Class Distribution:", test_class_counts)

# # Initialize tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# # Tokenize texts
# train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
# test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

# Define TextDataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure labels are integers
        return item


# # Prepare datasets
# train_dataset = TextDataset(train_encodings, train_labels)
# test_dataset = TextDataset(test_encodings, test_labels)

# # Create dataloaders
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16)

# # Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
# model.to(device)

# # Define optimizer and loss function
# optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
# loss_fn = torch.nn.CrossEntropyLoss()

# # Train the model
# for epoch in range(5):  # 5 epochs
#     model.train()
#     loop = tqdm(train_loader, leave=True)  # Initialize tqdm progress bar
#     for batch in loop:
#         inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
#         labels = batch['labels'].to(device)

#         optimizer.zero_grad()
#         outputs = model(**inputs)
#         loss = loss_fn(outputs.logits, labels)
#         loss.backward()
#         optimizer.step()

#         # Update tqdm bar with loss
#         loop.set_description(f"Epoch {epoch}")
#         loop.set_postfix(loss=loss.item())

# # Evaluate the model and store predictions
# model.eval()
# examples = []
# all_preds, all_labels = [], []
# with torch.no_grad():
#     for i, batch in enumerate(test_loader):
#         inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
#         labels = batch['labels'].to(device)

#         outputs = model(**inputs)
#         preds = torch.argmax(outputs.logits, dim=-1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

#         # Save example predictions
#         for text, true_label, pred_label in zip(test_texts[i * 16:(i + 1) * 16], labels.cpu().numpy(), preds.cpu().numpy()):
#             examples.append((text, true_label, pred_label))

# # Display some examples
# for text, true_label, pred_label in examples[:15]:  # Show first 5 examples
#     print(f"Text: {text}\nTrue Label: {true_label}, Predicted Label: {pred_label}\n")
    
# # Generate the classification report
# report = classification_report(all_labels, all_preds, output_dict=True)

# # Convert to list format
# results_list = []
# for label, metrics in report.items():
#     if isinstance(metrics, dict):  # Skip 'accuracy' as it is a scalar
#         results_list.append(
#             {
#                 "label": label,
#                 "precision": metrics["precision"],
#                 "recall": metrics["recall"],
#                 "f1-score": metrics["f1-score"],
#                 "support": metrics["support"]
#             }
#         )
# print(results_list)







# ________________________________
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def train_and_save_model(train_texts, train_labels, save_path="best_model.pt"):
    tokenizer = BertTokenizer.from_pretrained('avichr/heBERT')
    model = BertForSequenceClassification.from_pretrained('avichr/heBERT', num_labels=2)

    train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    train_dataset = TextDataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(5):
        model.train()
        for batch in train_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

train_and_save_model(train_texts, train_labels, save_path="best_model.pt")

# # ________________________________________grid_search_______________________

# param_grid = {
#     "model_name": [
#         "bert-base-multilingual-cased", 
#         "dicta-il/dictabert", 
#         "avichr/heBERT",
#         "onlplab/alephbert-base"
#     ],
#     "learning_rate": [2e-5, 3e-5],
#     "batch_size": [16, 32],
#     "max_length": [128, 256],
#     "epochs": [3, 5],
    
# }

# from sklearn.model_selection import ParameterGrid
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from torch.utils.data import DataLoader

# # Parameter grid
# grid = list(ParameterGrid(param_grid))

# best_model = None
# best_f1 = 0
# best_params = None

# for params in grid:
#     print(f"Testing parameters: {params}")
    
#     # Load model and tokenizer dynamically
#     tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
#     model = AutoModelForSequenceClassification.from_pretrained(params["model_name"], num_labels=2)
#     model.to(device)

#     # Tokenize data
#     train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=params["max_length"], return_tensors="pt")
#     test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=params["max_length"], return_tensors="pt")

#     # Prepare dataset and dataloader
#     train_dataset = TextDataset(train_encodings, train_labels)
#     test_dataset = TextDataset(test_encodings, test_labels)
#     train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=params["batch_size"])

#     # Optimizer and loss
#     optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
#     loss_fn = torch.nn.CrossEntropyLoss()

#     # Train model
#     for epoch in range(params["epochs"]):
#         model.train()
#         for batch in train_loader:
#             inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
#             labels = batch['labels'].to(device)

#             optimizer.zero_grad()
#             outputs = model(**inputs)
#             loss = loss_fn(outputs.logits, labels)
#             loss.backward()
#             optimizer.step()

#     # Evaluate model
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for batch in test_loader:
#             inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
#             labels = batch['labels'].to(device)

#             outputs = model(**inputs)
#             preds = torch.argmax(outputs.logits, dim=-1)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     # Calculate F1 score
#     report = classification_report(all_labels, all_preds, output_dict=True)
#     f1 = report["weighted avg"]["f1-score"]

#     # Update best model
#     if f1 > best_f1:
#         best_f1 = f1
#         best_model = model
#         best_params = params

# print("Best F1-Score:", best_f1)
# print("Best Parameters:", best_params)

