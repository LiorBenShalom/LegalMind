import torch
import pandas as pd
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score
)
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import docx
import numpy as np
from pathlib import Path
import docx
import re
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score
)
from pathlib import Path
import docx
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score
)
from transformers import AutoTokenizer, AutoModelForPreTraining

class HebrewLegalCitationPredictor:
    def __init__(self):
        print("Initializing predictor...")
        self.mapping_file='/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/code/lcp/fe - MAPPING.csv'

         #  self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        #   self.model = AutoModelForPreTraining.from_pretrained("nlpaueb/legal-bert-base-uncased")

        self.tokenizer = AutoTokenizer.from_pretrained('onlplab/alephbert-base')
        self.model = AutoModel.from_pretrained('onlplab/alephbert-base')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.documents = {}
        self.document_embeddings = {}

    def _get_document_embedding(self, text):
        """Generate embeddings for a given text."""
        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # print(f"Generated embedding of shape {outputs.last_hidden_state[:, 0, :].shape} for text")
        return outputs.last_hidden_state[:, 0, :].cpu()
    
    def _extract_citations(self, file_path):
        """Extract citations from text using regex patterns"""
        data = {
            'extracted_paragraphs': [],
            'kept_paragraphs': [],
            'removed_paragraphs': [],
            'citations': []
        }
        
        cited_cases = []  # Initialize here
        cleaned_text = ""  # Initialize here
        
        try:
            # Load mapping file
            df = pd.read_csv(self.mapping_file)
            df = df.drop_duplicates(subset=['Mapping Value', 'Target Number'])

            # Dropping unnecessary columns and cleaning the data
            df = df[['Mapping Value', 'Target Number']].dropna()

            # Creating the mapping_dict: Keys = Mapping Value, Values = Target Number
            mapping_dict = dict(zip(df['Mapping Value'], df['Target Number']))

            # Read document
            doc = docx.Document(file_path)
            doc_name = file_path.stem  # No extension
            doc_name_with_doc = doc_name + '.doc'  # With `.doc`
            doc_name_with_docx = file_path.name  # Full name with `.docx`

            # Check in the mapping_dict
            doc_id = (
                mapping_dict.get(doc_name) or
                mapping_dict.get(doc_name_with_doc) or
                mapping_dict.get(doc_name_with_docx) or
                doc_name  # Default to the original name if no match is found
            )
            if doc_id == doc_name:
                print("doc_name failed to map:", doc_name)

            elif not doc_id.startswith('ת"פ '):
                doc_id = f'ת"פ {doc_id}'

            # Continue with extracting citations
            all_citations = []
            full_text = []

            for i, paragraph in enumerate(doc.paragraphs):
                para_text = paragraph.text
                
                # Check for citations
                if 'ע"פ' in para_text or 'ת"פ' in para_text or 'עפ"ג' in para_text:
                    if len(para_text.strip()) > 0:
                        # Extract case citations as simple strings
                        para_citations = []  # Renamed from cited_cases to avoid confusion
                        
                        # Extract criminal appeals (ע"פ)
                        if 'ע"פ' in para_text:
                            matches = re.finditer(r'ע"פ (\d+/\d+)', para_text)
                            para_citations.extend(f'ע"פ {match.group(1)}' for match in matches)
                        
                        # Extract criminal cases (ת"פ)
                        if 'ת"פ' in para_text:
                            matches = re.finditer(r'ת"פ (\d+[-/]\d+[-/]\d+)', para_text)
                            para_citations.extend(f'ת"פ {match.group(1)}' for match in matches)
                        
                        # Extract עפ"ג cases
                        if 'עפ"ג' in para_text:
                            matches = re.finditer(r'עפ"ג (\d+/\d+)', para_text)
                            para_citations.extend(f'עפ"ג {match.group(1)}' for match in matches)
                        
                        # Store paragraph with simple citation list
                        data['extracted_paragraphs'].append({
                            'document': doc_id,
                            'paragraph_number': i,
                            'text': para_text,
                            'cited_cases': para_citations
                        })
                        
                        # Add to overall citations
                        all_citations.extend(para_citations)
                        cited_cases.extend(para_citations)  # Update the return value
                            
                else:
                    # Clean and store non-citation paragraphs
                    cleaned_text = self.clean_text(para_text)
                    if cleaned_text.strip():
                        data['kept_paragraphs'].append({
                            'document': doc_id,
                            'paragraph_number': i,
                            'text': cleaned_text
                        })
                        full_text.append(cleaned_text)

            cleaned_text = '\n'.join(full_text)  # Update the return value
            
            # Store document with simplified citation list
            self.documents[doc_id] = {
                'text': cleaned_text,
                'citations': list(set(all_citations)),  # Deduplicated list of citations
                'extracted_data': data
            }
        except Exception as e:
            print(f"Error processing document: {str(e)}")
        
        return list(set(cited_cases)), cleaned_text, doc_id
    def create_pairs(self):
        pairs = []
        labels = []

        if not self.documents:
            print("No documents found. Ensure documents are loaded correctly.")
            return pairs, labels

        positive_pairs = []
        negative_pairs = []

        all_docs = list(self.documents.keys())
        for doc_id, doc_data in self.documents.items():
            print(f"Document ID: {doc_id}, Citations: {doc_data['citations']}")
            for citation in doc_data['citations']:
                if citation in self.documents:
                    print(f"Match Found: {doc_id} -> {citation}")
                # else:
                    # print(f"No Match for Citation: {citation}")

        for doc_id, doc_data in self.documents.items():
            # Positive pairs
            for citation in doc_data["citations"]:
                if citation in self.documents:  # Ensure the citation exists in the dataset
                    positive_pairs.append((doc_id, citation))
                    labels.append(1)

            # Negative pairs
            negative_samples = [
                doc for doc in all_docs if doc not in doc_data["citations"]
            ]
            negative_samples = np.random.choice(
                negative_samples, size=min(len(doc_data["citations"]), len(negative_samples)), replace=False
            )
            for neg_doc in negative_samples:
                negative_pairs.append((doc_id, neg_doc))
                labels.append(0)

        # Debugging
        print(f"Positive pairs: {len(positive_pairs)}, Negative pairs: {len(negative_pairs)}")

        # Combine and balance
        pairs = positive_pairs + negative_pairs
        labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)

        return pairs, labels
    def generate_features(self, pairs, labels):
        """Generate feature vectors for document pairs."""
        features = []
        valid_labels = []  # Ensure labels are consistent with features
        for idx, (doc1, doc2) in enumerate(pairs):
            if doc1 in self.document_embeddings and doc2 in self.document_embeddings:
                emb1 = self.document_embeddings[doc1]
                emb2 = self.document_embeddings[doc2]
                combined = torch.cat((emb1, emb2, torch.abs(emb1 - emb2)), dim=-1)
                features.append(combined.numpy())
                valid_labels.append(labels[idx])  # Ensure valid labels match
        print(f"Generated features for {len(features)} pairs.")
        return np.array(features), np.array(valid_labels)

    def build_model(self, docs_directory):
        """Process documents, build embeddings, and prepare data."""
        print(f"Processing documents from: {docs_directory}")
        docs_path = Path(docs_directory)
        for file_path in docs_path.glob("*.docx"):
            try:
                citations, text,doc_id = self._extract_citations(file_path)

                # Use the same doc_id as determined in `_extract_citations`
                if citations and text:
                    self.document_embeddings[doc_id] = self._get_document_embedding(text)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        print(f"Processed {len(self.documents)} documents.")

    def train_and_evaluate_model(self):
        """Train a supervised model for citation prediction and evaluate it."""
        print("Creating training data...")
        pairs, labels = self.create_pairs()
        if not pairs:
            print("No data available for training and evaluation. Ensure documents are loaded and processed correctly.")
            return

        print("Generating features...")
        X, y = self.generate_features(pairs, labels)

        if len(X) == 0:
            print("No features generated. Check document embeddings and pairs.")
            return

        print("Splitting train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training set size: {len(y_train)}, Test set size: {len(y_test)}")

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            print("Train or test set is empty. Check the data split parameters and input data.")
            return
        
        print("Training Set Label Distribution:", np.unique(y_train, return_counts=True))
        print("Test Set Label Distribution:", np.unique(y_test, return_counts=True))
        print("Training Label Distribution:", np.unique(y_train, return_counts=True))
        print("Test Label Distribution:", np.unique(y_test, return_counts=True))


        print("Training classifier...")
        X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten for RandomForest
        X_test = X_test.reshape(X_test.shape[0], -1)

        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        print("Evaluating classifier...")
        y_pred = classifier.predict(X_test)

        # Handle single-class cases
        if len(classifier.classes_) == 1:
            print("Warning: Only one class present in the data. Skipping probability evaluation.")
            y_prob = np.zeros_like(y_pred)  # Set probabilities to zero
        else:
            y_prob = classifier.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float('nan')
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

        print("\n--- Model Performance ---")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC: {auc if not np.isnan(auc) else 'N/A'}")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        self.classifier = classifier
        self.metrics = {
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        false_positives = [(pairs[i], labels[i]) for i in range(len(y_pred)) if y_pred[i] == 1 and y_test[i] == 0]
        false_negatives = [(pairs[i], labels[i]) for i in range(len(y_pred)) if y_pred[i] == 0 and y_test[i] == 1]
        print("False Positives:", false_positives[:5])
        print("False Negatives:", false_negatives[:5])


    def predict_citations_supervised(self, input_text):
        """Predict citations for a new document."""
        print("Predicting citations...")
        input_embedding = self._get_document_embedding(input_text)
        predictions = []
        for doc_id, doc_embedding in self.document_embeddings.items():
            combined = torch.cat((input_embedding, doc_embedding, torch.abs(input_embedding - doc_embedding)), dim=-1)
            combined = combined.numpy().reshape(1, -1)
            prob = self.classifier.predict_proba(combined)[0][1]
            if prob > 0.8:  # Threshold for prediction
                predictions.append(doc_id)
        return predictions


    
    def clean_text(self, text):  
        """Clean Hebrew legal text"""
        # Remove direction markers
        text = re.sub(r'\{dir="[^"]+"\}', '', text)
        # Remove brackets
        text = re.sub(r'[\[\]]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def export_data_to_csv(self, filename="citation_data.csv"):
            """Export all processed data, pairs, and labels to a CSV for analysis."""
            rows = []

            # Add document-level data
            for doc_id, doc_data in self.documents.items():
                for citation in doc_data['citations']:
                    rows.append({
                        'doc_id': doc_id,
                        'text': doc_data['text'],
                        'citation': citation,
                        'label': 1  # Positive pair
                    })

            # Add negative pairs
            all_docs = list(self.documents.keys())
            for doc_id in self.documents.keys():
                negative_samples = np.random.choice(
                    [doc for doc in all_docs if doc not in self.documents[doc_id]['citations']],
                    size=min(len(self.documents[doc_id]['citations']), len(all_docs) - 1),
                    replace=False
                )
                for neg_doc in negative_samples:
                    rows.append({
                        'doc_id': doc_id,
                        'text': self.documents[doc_id]['text'],
                        'citation': neg_doc,
                        'label': 0  # Negative pair
                    })

            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"Data exported to {filename}")

    def extract_acronym_and_number(self,verdict_id):
        """Extract acronym and number for verdict IDs using space as separator."""
        parts = verdict_id.split(" ", 1)  # Split into two parts: before and after the first space
        acronym = parts[0] if len(parts) > 0 else "Unknown"
        number = parts[1] if len(parts) > 1 else "Unknown"
        return acronym.strip(), number.strip()

    def create_verdict_dataframe(self):
        """Create a DataFrame with all acronyms and IDs."""
        rows = set()  # Using a set to ensure unique rows
        for doc_id, doc_data in self.documents.items():
            doc_acronym, doc_number = self.extract_acronym_and_number(doc_id)
            rows.add((doc_acronym, doc_number, doc_id,",".join(doc_data['citations'])))  # Add document's acronym, number, and full ID
            for citation in doc_data['citations']:
                citation_acronym, citation_number = self.extract_acronym_and_number(citation)
                rows.add((citation_acronym, citation_number, citation))  # Add citation's acronym, number, and full ID

        # Convert to DataFrame
        df = pd.DataFrame(
            list(rows), 
            columns=["acronym", "id", "full_id","all citation"] 
        )
        df.to_csv("data.csv", index=False, encoding='utf-8')
        print(f"Data exported to data.csv")


    def compute_statistics(self):
        """Compute and display statistics on the verdict data."""
        rows = set()
        verdict_citation_counts = defaultdict(int)  # Track citation counts per verdict

        for doc_id, doc_data in self.documents.items():
            doc_acronym, doc_number = self.extract_acronym_and_number(doc_id)
            rows.add((doc_acronym, doc_number))
            for citation in doc_data['citations']:
                citation_acronym, citation_number = self.extract_acronym_and_number(citation)
                rows.add((citation_acronym, citation_number))
                verdict_citation_counts[citation] += 1  # Increment citation count

        # Convert to DataFrame
        df = pd.DataFrame(
            list(rows), 
            columns=["acronym", "id"]
        )
        
        # Total unique acronyms and verdict IDs
        total_acronyms = df['acronym'].nunique()
        total_verdict_ids = df['id'].nunique()

        # Frequency distribution of acronyms
        acronym_counts = df['acronym'].value_counts()

        # Mean, median, min, max citations per document
        citation_counts = [len(doc_data['citations']) for doc_data in self.documents.values()]
        mean_citations = np.mean(citation_counts)
        median_citations = np.median(citation_counts)
        min_citations = np.min(citation_counts)
        max_citations = np.max(citation_counts)

        # Most and least cited verdicts
        if verdict_citation_counts:
            most_cited_verdict = max(verdict_citation_counts, key=verdict_citation_counts.get)
            least_cited_verdict = min(verdict_citation_counts, key=verdict_citation_counts.get)
            most_cited_count = verdict_citation_counts[most_cited_verdict]
            least_cited_count = verdict_citation_counts[least_cited_verdict]
        else:
            most_cited_verdict = "N/A"
            least_cited_verdict = "N/A"
            most_cited_count = 0
            least_cited_count = 0

        # Display statistics
        print("\n--- Data Statistics ---")
        print(f"Total Unique Acronyms: {total_acronyms}")
        print(f"Total Unique Verdict IDs: {total_verdict_ids}")
        print(f"Top Acronyms:\n{acronym_counts.head(10)}")
        print(f"Mean Citations per Document: {mean_citations:.2f}")
        print(f"Median Citations per Document: {median_citations}")
        print(f"Min Citations per Document: {min_citations}")
        print(f"Max Citations per Document: {max_citations}")
        print(f"Most Cited Verdict: {most_cited_verdict} ({most_cited_count} citations)")
        print(f"Least Cited Verdict: {least_cited_verdict} ({least_cited_count} citations)")

        # Export statistics to CSV for analysis
        acronym_counts.to_csv("acronym_statistics.csv", index=True, header=["count"])
        print("Acronym frequency statistics exported to acronym_statistics.csv")

    def compute_top_cited_verdicts(self, top_n=10):
        """Compute the top N most cited verdicts."""
        citation_counts = {}

        for doc_id, doc_data in self.documents.items():
            for citation in doc_data['citations']:
                citation_counts[citation] = citation_counts.get(citation, 0) + 1

        sorted_citations = sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {top_n} Most Cited Verdicts:")
        for verdict, count in sorted_citations[:top_n]:
            print(f"{verdict}: {count} times")

        return sorted_citations[:top_n]





if __name__ == '__main__':
    predictor = HebrewLegalCitationPredictor()
    docs_directory ='/Users/liorb/Library/CloudStorage/GoogleDrive-liorkob@post.bgu.ac.il/.shortcut-targets-by-id/1f5AVMhCLkfM_ZoGYf_oDNiTxa2jVgL2B/חיזוי מתחמי ענישה/נתונים/New docx database/2018_cases_v1/docx'
    predictor.build_model(docs_directory)
    predictor.export_data_to_csv("citation_data.csv")
    predictor.compute_statistics()
    predictor.compute_top_cited_verdicts()
    predictor.create_verdict_dataframe()
    predictor.train_and_evaluate_model()

