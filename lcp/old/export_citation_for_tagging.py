import docx
import re
from pathlib import Path
import csv
import pandas as pd
from typing import List, Dict

# Global CSV file path for all citation paragraphs
# global_csv_path = 'global_citation_paragraphs.csv'

# def append_to_global_csv(data: List[Dict[str, str]]):
#     """Append citation paragraphs to a global CSV."""
#     if not Path(global_csv_path).exists():
#         # Create CSV file with headers if it doesn't exist
#         pd.DataFrame(data).to_csv(global_csv_path, index=False, encoding='utf-8-sig')
#     else:
#         # Append data to the existing file
#         existing_df = pd.read_csv(global_csv_path)
#         updated_df = pd.concat([existing_df, pd.DataFrame(data)], ignore_index=True)
#         updated_df.to_csv(global_csv_path, index=False, encoding='utf-8-sig')

def export_citation_analysis(file_path: str):
    # Dictionary to store all data
    data = {
        'citation_paragraphs': [],
        'kept_paragraphs': [],
    }
    
    try:
        # Read document
        doc = docx.Document(file_path)
        doc_name = Path(file_path).stem
        
        # First pass: Extract paragraphs with citations
        for i, paragraph in enumerate(doc.paragraphs):
            para_text = paragraph.text
            
            if (
    'ע"פ' in para_text or 'ת"פ' in para_text or 'עפ"ג' in para_text or 
    'ת.פ' in para_text or 'ע.פ' in para_text or 'עפ.ג' in para_text or
    'ע״פ' in para_text or 'ת״פ' in para_text or 'עפ״ג' in para_text
):

                if len(para_text.strip()) > 0:
                    # Store for citation paragraph processing
                    citation_data = {
                        'document': doc_name,
                        'paragraph_number': i,
                        'text': para_text
                    }
                    data['citation_paragraphs'].append(citation_data)
                    
            else:
                # Clean and store non-citation paragraphs
                cleaned_text = re.sub(r'\{dir="[^"]+"\}', '', para_text)
                cleaned_text = re.sub(r'[\[\]]', '', cleaned_text)
                cleaned_text = ' '.join(cleaned_text.split())
                
                if cleaned_text.strip():
                    data['kept_paragraphs'].append({
                        'document': doc_name,
                        'paragraph_number': i,
                        'text': cleaned_text
                    })
        
        # # Append citation paragraphs to the global CSV
        # append_to_global_csv(data['citation_paragraphs'])
        
        # Export to CSV files for individual document
        # output_dir = Path(file_path).parent
        # output_dir.mkdir(exist_ok=True)
        
        # # Export kept paragraphs
        # df_kept = pd.DataFrame(data['kept_paragraphs'])
        # df_kept.to_csv(output_dir / f'{Path(file_path).stem}_kept_paragraphs.csv', index=False, encoding='utf-8-sig')
        
        # # Export removed paragraphs
        # df_removed = pd.DataFrame(data['citation_paragraphs'])
        # df_removed.to_csv(output_dir / f'{Path(file_path).stem}_citation_paragraphs.csv', index=False, encoding='utf-8-sig')
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")


if __name__ == '__main__':
    docs_directory = '/Users/liorb/Library/CloudStorage/GoogleDrive-liorkob@post.bgu.ac.il/.shortcut-targets-by-id/1f5AVMhCLkfM_ZoGYf_oDNiTxa2jVgL2B/חיזוי מתחמי ענישה/נתונים/New docx database/2018_cases_v1/docx'
    docs_path = Path(docs_directory)
    for file_path in docs_path.glob("*.docx"):
        try:
            export_citation_analysis(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
