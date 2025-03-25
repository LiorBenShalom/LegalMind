from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from docx import Document
import torch
import csv
import requests
from anthropic import Anthropic
api_key = "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"

torch.cuda.empty_cache()
def chunk_text(text, max_length):
    """Chunk text into smaller parts to fit model constraints."""
    words = text.split()
    for i in range(0, len(words), max_length):
        yield " ".join(words[i:i + max_length])

# DictaLM model setup
dictalm_model_name = "dicta-il/dictalm2.0-instruct"
dictalm_tokenizer = AutoTokenizer.from_pretrained(dictalm_model_name)
device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9 else "cpu")
dictalm_model = AutoModelForCausalLM.from_pretrained(
    dictalm_model_name,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

dictalm_generator = pipeline("text-generation", model=dictalm_model, tokenizer=dictalm_tokenizer)

# Function to query DictaLM
def query_dictalm(text):
    max_length = 512  # הגבלת אורך הטקסט
    responses = []
    for chunk in chunk_text(text, max_length):
        prompt = (
            "במסמכי גזר דין, יש לאתר ולהציג במלואו את החלק העוסק בקביעת מתחם הענישה ו/או מדיניות הענישה הנוהגת. "
            "החלק עשוי להופיע תחת כותרות ברורות כגון 'מתחם הענישה', 'מדיניות הענישה הנוהגת', 'קביעת מתחם העונש', או להיות משולב בתוך הטקסט. "
            "עלייך להחזיר את הטקסט המלא שמתחיל מהתייחסות מפורשת או משתמעת למתחם הענישה, כולל כל הציטוטים, הניתוחים, הדוגמאות והחלטות השופט, ועד לסיום דיון זה. "
            "אין להסיר שום פרט מתוך החלק הזה, גם אם הוא כולל כותרות, ניתוחים או מידע שאינו נראה ישירות קשור. "
            "אם אין חלק כזה במסמך, החזר תשובה ברורה: 'לא נמצא טקסט העוסק במתחם הענישה.'\n\n"
            f"טקסט: {chunk}\n"
            "תשובה:"
        )
        try:
            response = dictalm_generator(prompt, max_new_tokens=1024, num_return_sequences=1)
            responses.append(response[0]["generated_text"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("Switching to CPU due to GPU memory constraints.")
            return "Error: CUDA out of memory"
    return " ".join(responses)

# Function to query Claude
def query_claude(text):
    client = Anthropic(api_key=api_key)
    prompt = (
        "במסמכי גזר דין, יש לאתר ולהציג במלואו את החלק העוסק בקביעת מתחם הענישה ו/או מדיניות הענישה הנוהגת. "
        "החלק עשוי להופיע תחת כותרות ברורות כגון 'מתחם הענישה', 'מדיניות הענישה הנוהגת', 'קביעת מתחם העונש', או להיות משולב בתוך הטקסט. "
        "עלייך להחזיר את הטקסט המלא שמתחיל מהתייחסות מפורשת או משתמעת למתחם הענישה, כולל כל הציטוטים, הניתוחים, הדוגמאות והחלטות השופט, ועד לסיום דיון זה. "
        "אין להסיר שום פרט מתוך החלק הזה, גם אם הוא כולל כותרות, ניתוחים או מידע שאינו נראה ישירות קשור. "
        "אם אין חלק כזה במסמך, החזר תשובה ברורה: 'לא נמצא טקסט העוסק במתחם הענישה.'\n\n"
        f"טקסט: {text}\n"
        "תשובה:"
    )
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.content[0].text
        return response_text 
    except Exception as e:
        print(f"Error processing with Claude API: {str(e)}")
        return f"Error: {e}"

# Function to read .docx files
def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

# Process files and query both models
def process_files(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".docx"):
            file_path = os.path.join(directory, filename)
            try:
                text = read_docx(file_path)
                dictalm_response = query_dictalm(text)
                claude_response = query_claude(text)
                results.append({
                    "Filename": filename,
                    "DictaLM Response": dictalm_response,
                    "Claude Response": claude_response
                })
                torch.cuda.empty_cache()  # Clear GPU cache after processing each file
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return results
# Save results to CSV
def save_to_csv(results, output_file):
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Filename", "DictaLM Response", "Claude Response"])
        writer.writeheader()
        writer.writerows(results)


# Main script
if __name__ == "__main__":
    docs_directory = "/home/liorkob/thesis/nlp_course/lcp/docx"
    results = process_files(docs_directory)
    output_file = "results.csv"
    save_to_csv(results, output_file)
    print(f"Results have been saved to {output_file}.")
