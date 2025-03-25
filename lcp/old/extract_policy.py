from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from docx import Document
import torch
device = torch.device("cuda") if torch.backends.mps.is_available() else "cpu"
# device = torch.device("cpu")
torch.cuda.empty_cache()
# model_name = "dicta-il/dictalm2.0"
model_name = "dicta-il/dictalm2.0-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
# model = model.to(device)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # Optional: reduce memory usage
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)



def read_docx(file_path):
    """Read the content of a .docx file."""
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

# def query_dictalm(text):
#     """Query DictaLM with a specific prompt."""
#     prompt = (
#         "בגזרי דין על מנת לקבוע את מתחם הענישה לנאשם יש שימוש בציטוטים של מקרים דומים. "
#         "חלק זה נקרא מדיניות הענישה הנוהגת. "
#         "שאלה: מהו החלק המתאר את מדיניות הענישה הנוהגת? אם אין חלק כזה השיבי שאין חלק כזה.\n"
#         f"טקסט: {text}\n"
#         "תשובה:"
#     )
#     response = generator(prompt, max_new_tokens=400, num_return_sequences=1)
#     return response[0]["generated_text"]

def query_dictalm(text):
    """Query DictaLM with a specific prompt."""
    prompt = (
        "במסמכי גזר דין, מתחם הענישה מופיע לעיתים תחת כותרות ברורות כמו 'מתחם העונש' או 'מדיניות הענישה הנוהגת', "
        "ולעיתים הוא משולב בתוך הטקסט לאחר אזכורים כמו 'בבחינת נסיבות המקרה' או 'בהתאם לפסיקה הנוהגת'. "
        "אנא מצאי והציגי את החלק המתאר בטקסט את מדיניות הענישה הנוהגת או מתחם הענישה, תוך זיהוי סמנטי, גם אם אין כותרת מפורשת. "
        "אם אין חלק כזה במסמך, כתבי שאין חלק כזה.\n\n"
        "דוגמאות לזיהוי:\n"
        "- טקסט שמתחיל ב'בבחינת מדיניות הענישה הנוהגת...'\n"
        "- כותרת כמו 'מתחם העונש' או 'מדיניות הענישה הנוהגת'\n"
        "- אזכור של מקרים דומים או תיקים דומים עם קביעת מתחם ענישה (לדוגמה: 'נקבע כי המתחם נע בין X ל-Y').\n\n"
        f"טקסט: {text}\n"
        "תשובה:"
    )
    response = generator(prompt, max_new_tokens=400, num_return_sequences=1)
    return response[0]["generated_text"]




# def extract_last_half(text):
#     """Extract the last 50% of the text."""
#     lines = text.split("\n")
#     half_index = len(lines) // 2
#     return "\n".join(lines[half_index:])

def process_files(directory):
    """Process all .docx files in a directory."""
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith(".docx"):
            file_path = os.path.join(directory, filename)
            try:
                full_text = read_docx(file_path)
                # last_half = extract_last_half(full_text)
                response = query_dictalm(full_text)
                results[filename] = response
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return results



# Example usage
if __name__ == "__main__":
    docs_directory = "/home/liorkob/thesis/nlp_course/lcp/docx"
    # docs_directory = "/Users/liorb/Library/CloudStorage/GoogleDrive-liorkob@post.bgu.ac.il/.shortcut-targets-by-id/1f5AVMhCLkfM_ZoGYf_oDNiTxa2jVgL2B/חיזוי מתחמי ענישה/נתונים/New docx database/2018_cases_v1/docx"
    results = process_files(docs_directory)

    output_file = "results.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        for doc, response in results.items():
            file.write(f"File: {doc}\nResponse:\n{response}\n\n")

    print(f"Results have been saved to {output_file}.")
