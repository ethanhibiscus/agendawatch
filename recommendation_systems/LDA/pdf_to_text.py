import os
import fitz  # PyMuPDF

def convert_pdf_to_text(pdf_path, txt_path):
    try:
        # Open the PDF file
        pdf_document = fitz.Document(pdf_path)
        
        # Extract text from each page
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()

        # Save the extracted text to a text file
        with open(txt_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)
        print(f"Successfully converted {pdf_path} to {txt_path}")
    except Exception as e:
        print(f"Failed to convert {pdf_path}: {str(e)}")

# Directory containing the PDFs
pdf_dir = './civicplus_docs-3'
# Directory to save the text files
txt_dir = './converted_text_files'

if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)

# Convert each PDF in the directory
for pdf_filename in os.listdir(pdf_dir):
    if pdf_filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, pdf_filename)
        txt_filename = os.path.splitext(pdf_filename)[0] + '.txt'
        txt_path = os.path.join(txt_dir, txt_filename)
        convert_pdf_to_text(pdf_path, txt_path)
