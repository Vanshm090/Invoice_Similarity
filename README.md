# Invoice_Similarity
For DeepLogicAI

This Python script compares an input invoice (PDF format) to a database of existing invoices to identify the most similar invoice based on content and structural similarity.

## Requirements

- Python 3.x
- PyMuPDF (fitz)
- scikit-learn

Install the required packages using pip:

```sh
pip install pymupdf scikit-learn
```

# Usage
1) Place your training and test PDF invoices in separate folders.
2) Update the paths in the script:

```
training_invoices_folder = r"path/to/train"
test_invoices_folder = r"path/to/test"
```
3) Run the script

# Output
The script will print the most similar training invoice for each test invoice along with the similarity score.

# Script Overview
Extract text from PDFs.
Preprocess the text.
Extract features using TF-IDF.
Compare the similarity between test and training invoices.

# Sample Output
Test Invoice test_invoice_1.pdf:
Most similar training invoice: train_invoice_3.pdf
Similarity score: 0.8794

