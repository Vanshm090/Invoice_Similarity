import fitz  
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Text Extraction
def extract_text_from_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

#Text Preprocessing
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  
    text = text.lower()  
    return text

#Extracting Features
def extract_features(texts):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

#Finding Similarites
def find_most_similar_invoice(input_text, existing_texts):
    all_texts = existing_texts + [input_text]
    features, vectorizer = extract_features(all_texts)

    similarity_matrix = cosine_similarity(features)
    similarities = similarity_matrix[-1, :-1]

    most_similar_index = similarities.argmax()
    return most_similar_index, similarities[most_similar_index]

#Loading data
def load_invoices(folder_path):
    texts = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path)
            preprocessed_text = preprocess_text(text)
            texts.append(preprocessed_text)
            filenames.append(filename)
    return texts, filenames

#Commparing input data
def compare_test_to_training(test_texts, training_texts):
    results = []
    for test_text in test_texts:
        most_similar_index, similarity_score = find_most_similar_invoice(test_text, training_texts)
        results.append((most_similar_index, similarity_score))
    return results

#Taking paths from local storage where invoice are kept
training_invoices_folder = r"D:\OneDrive\Desktop\invoice files\train" #path is where I stored files in my local storage, must be modified accordingly
test_invoices_folder = r"D:\OneDrive\Desktop\invoice files\test" #Please give your local path before running, else code will fail to run


training_texts, training_filenames = load_invoices(training_invoices_folder)


test_texts, test_filenames = load_invoices(test_invoices_folder)

# Compare test invoices to training invoices
comparison_results = compare_test_to_training(test_texts, training_texts)

# results 
for i, (index, score) in enumerate(comparison_results):
    print(f"Test Invoice {test_filenames[i]}:")
    print(f"  Most similar training invoice: {training_filenames[index]}")
    print(f"  Similarity score: {score:.4f}")
