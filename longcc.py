import fitz  # PyMuPDF
import re
from transformers import pipeline
from rank_bm25 import BM25Okapi
import numpy as np

# Define the dictionary of models
models = {
    "bert": pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"),
   # "roberta": pipeline("question-answering", model="deepset/roberta-base-squad2"),
   # "distilbert": pipeline("question-answering", model="distilbert-base-uncased-distilled-squad"),
  #  "albert": pipeline("question-answering", model="mfeb/albert-xxlarge-v2-squad2"),
   # "xlnet": pipeline("question-answering", model="mylas02/XLNET_SQuaD_FineTuned_v2"),
}

# Extract text from multiple PDFs
def extract_text_from_pdfs(pdf_paths):
    all_text = {}
    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text("text") + "\n\n"  # Preserve paragraph structure
        all_text[pdf_path] = text
    return all_text

# Clean extracted text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    return text.strip()

# Split text into multiple contexts (paragraphs)
def split_text_into_contexts(text, num_contexts=100):
    contexts = re.split(r'\n\n+', text)  # Split by paragraph-like spacing
    contexts = [clean_text(c) for c in contexts if c.strip()]
    return contexts[:num_contexts]

# Extract questions from a PDF
def extract_questions_from_pdf(pdf_path):
    # Get the text from the question PDF
    all_text = extract_text_from_pdfs([pdf_path])  # This returns a dictionary
    text = all_text.get(pdf_path, "")  # Extract the text from the dictionary using the PDF path
    questions = re.split(r'\n(?=[A-Z])', text)  # Assuming each question starts with a capital letter
    return [q.strip() for q in questions if q.strip()]

# Retrieve top relevant contexts using BM25
def retrieve_relevant_contexts(texts, query, num_contexts=5):
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:num_contexts]
    return [texts[i] for i in top_indices]

# Generate answers for multiple contexts using the models
def generate_answers_for_contexts(contexts, questions, models, min_length=5, max_length=50):
    answers = {}
    for model_name, model in models.items():
        model_answers = {version: [] for version in contexts}
        for version, context_set in contexts.items():
            for i, question in enumerate(questions[:10]):  # Only take the first 10 questions
                relevant_contexts = retrieve_relevant_contexts(context_set, question)
                context_answers = []  # Store answers for each question in the context
                for context in relevant_contexts:
                    try:
                        result = model(question=question, context=context)
                        answer_text = result['answer']
                    except Exception as e:
                        answer_text = f"Error: {str(e)}"
                    context_answers.append(f"{question} {answer_text}")
                model_answers[version].extend(context_answers)
        answers[model_name] = model_answers
    return answers

# Main function
def main():
    # Paths to PDFs for different contexts
    contexts_pdf_paths = {
        'long': 'long.pdf',
        'short': 'short.pdf',
        'ommit': 'ommit.pdf',
        'noisy': 'noisy.pdf',
        'para': 'para.pdf',
        'pv': 'pv.pdf',
    }

    questions_pdf_path = 'undpq.pdf'  # PDF containing the questions

    # Extract questions
    questions = extract_questions_from_pdf(questions_pdf_path)

    # Extract and process text from each PDF separately
    all_text = extract_text_from_pdfs(list(contexts_pdf_paths.values()))

    # Split text into contexts for each PDF
    all_contexts = {}
    for version, text in all_text.items():
        contexts = split_text_into_contexts(text, num_contexts=100)
        all_contexts[version] = contexts

    # Define the minimum and maximum length for answers
    min_length = 25
    max_length = 100

    # Generate answers for each model
    answers = generate_answers_for_contexts(all_contexts, questions, models, min_length, max_length)

    # Print answers grouped by model and context version
    for model_name, model_answers in answers.items():
        print(f"{model_name} answers:")
        for version, context_answers in model_answers.items():
            print(f"{version}: ", end="")
            print(" ".join(context_answers))
        print("=" * 80)

if __name__ == "__main__":
    main()
