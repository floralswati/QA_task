import fitz  # PyMuPDF
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

# Define the dictionary of models with parameters for min and max length for generative models
models = {
    "bert": pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"),
  #"roberta": pipeline("question-answering", model="deepset/roberta-base-squad2"),
  # "bert": pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"),
   #"distilbert": pipeline("question-answering", model="distilbert-base-uncased-distilled-squad"),
   # "albert": pipeline("question-answering", model="mfeb/albert-xxlarge-v2-squad2"),
   # "xlnet": pipeline("question-answering", model="mylas02/XLNET_SQuaD_FineTuned_v2"),
    # "t5": pipeline("question-answering", model="allenai/t5-small-squad2-question-generation"),
}

# Function to extract text from a PDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")  # Use the "text" mode to extract raw text
    return text

# Function to clean the extracted text
def clean_text(text):
    # Clean up unwanted characters or split errors
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    return text.strip()

# Function to split text into multiple contexts (based on paragraphs)
def split_text_into_contexts(text, num_contexts=100):
    contexts = re.split(r'\n\n+', text)  # Use regex to split by one or more newline characters
    contexts = [clean_text(c) for c in contexts]
    return contexts[:num_contexts]

# Function to extract questions from a PDF (assuming each question is on a new line)
def extract_questions_from_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    questions = re.split(r'\n(?=[A-Z])', text)  # Assuming each question is on a new line
    return [q.strip() for q in questions if q.strip()]

# Function to apply BM25 to get the best context
def retrieve_relevant_contexts(texts, query, num_contexts=5):
    # Using BM25 for text retrieval
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)

    # Get the indices of the top contexts
    top_indices = np.argsort(scores)[::-1][:num_contexts]
    return [texts[i] for i in top_indices]

# Function to generate answers for multiple questions for each context using the provided models
def generate_answers_for_contexts(contexts, questions, models, min_length=5, max_length=50):
    answers = {}
    for model_name, model in models.items():
        model_answers = []
        for context in contexts:
            context_answers = []  # To store answers for each question in the context
            for question in questions:
                try:
                    if model_name in ["t5", "bart"]:  # Generative models
                        # Define a clear prompt for generative models
                        prompt = prompt = f"Context: {context}\nAnswer the following question in your own ten words:\n{question}"
                        result = model(prompt, max_length=max_length, min_length=min_length, do_sample=False)
                        generated_answer = result[0]['generated_text'].strip()
                        context_answers.append({
                            "question": question,
                            "answer": generated_answer
                        })
                    else:  # Extractive models (BERT, RoBERTa, etc.)
                        result = model(question=question, context=context)
                        context_answers.append({
                            "question": question,
                            "answer": result['answer']
                        })
                except Exception as e:
                    context_answers.append({
                        "question": question,
                        "answer": f"Error: {str(e)}"
                    })
            model_answers.append({
                "answers": context_answers
            })
        answers[model_name] = model_answers
    return answers

# Main function to process the PDF and generate answers
def main():
    # Paths to the PDF files
    contexts_pdf_path =  'long.pdf' # PDF containing the contexts
    questions_pdf_path = 'busq.pdf'  # PDF containing the questions

    # Extract text from the PDF for contexts
    pdf_text = extract_text_from_pdf(contexts_pdf_path)

    # Clean and split the extracted text into contexts
    contexts = split_text_into_contexts(pdf_text, num_contexts=100)

    # Extract questions from the questions PDF
    questions = extract_questions_from_pdf(questions_pdf_path)

    # Define the minimum and maximum length for generative models
    min_length = 25  # Minimum length of the answer (can adjust based on your needs)
    max_length = 100  # Maximum length of the answer (can adjust based on your needs)

    # Iterate through each question and use BM25 to retrieve the top relevant contexts
    for question in questions:
        relevant_contexts = retrieve_relevant_contexts(contexts, question)

        # Generate answers for all relevant contexts using the models
        answers = generate_answers_for_contexts(relevant_contexts, [question], models, min_length, max_length)

        # Print answers for each model (without context)
        for model_name, model_answers in answers.items():
            print(f"\nAnswers from {model_name} model:\n")
            for context_answer in model_answers:
                for qa in context_answer['answers']:
                    print(f"  Question: {qa['question']}")
                    print(f"  Answer: {qa['answer']}")
            print("=" * 50)

if __name__ == "__main__":
    main()
