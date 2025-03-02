import fitz  # PyMuPDF
import re
from transformers import pipeline

# Define the dictionary of models with parameters for min and max length for generative models
models = {
    #"bert": pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"),
    #"roberta": pipeline("question-answering", model="deepset/roberta-base-squad2"),
    #"distilbert": pipeline("question-answering", model="distilbert-base-uncased-distilled-squad"),
   # "albert": pipeline("question-answering", model="mfeb/albert-xxlarge-v2-squad2"),
    "xlnet": pipeline("question-answering", model="mylas02/XLNET_SQuaD_FineTuned_v2"),
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
    # Attempt a more robust split using regex to capture potential paragraph breaks or blank lines
    contexts = re.split(r'\n\n+', text)  # Use regex to split by one or more newline characters
    contexts = [clean_text(c) for c in contexts]
    return contexts[:num_contexts]  # Limit the number of contexts to the specified maximum

# Function to extract questions from a PDF (assuming each question is on a new line)
def extract_questions_from_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    questions = text.split('\n')  # Assuming each question is on a new line
    return [q.strip() for q in questions if q.strip()]

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
                        result = model(question=question, context=context, min_length=min_length, max_length=max_length)
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

# Main function to process the PDFs and generate answers
def main():
    # Paths to the PDF files
    contexts_pdfs_paths = ['long.pdf', 'short.pdf','ommit.pdf','noisy.pdf', 'para.pdf','pv.pdf']  # List of context PDFs
    #contexts_pdfs_paths =  ['long.pdf', 'short copy.pdf','ommit copy.pdf','noisy copy.pdf', 'para copy.pdf','pv copy.pdf']
    #contexts_pdfs_paths=['pv.pdf']
    questions_pdf_path = 'entq.pdf'  # PDF containing the questions

    # Extract questions from the questions PDF
    questions = extract_questions_from_pdf(questions_pdf_path)

    # Initialize an empty list to hold all the contexts
    all_contexts = []

    # Process all the context PDFs
    for context_pdf_path in contexts_pdfs_paths:
        pdf_text = extract_text_from_pdf(context_pdf_path)  # Extract text from each context PDF
        contexts = split_text_into_contexts(pdf_text, num_contexts=100)  # Split text into contexts
        all_contexts.extend(contexts)  # Add these contexts to the global list

    # Define the minimum and maximum length for generative models
    min_length = 25  # Minimum length of the answer (can adjust based on your needs)
    max_length = 100  # Maximum length of the answer (can adjust based on your needs)

    # Generate answers for all contexts and all questions using the models
    answers = generate_answers_for_contexts(all_contexts, questions, models, min_length, max_length)

    # Print answers for each model
    for model_name, model_answers in answers.items():
        print(f"\nAnswers from {model_name} model:\n")
        for context_answer in model_answers:
            for qa in context_answer['answers']:
                print(f"  Question: {qa['question']}")
                print(f"  Answer: {qa['answer']}")
        print("=" * 50)

if __name__ == "__main__":
    main()