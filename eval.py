import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from collections import Counter
import nltk
import re

# Download necessary NLTK data
nltk.download('punkt')

# Function to compute similarity between model answer and multiple reference answers
def compute_cosine_similarity(model_answer, reference_answers):
    vectorizer = TfidfVectorizer(stop_words='english')
    cosine_scores = []
    
    for reference_answer in reference_answers:
        answers = [model_answer, reference_answer]
        tfidf_matrix = vectorizer.fit_transform(answers)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        cosine_scores.append(similarity[0][0])
    
    return sum(cosine_scores) / len(cosine_scores)  # Average over all reference answers

# Function to compute Precision, Recall, and F1 Score manually

def preprocess(text):
    """Lowercase and remove punctuation before tokenization."""
    if not isinstance(text, str):
        text = str(text)  # Ensure it's a string
    text = text.lower()  # Normalize case
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.split()

def compute_precision_recall_f1(model_answer, reference_answers):
    # Tokenize model and reference answers
    model_tokens = preprocess(model_answer)
    reference_tokens = [preprocess(ref) for ref in reference_answers]
    
    # Flatten reference tokens
    reference_tokens = [token for sublist in reference_tokens for token in sublist]
    
    # Count occurrences of each token
    model_token_counts = Counter(model_tokens)
    reference_token_counts = Counter(reference_tokens)
    
    # Debugging: Print token lists
    print("Model Tokens:", model_tokens)
    print("Reference Tokens:", reference_tokens)

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    true_positives = sum((model_token_counts & reference_token_counts).values())
    false_positives = sum((model_token_counts - reference_token_counts).values())
    false_negatives = sum((reference_token_counts - model_token_counts).values())
    
    # Calculate Precision, Recall, and F1 Score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Function to compute BLEU score using multiple references with smoothing
def compute_bleu_score(model_answer, reference_answers):
    model_tokens = model_answer.split()
    reference_tokens_list = [ref.split() for ref in reference_answers]
    bleu_scores = []
    
    smoothing_function = SmoothingFunction().method4  # Use smoothing to avoid zero scores
    
    for reference_tokens in reference_tokens_list:
        score = sentence_bleu([reference_tokens], model_tokens, smoothing_function=smoothing_function)
        bleu_scores.append(score)
    
    return sum(bleu_scores) / len(bleu_scores)  # Average over all reference answers

# Function to compute ROUGE score using multiple references
def compute_rouge_score(model_answer, reference_answers):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_1_f1, rouge_2_f1, rouge_l_f1 = 0, 0, 0
    
    for reference_answer in reference_answers:
        scores = scorer.score(reference_answer, model_answer)
        rouge_1_f1 += scores['rouge1'].fmeasure
        rouge_2_f1 += scores['rouge2'].fmeasure
        rouge_l_f1 += scores['rougeL'].fmeasure
    
    # Average over all reference answers
    num_references = len(reference_answers)
    return rouge_1_f1 / num_references, rouge_2_f1 / num_references, rouge_l_f1 / num_references

# Load the CSV file with questions, model answers, and multiple reference answers
data = pd.read_csv('distil_article.csv')

# Strip whitespace from column names to ensure no hidden characters
data.columns = data.columns.str.strip()

# Initialize an empty list to store the results
scores = []

# Loop through the rows and compute similarity score for each question
for index, row in data.iterrows():
    question_id = row['question_id']
    model_answer = row['model_answer']
    
    # Collect all reference answers (assumed to be non-empty)
    reference_answers = [row[f'reference_answer{i}'] for i in range(1, 4)]
    
    # Debugging print to show that references are correctly collected
    print(f"Question {question_id} references collected: {len(reference_answers)}")
    
    # Compute Precision, Recall, F1 Score
    precision, recall, f1_score_value = compute_precision_recall_f1(model_answer, reference_answers)
    
    # Compute other metrics (Cosine, BLEU, ROUGE)
    cosine_score = compute_cosine_similarity(model_answer, reference_answers)
    bleu_score = compute_bleu_score(model_answer, reference_answers)
    rouge_1, rouge_2, rouge_l = compute_rouge_score(model_answer, reference_answers)
    
    # Append the results for this question
    scores.append({
        'question_id': question_id,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score_value,
        'cosine_similarity': cosine_score,
        'bleu_score': bleu_score,
        'rouge_1_f1': rouge_1,
        'rouge_2_f1': rouge_2,
        'rouge_l_f1': rouge_l
    })

# Create a DataFrame to store the results
results_df = pd.DataFrame(scores)

# Save the results to a CSV file
results_df.to_csv('distil_score_article.csv', index=False)

print("Evaluation scores (Precision, Recall, F1, Cosine, BLEU, ROUGE) have been calculated and saved to 'evaluation_scores.csv'")
