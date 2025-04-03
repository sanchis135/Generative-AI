import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import faiss
import os

# Set the environment variable to avoid the OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Define the function for query vector representation
def query_vector_representation(query):
    return [1, 2]

# Let's assume we have a list of documents
documents = ["Paris is the capital of France.", "London is the capital of England.", "Berlin is the capital of Germany."]

# Convert the documents into vectors (this is just a simple example)
document_vectors = np.array([[1, 2], [3, 4], [5, 6]]).astype('float32')

# Create a FAISS index
index = faiss.IndexFlatL2(document_vectors.shape[1])
index.add(document_vectors)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

def retrieve_and_generate(query, documents, index, model, tokenizer):
    try:
        # Convert the query into a vector
        query_vector = np.array([query_vector_representation(query)]).astype('float32')
        
        # Retrieve relevant documents
        D, I = index.search(query_vector, k=1)
        retrieved_document = documents[I[0][0]]
        
        # Generate an answer using the generative model
        inputs = tokenizer(query, retrieved_document, return_tensors="pt")
        outputs = model(**inputs)
        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax()
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end+1]))
        
        return answer
    except Exception as e:
        print(f"Error: {e}")
        return None

# Usage example
query = "What is the capital of France?"
answer = retrieve_and_generate(query, documents, index, model, tokenizer)
print(f"Question: {query}")
print(f"Answer: {answer}")


# Define use cases
#use_cases = [
#    {"query": "What is the capital of France?", "expected_answer": "Paris"},
#    {"query": "What is the capital of England?", "expected_answer": "London"},
#    {"query": "What is the capital of Germany?", "expected_answer": "Berlin"}
#]
#
# Evaluate the system
#for case in use_cases:
#    query = case["query"]
#    expected_answer = case["expected_answer"]
#    answer = retrieve_and_generate(query, documents, index, model, tokenizer)
#    print(f"Question: {query}")
#    print(f"Expected answer: {expected_answer}")
#    print(f"Generated answer: {answer}")
#    print(f"Correct: {answer == expected_answer}\n")
#
# Calculate accuracy
#correct_answers = 0
#total_cases = len(use_cases)
#
#for case in use_cases:
#    query = case["query"]
#    expected_answer = case["expected_answer"]
#    answer = retrieve_and_generate(query, documents, index, model, tokenizer)
#    if answer == expected_answer:
#        correct_answers += 1
#
#accuracy = correct_answers / total_cases
#print(f"Accuracy: {accuracy * 100}%")