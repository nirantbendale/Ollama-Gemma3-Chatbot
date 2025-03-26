import json
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the knowledge base
try:
    with open('knowledge_base_computer.json', 'r') as f:
        knowledge_base = json.load(f)
except FileNotFoundError:
    print("Error: 'knowledge_base_computer.json' not found. Please ensure the file exists.")
    exit()
except json.JSONDecodeError:
    print("Error: Invalid JSON in 'knowledge_base_computer.json'. Please check the file format.")
    exit()

# Initialize the sentence transformer for semantic matching
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare embeddings for faults in the knowledge base
faults = [entry['fault'] for entry in knowledge_base]
fault_embeddings = embedder.encode(faults)

# Define a domain description to check if queries are on-topic
domain_description = "Issues with computers and equipment only."
domain_embedding = embedder.encode([domain_description])[0]

# Set similarity thresholds
off_topic_threshold = 0.2  # Below this, the query is considered off-topic
match_threshold = 0.7     # Above this, a knowledge base match is accepted

# Chatbot loop
print("Welcome to the computer issues Chatbot!")
print("Describe your issue, or type 'exit' to quit.\n")

while True:
    user_input = input("Your issue: ").strip()
    
    # Exit condition
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    
    # Handle empty input
    if not user_input:
        print("Please describe your computer issue.")
        continue
    
    # Compute embedding for user input
    user_embedding = embedder.encode([user_input])[0]
    
    # Check if the query is on-topic
    domain_similarity = cosine_similarity([user_embedding], [domain_embedding])[0][0]
    if domain_similarity < off_topic_threshold:
        print("I can only assist with computer issues. Please ask a relevant question.")
        continue
    
    # Search the knowledge base for a match
    similarities = cosine_similarity([user_embedding], fault_embeddings)[0]
    max_similarity = max(similarities)
    
    if max_similarity > match_threshold:
        # Found a match in the knowledge base
        matched_idx = similarities.argmax()
        matched_entry = knowledge_base[matched_idx]
        print(f"Diagnosis: {matched_entry['cause']}. Recommended actions: {matched_entry['solution']}.")
    else:
        # No match found; use the LLM
        # Format the knowledge base as a string for the prompt
        knowledge_base_str = "\n\n".join([
            f"Fault: {entry['fault']}\nCause: {entry['cause']}\nSolution: {entry['solution']}"
            for entry in knowledge_base
        ])
        prompt = (
            f"Here is a knowledge base of common faults in computers:\n\n"
            f"{knowledge_base_str}\n\n"
            f"Based on this knowledge base, please provide a diagnosis and recommended in only 2 sentences maximum by describing their cause and solution."
            f"actions for the following issue: {user_input}"
        )
        
        # Prepare the payload for the Ollama API
        payload = {
            "model": "gemma3",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in diagnosing faults in computers. "
                               "You can only respond to queries related to computers in only 2 sentences maximum by describing their cause and solution. If the issue is not related to computer, simply reject the user prompt with the following message 'I can only assist with beer flow system issues. Please ask a relevant question.'"
                },
                {"role": "user", "content": prompt}
            ]
        }
        
        # Send request to the LLM with streaming response
        try:
            response = requests.post("http://127.0.0.1:11434/api/chat", json=payload, stream=True)
            if response.status_code != 200:
                print(f"Error: HTTP {response.status_code}. Could not reach the LLM.")
                continue
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_obj = json.loads(line.decode('utf-8'))
                        if "message" in json_obj and "content" in json_obj["message"]:
                            full_response += json_obj["message"]["content"]
                        if "done" in json_obj and json_obj["done"]:
                            break
                    except json.JSONDecodeError:
                        continue
            
            if full_response:
                print(full_response)
            else:
                print("Error: No response from the LLM.")
        
        except requests.exceptions.RequestException as e:
            print(f"Error: Could not connect to the Ollama API. {str(e)}")
            continue