from gtts import gTTS
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
import concurrent.futures
import time, logging
from transformers import pipeline

# Set the logging level to suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize summarization pipelines
extractive_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def qa_session(story_content):
    story = story_content
    while True:
        print("Story Master: Please ask a question? (Type 'exit' to end Q/A session)")
        question = input("User: ")
        if question.lower() == 'exit':
            break
        answer_question(question, story)

def listen_to_story(story):
    os.system(f"start speech/{story}.mp3")

def continue_after_action():
    print("\nStory Master: What would you like to do next?")
    print("  1: Continue with this story")
    print("  2: Choose another story")
    print("  0: Exit")
    next_action = input("User: Enter your choice (0-2): ")
    if next_action == '2':
        return True  # This will go back to the main loop
    elif next_action == '0':
        print("Story Master: Thank you for using the Storyteller Services. See you soon!")
        exit()
    return False


def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    
    return ' '.join(tokens)

def preprocess_query(query):
    # Tokenization
    tokens = word_tokenize(query)
    
    # Remove punctuation and convert to lowercase
    tokens = [word.lower() for word in tokens if word.isalpha()]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    
    return ' '.join(tokens)

def index_documents(folder_path):
    documents = []
    document_names = []  # New list to store document names
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            document_names.append(filename)  # Store document name
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                processed_content = preprocess_text(content)
                documents.append(processed_content)
    
    return documents, document_names

def vsm_search(query, documents, document_names):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query] + documents)
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
    
    # Sort documents by similarity in descending order
    sorted_documents = sorted(zip(document_names, cosine_similarities), key=lambda x: x[1], reverse=True)
    
    return sorted_documents

def read_story(selected_story):
    file_path = f'stories\{selected_story}' 
    with open(file_path, 'r', encoding= 'utf-8') as file:
        text_from_file = file.read()
    return text_from_file

def read_story_action(story_content):
    print(story_content)
    while True:
        print("\nType 'exit' to go back to options.")
        action = input("User: ").lower()
        if action == 'exit':
            break

def listen_to_story_action(story_file):
    listen_to_story(story_file)
    while True:
        print("\nType 'exit' to stop listening.")
        action = input("User: ").lower()
        if action == 'exit':
            break

def summarize_story_action(story_content):
    while True:
        print("\nStory Master: Choose a summarization option or type 'exit' to choose other options:")
        print(" 1: Summarize in 200 words")
        print(" 2: Summarize in 500 words")
        print(" 3: Summarize in 800 words" )
        sum_choice = input("\n User: ").lower()
        
        if sum_choice == 'exit':
            break
        elif sum_choice in ['1', '2', '3']:
            max_len = 200 if sum_choice == '1' else 500 if sum_choice == '2' else 800
            print("Story Master: Summarizing the story...\n")
            summarize_story(story_content, max_len)

def answer_question(question, context):
    # Load the trained model and tokenizer
    #model_path = r"C:\Users\Sai_iluru\OneDrive - UMBC\Desktop\learning\707 project\model"
    #model = T5ForConditionalGeneration.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=1024, legacy=False)

    # Tokenize the input text
    inputs = tokenizer.encode("context: " + context + "question: " + question, return_tensors="pt")

    # Generate the output
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)

    # Decode and print the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated answer:\n", decoded_output,"\n")

def chunk_text(text, chunk_size):
    start_time = time.time()
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    end_time = time.time()
    #print(f"\nTime taken to chunk text into {len(chunks)} chunks: {end_time - start_time} seconds")
    return chunks

def summarize_chunk(chunk):
    start_time = time.time()
    summary = extractive_summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    end_time = time.time()
    #print(f"Time taken to summarize a chunk: {end_time - start_time} seconds")
    return summary

def parallel_summarize(chunks):
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {executor.submit(summarize_chunk, chunk): i for i, chunk in enumerate(chunks)}
        results = [None] * len(chunks)
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    combined_summary = ' '.join(results)
    end_time = time.time()
    #print(f"\nTotal time for parallel summarization: {end_time - start_time} seconds")
    return combined_summary

def abstractive_summarize_chunk(chunk):
    summary = abstractive_summarizer(chunk, max_length=150, min_length=75, do_sample=False)[0]['summary_text']
    return summary

def final_abstractive_summary(extractive_summary):
    start_time = time.time()
    chunks = chunk_text(extractive_summary, 1024)
    #print(f"\nNumber of chunks for abstractive summarization: {len(chunks)}")
    
    abstractive_summaries = []
    for chunk in chunks:
        summary = abstractive_summarize_chunk(chunk)
        abstractive_summaries.append(summary)

    final_summary = ' '.join(abstractive_summaries)
    end_time = time.time()
    #print(f"\nTime taken for final abstractive summary: {end_time - start_time} seconds")
    return final_summary

def controlled_summarization(text, target_length, max_iterations=10):
    current_text = text
    for iteration in range(max_iterations):
        #print(f"\nIteration {iteration + 1} for controlled summarization:")
        extractive_summary = parallel_summarize(chunk_text(current_text, 1024))
        abstractive_summary = final_abstractive_summary(extractive_summary)

        word_count = len(abstractive_summary.split())
        if word_count <= target_length:
            #print(f"\nAchieved target length of {target_length} words.")
            return abstractive_summary
        else:
            current_text = abstractive_summary
            #print(f"\nSummary length after iteration {iteration + 1}: {word_count} words.")

    #print("\nMaximum iterations reached. Returning the last summary.")
    return abstractive_summary

def summarize_story(text_from_file,len):
    start_overall = time.time()

    #print("Story Master: Starting controlled summarization process...\n")
    final_summary = controlled_summarization(text_from_file, len)

    end_overall = time.time()
    #print(f"\nTotal time taken for the entire summarization process: {end_overall - start_overall} seconds\n")

    print(f"Final Summary in {len} words:", final_summary)


def main():
    story_dir = 'stories'  # Update this to your story folder path
    
    while True:
        documents, document_names = index_documents(story_dir)

        print("Story Master: Welcome to our interactive Storyteller!")
        print("Story Master: Let's embark on a journey of fascinating tales and adventures.")
        print("Story Master: Type 'list' to see all stories, or enter keywords for suggestions: \n")
        user_input = input("User: ")

        if user_input.strip().lower() == 'list':
            print("Story Master: Here are the available stories: \n")
            # List all stories
            for index, story in enumerate(document_names, start=1):
                print(f"{index}: {story}")

            choice = input("\n User: Choose a story number: \n")

        # Validate choice
            if choice.isdigit() and 1 <= int(choice) <= len(document_names):
                story =  document_names[int(choice) - 1]
                # After suggesting a story
                #selected_story = input("Enter the name of the story you want to explore: ")
                print(story)
                story_content = read_story(story)
            else:
                print("Story Master: The previous entry is wrong !!\n" )

        else:
            # Handle keyword search
            processed_query = preprocess_query(user_input)
            suggested_stories = vsm_search(processed_query, documents, document_names)

            print("Story Master: Suggested Stories Based on Your Keywords:\n")
            for story, similarity in suggested_stories:
                if similarity > 0:  # Display only if similarity is greater than 0
                    print(f"{story} (Similarity: {similarity:.2f})")
                    story_content = read_story(story)

        while True:
            print("\nStory Master: Choose an option:")
            print(" 1: Read Story")
            print(" 2: Listen to Story")
            print(" 3: Summarize Story")
            print(" 4: Ask a Question about Story")
            print(" 5: Choose another Story")
            print(" 0: Exit \n")
            
            user_choice = input("User: Enter your choice (0-5):\n ")
            
            if user_choice == '1':
                read_story_action(story_content)
            elif user_choice == '2':
                mp3files = ['Aladdin_and_Princess_A_Magical_Journey', 'Ali_Baba_and_the_Hidden_Treasure_of_the_Thieves', 'Huckleberrys_Riddling_Adventures', 'TheJourneyofAnya', 'The_Tale_of_King_Alexander']
                story_file = mp3files[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(mp3files) else 'default.mp3'
                listen_to_story_action(story_file)
            elif user_choice == '3':
                summarize_story_action(story_content)
            elif user_choice == '4':
                qa_session(story_content)
            elif user_choice == '5':
                break  # Breaks the inner loop, going back to story selection
            elif user_choice == '0':
                print("Story Master: Thank you for using the Storyteller. Goodbye!\n")
                return  # Exits the program
            else:
                print("Story Master: Invalid choice, please try again.\n")

if __name__ == "__main__":
    main()