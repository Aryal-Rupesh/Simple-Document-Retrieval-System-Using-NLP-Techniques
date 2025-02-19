# import necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

# function to load the dataset
def load_dataset(file_path):
    with open(file_path, "r") as file:
        text_list = file.readlines() # loads each line from the file as an element in a list
    text_list = [text.strip() for text in text_list] # strip(), to remove whitespace characters from both the left and right sides of a string
    return text_list

# function for preprocessing
def pre_processing(text_list):
    preprocessed_list = []
    for element in text_list:
        # lowercasing and tokenize
        tokenized_element = word_tokenize(element.lower())
        # remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_element = [word for word in tokenized_element if word not in stop_words]
        # lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_element = [lemmatizer.lemmatize(word) for word in filtered_element]
        preprocessed_list.append(lemmatized_element)
    return preprocessed_list

# Bag of Words vectorizer function
def BoW_vectorizer(vocab, preprocessed_list):
    preprocessed_list = [" ".join(element) for element in preprocessed_list] # joins the tokens of each inner list into a single string because vectorizer expects a list of sentences as input # I took assistnace from ChatGPT for the syntax
    vectorizer = CountVectorizer(lowercase=False, vocabulary=vocab)
    vectorized_list = vectorizer.fit_transform(preprocessed_list)
    vectorized_list = vectorized_list.toarray().tolist()
    return vectorized_list

# TF-IDF vectorizer function
def Tfidf_vectorizer(vocab, preprocessed_list):
    preprocessed_list = [" ".join(element) for element in preprocessed_list] # joins the tokens of each inner list into a single string because vectorizer expects a list of sentences as input # I took assistnace from ChatGPT for the syntax
    vectorizer = TfidfVectorizer(lowercase=False, vocabulary=vocab)
    vectorized_list = vectorizer.fit_transform(preprocessed_list)
    vectorized_list = vectorized_list.toarray().tolist()
    return vectorized_list

# cosine similarity computation function
def cosine_similarity(vectorized_query, vectorized_documents):
    best_similarity = -1
    best_index = 0
    vectorized_query = np.array(vectorized_query)
    for i in range(len(vectorized_documents)): # iterate over eah vector within the vectorized documents
        dot_product = np.dot(vectorized_query, np.array(vectorized_documents[i])) # compute the dot product between the query vector and the iᵗʰ vector within the vectorized documents
        magnitude1 = np.linalg.norm(vectorized_query) # compute the magnitude of the query vector # I took assistnace from ChatGPT for the syntax
        magnitude2 = np.linalg.norm(np.array(vectorized_documents[i])) # compute the magnitude of the iᵗʰ vector within the vectorized documents # I took assistnace from ChatGPT for the syntax
        similarity = dot_product / (magnitude1 * magnitude2) # compute the cosine similarity using the dot product and the magnitudes of the vectors
        if similarity > best_similarity:
            best_similarity = similarity 
            best_index = i
    return best_similarity, best_index

# function to compute similarity using Euclidean distance 
def euclidean_similarity(vectorized_query, vectorized_documents):
    best_distance = float("inf")
    best_index = 0
    vectorized_query = np.array(vectorized_query)
    for i in range(len(vectorized_documents)): # iterate over eah vector within the vectorized documents
        distance = np.linalg.norm(vectorized_query - np.array(vectorized_documents[i])) # compute the Euclidean distance between the query vector and the iᵗʰ vector within the vectorized documents
        if distance < best_distance:
            best_distance = distance
            best_index = i
    return best_distance, best_index

# main function
def main():
    documents_list = load_dataset("documents.txt") # load documents.txt as a list
    queries_list = load_dataset("queries.txt") # load queries.txt as a list
    
    preprocessed_documents = pre_processing(documents_list) # preprocess documents
    preprocessed_queries = pre_processing(queries_list) # preprocess queries
    
    vocab = list(set([element for inner_list in preprocessed_documents + preprocessed_queries for element in inner_list])) # create vocabulary for vectorization
    
    BoW_vectorized_documents = BoW_vectorizer(vocab, preprocessed_documents) # vectorize preprocessed documents using Bag of Words 
    BoW_vectorized_queries = BoW_vectorizer(vocab, preprocessed_queries) # vectorize preprocessed queries using Bag of Words 
    
    Tfidf_vectorized_documents = Tfidf_vectorizer(vocab, preprocessed_documents) # vectorize preprocessed documents using TF-IDF
    Tfidf_vectorized_queries = Tfidf_vectorizer(vocab, preprocessed_queries) # vectorize preprocessed queries using TF-IDF

    option = int(input("Choose the method you want to use for similarity computation:\nInput 1 for Cosine Similarity\nInput 2 for Euclidean Distance\n")) # choose the similarity computation method that we want to use
    print("-"*200) # for better readability
    match option: # match statement to switch between our chosen methods of similarity computation # ensure that the Python version is 3.10 or greater to use the match statement
        case 1: # cosine similarity
            for i in range(len(queries_list)): # iterate over each query within the list of queries
                print(f"Query: {queries_list[i]}")
                similarity_BoW, index_BoW = cosine_similarity(BoW_vectorized_queries[i], BoW_vectorized_documents) # compute the cosine similarity of BoW vectors
                similarity_Tfidf, index_Tfidf = cosine_similarity(Tfidf_vectorized_queries[i], Tfidf_vectorized_documents) # compute the cosine similarity of TF-IDF vectors
                print(f"Top Matching Document with BOW: {documents_list[index_BoW]}")
                print(f"Top Matching Document with TFIDF: {documents_list[index_Tfidf]}")
                print(f"Cosine Similarity Score with BOW: {similarity_BoW}")
                print(f"Cosine Similarity Score with TFIDF: {similarity_Tfidf}")
                print("-"*200) # for better readability
        case 2: # euclidean distance 
            for i in range(len(queries_list)): # iterate over each query within the list of queries
                print(f"Query: {queries_list[i]}")
                similarity_BoW, index_BoW = euclidean_similarity(BoW_vectorized_queries[i], BoW_vectorized_documents) # compute the similarity of BoW vectors using the Euclidean distance
                similarity_Tfidf, index_Tfidf = euclidean_similarity(Tfidf_vectorized_queries[i], Tfidf_vectorized_documents) # compute the similarity of TF-IDF vectors using the Euclidean distance
                print(f"Top Matching Document with BOW: {documents_list[index_BoW]}")
                print(f"Top Matching Document with TFIDF: {documents_list[index_Tfidf]}")
                print(f"Euclidean Similarity Score with BOW: {similarity_BoW}")
                print(f"Euclidean Similarity Score with TFIDF: {similarity_Tfidf}")
                print("-"*200) # for better readability
        case _:
            print("Enter valid input!") 
main()