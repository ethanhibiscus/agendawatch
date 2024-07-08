import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora, models
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

warnings.filterwarnings('ignore')

# Download NLTK resources once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_texts_from_directory(directory):
    """
    Loads all text documents from the specified directory.
    
    Args:
        directory (str): Path to the directory containing text files.
    
    Returns:
        texts (list): List of extracted text from each file.
        filenames (list): List of filenames corresponding to the extracted texts.
    """
    texts = []
    filenames = []
    for filename in tqdm(os.listdir(directory), desc=f'Loading texts from {directory}'):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                text = f.read()
            texts.append(text)
            filenames.append(filename)
    return texts, filenames

def preprocess(text):
    """
    Preprocesses the text by tokenizing, removing stopwords, and lemmatizing.
    
    Args:
        text (str): The text to preprocess.
    
    Returns:
        tokens (list): List of processed tokens.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

def visualize_token_representations(processed_corpus, output_dir):
    """
    Visualizes token representations using word clouds and bar plots.
    
    Args:
        processed_corpus (list): List of processed documents.
        output_dir (str): Directory where the visualizations will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_tokens = [token for doc in processed_corpus for token in doc]
    word_freq = nltk.FreqDist(all_tokens)
    
    # Word cloud
    wordcloud = WordCloud(width=800, height=400, max_words=100).generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Tokens")
    plt.savefig(os.path.join(output_dir, "word_cloud_tokens.png"))
    plt.close()
    
    # Bar plot
    most_common_words = word_freq.most_common(100)
    words, frequencies = zip(*most_common_words)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=frequencies, y=words)
    plt.title("Top 20 Most Common Words")
    plt.savefig(os.path.join(output_dir, "bar_plot_tokens.png"))
    plt.close()

def train_lda_model(processed_corpus, num_topics=10, passes=15, alpha='symmetric'):
    """
    Trains the LDA model on the processed corpus.
    
    Args:
        processed_corpus (list): List of processed documents.
        num_topics (int): Number of topics to extract.
        passes (int): Number of passes through the corpus during training.
        alpha (str): Parameter that controls the sparsity of the document-topic distribution.
    
    Returns:
        lda_model (LdaModel): Trained LDA model.
        dictionary (Dictionary): Dictionary created from the corpus.
        doc_term_matrix (list): Document-term matrix for the corpus.
    """
    dictionary = corpora.Dictionary(processed_corpus)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_corpus]
    lda_model = models.LdaMulticore(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=passes, alpha=alpha)
    return lda_model, dictionary, doc_term_matrix

def generate_document_topic_matrix(lda_model, doc_term_matrix, filenames):
    """
    Generates the document-topic matrix.
    
    Args:
        lda_model (LdaModel): Trained LDA model.
        doc_term_matrix (list): Document-term matrix for the corpus.
        filenames (list): List of filenames corresponding to the documents.
    
    Returns:
        doc2topic (DataFrame): Document-topic matrix.
    """
    doc2topic = lda_model.get_document_topics(doc_term_matrix, minimum_probability=0)
    doc2topic = pd.DataFrame(list(doc2topic))
    num_topics = lda_model.num_topics
    doc2topic.columns = ['Topic'+str(i+1) for i in range(num_topics)]
    for i in range(len(doc2topic.columns)):
        doc2topic.iloc[:,i] = doc2topic.iloc[:,i].apply(lambda x: x[1])
    doc2topic['Automated_topic_id'] = doc2topic.apply(lambda x: np.argmax(x), axis=1)
    doc2topic['Filename'] = filenames
    return doc2topic

def save_model_and_data(lda_model, dictionary, tfidf_vectorizer, tfidf_matrix, doc2topic, file_prefix, output_dir):
    """
    Saves the LDA model, dictionary, TF-IDF vectorizer, TF-IDF matrix, and document-topic matrix to disk.
    
    Args:
        lda_model (LdaModel): Trained LDA model.
        dictionary (Dictionary): Dictionary created from the corpus.
        tfidf_vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): TF-IDF matrix of the corpus.
        doc2topic (DataFrame): Document-topic matrix.
        file_prefix (str): Prefix for the saved file names.
        output_dir (str): Directory where the models and data will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, f'{file_prefix}_lda_model.pkl'), 'wb') as f:
        pickle.dump(lda_model, f)
    with open(os.path.join(output_dir, f'{file_prefix}_dictionary.pkl'), 'wb') as f:
        pickle.dump(dictionary, f)
    with open(os.path.join(output_dir, f'{file_prefix}_tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(os.path.join(output_dir, f'{file_prefix}_tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    doc2topic.to_csv(os.path.join(output_dir, f'{file_prefix}_doc2topic.csv'), index=False)

def plot_coherence_scores(coherence_scores_dict, output_dir):
    """
    Plots coherence scores for different numbers of topics.
    
    Args:
        coherence_scores_dict (dict): Dictionary where keys are number of topics and values are coherence scores.
        output_dir (str): Directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 5))
    for lambda_value, coherence_scores in coherence_scores_dict.items():
        plt.plot(list(coherence_scores.keys()), list(coherence_scores.values()), marker='o', label=f'Lambda={lambda_value}')
    plt.title('Coherence Scores by Number of Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'coherence_scores.png'))
    plt.close()

def calculate_coherence(lda_model, processed_corpus, dictionary):
    """
    Calculates the coherence score for the given LDA model.
    
    Args:
        lda_model (LdaModel): Trained LDA model.
        processed_corpus (list): List of processed documents.
        dictionary (Dictionary): Dictionary created from the corpus.
    
    Returns:
        coherence (float): Coherence score.
    """
    coherence_model = models.CoherenceModel(model=lda_model, texts=processed_corpus, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_coherence()
    return coherence

def visualize_lda_model(lda_model, dictionary, doc_term_matrix, output_dir, num_topics):
    """
    Visualizes the LDA model using pyLDAvis and saves it as an HTML file.
    
    Args:
        lda_model (LdaModel): Trained LDA model.
        dictionary (Dictionary): Dictionary created from the corpus.
        doc_term_matrix (list): Document-term matrix for the corpus.
        output_dir (str): Directory where the visualization will be saved.
        num_topics (int): Number of topics for the current LDA model.
    """
    vis_data = gensimvis.prepare(lda_model, doc_term_matrix, dictionary)
    pyLDAvis.save_html(vis_data, os.path.join(output_dir, f'lda_visualization_{num_topics}_topics.html'))

def main_training():
    text_dir = '../data/text_files'
    output_dir = 'train_output'
    
    # Load data from the text directory
    print("Loading text files...")
    corpus, filenames = load_texts_from_directory(text_dir)
    
    # Preprocess data
    print("Preprocessing documents...")
    processed_corpus = [preprocess(doc) for doc in tqdm(corpus, desc='Preprocessing')]
    
    # Visualize token representations
    print("Visualizing token representations...")
    visualize_token_representations(processed_corpus, 'train_results_all')
    
    # Hyperparameters
    passes = 20
    coherence_scores_dict = {}
    lambda_values = 0.3
    #lambda_values = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75 ,0.9]
    for lambda_value in lambda_values:
        coherence_scores_dict[lambda_value] = {}

    # Train LDA model for different numbers of topics and lambda values
    for num_topics in range(10, 12):
        for lambda_value in lambda_values:
            print(f"Training LDA model with {num_topics} topics and lambda={lambda_value}...")
            lda_model, dictionary, doc_term_matrix = train_lda_model(processed_corpus, num_topics, passes, alpha=lambda_value)
            coherence = calculate_coherence(lda_model, processed_corpus, dictionary)
            coherence_scores_dict[lambda_value][num_topics] = coherence
            
            output_subdir = os.path.join(output_dir, f'train_results_{num_topics}topics')
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            visualize_lda_model(lda_model, dictionary, doc_term_matrix, output_subdir, num_topics)
            
            flattened_corpus = [' '.join(doc) for doc in processed_corpus]
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(flattened_corpus)
            
            doc2topic = generate_document_topic_matrix(lda_model, doc_term_matrix, filenames)
            
            save_model_and_data(lda_model, dictionary, tfidf_vectorizer, tfidf_matrix, doc2topic, f'agenda_watch_{num_topics}_topics', output_subdir)

    # Plot coherence scores
    print("Plotting coherence scores...")
    plot_coherence_scores(coherence_scores_dict, 'train_results_all')

if __name__ == "__main__":
    main_training()
