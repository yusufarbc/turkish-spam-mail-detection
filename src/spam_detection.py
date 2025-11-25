"""
Turkish Spam Email Detection using K-Nearest Neighbors

This module implements a spam detection system for Turkish emails using
the K-Nearest Neighbors (KNN) algorithm. It includes text preprocessing,
feature extraction, model training, and visualization capabilities.

Author: yusufarbc
Dataset: Turkish Spam V01 (825 emails)
"""

import string
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def get_count(text: str) -> Dict[str, int]:
    """
    Count word frequencies in a given text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict[str, int]: Dictionary mapping words to their frequencies
    """
    word_counts = {}
    for word in text.split():
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    return word_counts

def euclidean_difference(test_word_counts: Dict[str, int], 
                         training_word_counts: Dict[str, int]) -> float:
    """
    Calculate Euclidean distance between two word count dictionaries.
    
    Args:
        test_word_counts (Dict[str, int]): Word frequencies from test email
        training_word_counts (Dict[str, int]): Word frequencies from training email
        
    Returns:
        float: Euclidean distance between the two word count vectors
    """
    total = 0
    # Create a copy to avoid modifying original dictionary
    training_copy = training_word_counts.copy()
    
    for word in test_word_counts:
        if word in training_copy:
            total += (test_word_counts[word] - training_copy[word]) ** 2
            del training_copy[word]
        else:
            total += test_word_counts[word] ** 2

    for word in training_copy:
        total += training_copy[word] ** 2
    
    return total ** 0.5

def get_class(selected_k_values: List[Tuple[str, float]]) -> str:
    """
    Determine the class (spam or ham) based on K nearest neighbors.
    
    Args:
        selected_k_values (List[Tuple[str, float]]): List of (label, distance) tuples
        
    Returns:
        str: Predicted class ('spam' or 'ham')
    """
    spam_count = 0
    ham_count = 0
    
    for value in selected_k_values:
        if value[0] == "spam":
            spam_count += 1
        else:
            ham_count += 1
    
    return "spam" if spam_count > ham_count else "ham"
    
def knn_classifier(training_data: np.ndarray, 
                   training_labels: np.ndarray, 
                   test_data: np.ndarray, 
                   k: int, 
                   tsize: int) -> List[str]:
    """
    Classify emails using K-Nearest Neighbors algorithm.
    
    Args:
        training_data (np.ndarray): Array of training email texts
        training_labels (np.ndarray): Array of training labels ('spam' or 'ham')
        test_data (np.ndarray): Array of test email texts
        k (int): Number of nearest neighbors to consider
        tsize (int): Number of test samples to process
        
    Returns:
        List[str]: List of predicted labels for test data
    """
    print("Running KNN Classifier...")
    
    result = []
    
    # Precompute word counts for all training emails
    training_word_counts = []
    for training_text in training_data:
        training_word_counts.append(get_count(training_text))
    
    # Process each test email
    for test_text in test_data:
        similarity = []  # List of (label, euclidean_distance) tuples
        test_word_counts = get_count(test_text)
        
        # Calculate euclidean distance to each training email
        for index in range(len(training_data)):
            euclidean_diff = euclidean_difference(
                test_word_counts, 
                training_word_counts[index]
            )
            similarity.append([training_labels[index], euclidean_diff])
        
        # Sort by distance (ascending order)
        similarity = sorted(similarity, key=lambda i: i[1])
        
        # Select K nearest neighbors
        selected_k_values = similarity[:k]
        
        # Predict the class based on majority vote
        result.append(get_class(selected_k_values))
    
    return result

def load_data(filepath: str = "data/trspam.csv") -> np.ndarray:
    """
    Load email data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file containing email data
        
    Returns:
        np.ndarray: Array of [text, label] pairs
    """
    print("Loading data...")
    data = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            label = str(row[-1])
            del row[-1]
            text = ''.join(row)
            data.append([text, label])
    
    # Remove header and last empty row
    del data[0]
    del data[-1]
    
    return np.array(data)


# Load the dataset
data = load_data()
print(f"Total emails loaded: {len(data)}")


def preprocess_data(data: np.ndarray, stopwords_file: str = "data/stopwords-tr.txt") -> np.ndarray:
    """
    Preprocess email text data: remove punctuation, convert to lowercase, filter stopwords.
    
    Args:
        data (np.ndarray): Array of [text, label] pairs
        stopwords_file (str): Path to Turkish stopwords file
        
    Returns:
        np.ndarray: Preprocessed data
    """
    print("Preprocessing data...")
    
    punc = string.punctuation
    
    # Load Turkish stopwords
    with open(stopwords_file, "r", encoding="utf-8") as f:
        stopwords = f.read().splitlines()
    
    for record in data:
        # Remove punctuation
        text = record[0]
        for char in punc:
            text = text.replace(char, "")
        
        # Split, lowercase, and filter stopwords
        words = text.split()
        filtered_words = [
            word.lower() 
            for word in words 
            if word not in stopwords
        ]
        
        record[0] = " ".join(filtered_words)
    
    return data


# Preprocess the data
data = preprocess_data(data)


def plot_word_count_histogram(data: np.ndarray, output_dir: str = "outputs/plots") -> None:
    """
    Plot histograms of word counts for spam and ham emails.
    
    Args:
        data (np.ndarray): Array of [text, label] pairs
        output_dir (str): Directory to save plots
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    count_ham_list = []
    count_spam_list = []
    
    for record in data:
        word_count = len(record[0].split())
        if record[1] == "ham":
            count_ham_list.append(word_count)
        else:
            count_spam_list.append(word_count)
    
    # Plot ham emails histogram
    plt.figure(figsize=(10, 5))
    plt.title("Histogram of Ham E-mails' Word Counts", fontsize=14, fontweight='bold')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.hist(count_ham_list, bins=40, color='green', alpha=0.7, edgecolor='black')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f'{output_dir}/hist_ham.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot spam emails histogram
    plt.figure(figsize=(10, 5))
    plt.title("Histogram of Spam E-mails' Word Counts", fontsize=14, fontweight='bold')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.hist(count_spam_list, bins=40, color='red', alpha=0.7, edgecolor='black')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f'{output_dir}/hist_spam.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Ham emails - Mean word count: {np.mean(count_ham_list):.2f}")
    print(f"Spam emails - Mean word count: {np.mean(count_spam_list):.2f}")


# Plot word count histograms
plot_word_count_histogram(data)

def calculate_word_frequencies(data: np.ndarray, 
                               min_count: int = 100, 
                               max_count: int = 150) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Calculate word frequencies for spam and ham emails.
    
    Args:
        data (np.ndarray): Array of [text, label] pairs
        min_count (int): Minimum word frequency to include
        max_count (int): Maximum word frequency to include
        
    Returns:
        Tuple: (ham_words, ham_counts, spam_words, spam_counts)
    """
    frequency_ham_word_list = []
    frequency_ham_count_list = []
    frequency_spam_word_list = []
    frequency_spam_count_list = []

    for record in data:
        words = record[0].split()
        if record[1] == "ham":
            for word in words:
                if word in frequency_ham_word_list:
                    index = frequency_ham_word_list.index(word)
                    frequency_ham_count_list[index] += 1
                else:
                    frequency_ham_word_list.append(word)
                    frequency_ham_count_list.append(1)
        else:
            for word in words:
                if word in frequency_spam_word_list:
                    index = frequency_spam_word_list.index(word)
                    frequency_spam_count_list[index] += 1
                else:
                    frequency_spam_word_list.append(word)
                    frequency_spam_count_list.append(1)

    # Filter by frequency range
    ham_filtered = [
        (word, count) 
        for word, count in zip(frequency_ham_word_list, frequency_ham_count_list)
        if min_count <= count <= max_count
    ]
    spam_filtered = [
        (word, count) 
        for word, count in zip(frequency_spam_word_list, frequency_spam_count_list)
        if min_count <= count <= max_count
    ]
    
    if ham_filtered:
        ham_words, ham_counts = zip(*ham_filtered)
    else:
        ham_words, ham_counts = [], []
    
    if spam_filtered:
        spam_words, spam_counts = zip(*spam_filtered)
    else:
        spam_words, spam_counts = [], []
    
    return list(ham_words), list(ham_counts), list(spam_words), list(spam_counts)


def plot_word_frequencies(ham_words: List[str], ham_counts: List[int],
                         spam_words: List[str], spam_counts: List[int],
                         output_dir: str = "outputs/plots") -> None:
    """
    Plot bar charts of most frequent words in spam and ham emails.
    
    Args:
        ham_words (List[str]): List of words from ham emails
        ham_counts (List[int]): Frequency counts for ham words
        spam_words (List[str]): List of words from spam emails
        spam_counts (List[int]): Frequency counts for spam words
        output_dir (str): Directory to save plots
    """
    # Plot ham words
    if ham_words:
        plt.figure(figsize=(20, 5))
        plt.title("Most Used Words in Ham E-mails", fontsize=16, fontweight='bold')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.bar(ham_words, ham_counts, color='green', alpha=0.7, edgecolor='black')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/plot_ham.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Ham words (100-150 frequency): {ham_words}")
    
    # Plot spam words
    if spam_words:
        plt.figure(figsize=(20, 5))
        plt.title("Most Used Words in Spam E-mails", fontsize=16, fontweight='bold')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.bar(spam_words, spam_counts, color='red', alpha=0.7, edgecolor='black')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/plot_spam.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Spam words (100-150 frequency): {spam_words}")


# Calculate and plot word frequencies
ham_words, ham_counts, spam_words, spam_counts = calculate_word_frequencies(data)
print(f"Number of ham words in range: {len(ham_words)}")
print(f"Number of spam words in range: {len(spam_words)}")
plot_word_frequencies(ham_words, ham_counts, spam_words, spam_counts)


# Split data into training and testing sets
print("\nSplitting data...")
features = data[:, 0]  # Email text bodies
labels = data[:, 1]    # Labels ('spam' or 'ham')

training_data, test_data, training_labels, test_labels = train_test_split(
    features, labels, test_size=0.30, random_state=42
)

print(f"Training set size: {len(training_data)}")
print(f"Test set size: {len(test_data)}")

# Set KNN parameters
K = 24
tsize = len(test_data)

print(f"\nK value: {K}")
print(f"Samples to test: {tsize}")

# Train and test the model
print("\nModel Training...")
result = knn_classifier(training_data, training_labels, test_data[:tsize], K, tsize)

print("\nModel Testing...")
accuracy = accuracy_score(test_labels[:tsize], result)

# Display results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Training data size\t: {len(training_data)}")
print(f"Test data size\t\t: {len(test_data)}")
print(f"K value\t\t\t: {K}")
print(f"Samples tested\t\t: {tsize}")
print(f"Accuracy\t\t: {accuracy * 100:.2f}%")
print(f"Number correct\t\t: {int(accuracy * tsize)}")
print(f"Number wrong\t\t: {int((1 - accuracy) * tsize)}")
print("="*60)

# Classification Report
print("\nClassification Report:")
print(classification_report(test_labels[:tsize], result, target_names=['ham', 'spam']))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(test_labels[:tsize], result, labels=['ham', 'spam'])
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization files saved:")
print("  - outputs/plots/hist_ham.png")
print("  - outputs/plots/hist_spam.png")
print("  - outputs/plots/plot_ham.png")
print("  - outputs/plots/plot_spam.png")
print("  - outputs/plots/confusion_matrix.png")

