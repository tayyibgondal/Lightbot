import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Tokenizes the input sentence using NLTK library.

    Parameters:
    sentence (str): Input sentence to be tokenized.

    Returns:
    list: List of tokens extracted from the input sentence.
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Returns the root form of the input word by stemming.

    Parameters:
    word (str): Input word to be stemmed.

    Returns:
    str: Root form of the input word after stemming.
    
    Examples:
    words = ["organize", "organizes", "organizing"]
    stemmed_words = [stem(w) for w in words]
    Result: ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    Generates a bag of words array representing the presence of known words in the sentence.

    Parameters:
    tokenized_sentence (list): List of tokens from the input sentence.
    words (list): List of known words.

    Returns:
    numpy.ndarray: Bag of words array representing the presence of known words in the sentence.
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
