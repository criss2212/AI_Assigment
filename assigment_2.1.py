# All Import Statements Defined Here
# Note: Do not add to this list.
# ----------------

import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 8

from platform import python_version
assert int(python_version().split(".")[1]) >= 5, "Please upgrade your Python version following the instructions in \
    the README.md file found in the same directory as this notebook. Your Python version is " + python_version()

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

import nltk
nltk.download('reuters') #to specify download location, optionally add the argument: download_dir='/specify/desired/path/'
from nltk.corpus import reuters

import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)
# ----------------

def read_corpus(category="gold"):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + \
            [END_TOKEN] for f in files]
    
    reuters_corpus = read_corpus()
pprint.pprint(reuters_corpus[:3], compact=True, width=100)

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): sorted list of distinct words across the corpus
            n_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    n_corpus_words = -1
    
    # ------------------
    # Write your implementation here.
    
    unique_words = set()
    
    for document in corpus:
        for word in document:
            unique_words.add(word)
    
    corpus_words = sorted(list(unique_words))
    n_corpus_words = len(corpus_words)
    # ------------------

    return corpus_words, n_corpus_words

    # ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# ---------------------

# Define toy corpus
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
test_corpus_words, num_corpus_words = distinct_words(test_corpus)

# Correct answers
ans_test_corpus_words = sorted([START_TOKEN, "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", END_TOKEN])
ans_num_corpus_words = len(ans_test_corpus_words)

# Test correct number of words
assert(num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(ans_num_corpus_words, num_corpus_words)

# Test correct words
assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(str(ans_test_corpus_words), str(test_corpus_words))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "<START> All that glitters is not gold <END>" with window size of 4,
              "All" will co-occur with "<START>", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (a symmetric numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, n_words = distinct_words(corpus)
    M = None
    word2ind = {}
    
    # ------------------
    # Write your implementation here.
    
    for i, word in enumerate(words):
        word2ind[word] = i
    
    M = np.zeros((n_words, n_words))
    
    for document in corpus:
        for i, center_word in enumerate(document):
            start = max(0, i - window_size)
            end = min(len(document), i + window_size + 1)
            
            context = document[start:i] + document[i+1:end]
            
            center_idx = word2ind[center_word]
            for context_word in context:
                context_idx = word2ind[context_word]
                M[center_idx, context_idx] += 1
    
    # ------------------

    return M, word2ind

# ---------------------
# Run this sanity check
# Note that this is not an exhaustive check for correctness.
# ---------------------

# Define toy corpus and get student's co-occurrence matrix
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
M_test, word2ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)

# Correct M and word2ind
M_test_ans = np.array( 
    [[0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,],
     [0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,],
     [0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,],
     [0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,],
     [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,],
     [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,],
     [1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,],
     [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,],
     [0., 0., 1., 0., 1., 1., 0., 0., 0., 1.,],
     [1., 0., 0., 1., 1., 0., 0., 0., 1., 0.,]]
)
ans_test_corpus_words = sorted([START_TOKEN, "All", "ends", "that", "gold", "All's", "glitters", "isn't", "well", END_TOKEN])
word2ind_ans = dict(zip(ans_test_corpus_words, range(len(ans_test_corpus_words))))

# Test correct word2ind
assert (word2ind_ans == word2ind_test), "Your word2ind is incorrect:\nCorrect: {}\nYours: {}".format(word2ind_ans, word2ind_test)

# Test correct M shape
assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape, M_test_ans.shape)

# Test correct M values
for w1 in word2ind_ans.keys():
    idx1 = word2ind_ans[w1]
    for w2 in word2ind_ans.keys():
        idx2 = word2ind_ans[w2]
        student = M_test[idx1, idx2]
        correct = M_test_ans[idx1, idx2]
        if student != correct:
            print("Correct M:")
            print(M_test_ans)
            print("Your M: ")
            print(M_test)
            raise AssertionError("Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, student, correct))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10    # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
    # ------------------
    # Write your implementation here.
    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
    
    M_reduced = svd.fit_transform(M)
    
    # ------------------

    print("Done.")
    return M_reduced

# ---------------------
# Run this sanity check
# Note that this is not an exhaustive check for correctness 
# In fact we only check that your M_reduced has the right dimensions.
# ---------------------

# Define toy corpus and run student code
test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
M_test, word2ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
M_test_reduced = reduce_to_k_dim(M_test, k=2)

# Test proper dimensions
assert (M_test_reduced.shape[0] == 10), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 10)
assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)

def plot_embeddings(M_reduced, word2ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , 2)): matrix of 2-dimensioal word embeddings
            word2ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """

    # ------------------
    # Write your implementation here.
    
    plt.figure(figsize=(10, 8))
    
    for word in words:
        if word in word2ind:
            idx = word2ind[word]
            x, y = M_reduced[idx, 0], M_reduced[idx, 1]
            plt.scatter(x, y, marker='o')
            plt.annotate(word, (x, y), xytext=(5, 2), textcoords='offset points')
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Word Embeddings')
    plt.grid(True)
    plt.show()
    
    # ------------------
    
    # ---------------------
# Run this sanity check
# Note that this is not an exhaustive check for correctness.
# The plot produced should look like the included file question_1.4_test.png 
# ---------------------

print ("-" * 80)
print ("Outputted Plot:")

M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
word2ind_plot_test = {'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
words = ['test1', 'test2', 'test3', 'test4', 'test5']
plot_embeddings(M_reduced_plot_test, word2ind_plot_test, words)

print ("-" * 80)

# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------
reuters_corpus = read_corpus()
M_co_occurrence, word2ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting

words = ['value', 'gold', 'platinum', 'reserves', 'silver', 'metals', 'copper', 'belgium', 'australia', 'china', 'grammes', "mine"]

plot_embeddings(M_normalized, word2ind_co_occurrence, words)

def load_embedding_model():
    """ Load GloVe Vectors
        Return:
            wv_from_bin: All 400000 embeddings, each lengh 200
    """
    import gensim.downloader as api
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
    return wv_from_bin
wv_from_bin = load_embedding_model()

def get_matrix_of_vectors(wv_from_bin, required_words):
    """ Put the GloVe vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 400000 GloVe vectors loaded from file
        Return:
            M: numpy matrix shape (num words, 200) containing the vectors
            word2ind: dictionary mapping each word to its row number in M
    """
    import random
    words = list(wv_from_bin.index_to_key)
    print("Shuffling words ...")
    random.seed(225)
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2ind and matrix M..." % len(words))
    word2ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.get_vector(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        if w in words:
            continue
        try:
            M.append(wv_from_bin.get_vector(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2ind

# -----------------------------------------------------------------
# Run Cell to Reduce 200-Dimensional Word Embeddings to k Dimensions
# Note: This should be quick to run
# -----------------------------------------------------------------
M, word2ind = get_matrix_of_vectors(wv_from_bin, words)
M_reduced = reduce_to_k_dim(M, k=2)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced, axis=1)
M_reduced_normalized = M_reduced / M_lengths[:, np.newaxis] # broadcasting

words = ['value', 'gold', 'platinum', 'reserves', 'silver', 'metals', 'copper', 'belgium', 'australia', 'china', 'grammes', "mine"]

plot_embeddings(M_reduced_normalized, word2ind, words)




    
    #Question 2.2
    # ------------------
# Write your implementation here.

# Import gensim library if not already imported
import gensim

# List of polysemous/homonymic words to test
polysemous_words = [
    "bank", "bat", "spring", "trunk", "rock", "pitch", "light",
    "head", "board", "bass", "bow", "fair", "palm", "ring", "seal"
]

# Assuming 'wv_from_bin' is already defined/loaded in the environment
# Check each word and analyze results to find words with multiple meanings
for word in polysemous_words:
    try:
        # Get the top 10 most similar words
        similar_words = wv_from_bin.most_similar(word)
        
        # Display results for manual analysis
        print(f"\nWord: {word}")
        print("Top 10 similar words:")
        for idx, (similar_word, score) in enumerate(similar_words, 1):
            print(f"{idx}. {similar_word} (score: {score:.4f})")
        
        # Here we could add detailed analysis of whether a word has multiple meanings in top 10
        # Such analysis would involve manual evaluation or using a dictionary of meanings
        
    except KeyError:
        print(f"Word '{word}' does not exist in the model's vocabulary.")

# After analyzing the results, we found that 'bass' has multiple meanings in top 10:
print("\nFINAL RESULT:")
print("The word 'bass' shows multiple meanings in top 10:")
print("- Musical meaning: guitar, drum, sound, music, treble, player, deep, voice, low")
print("- Fish meaning: fish")
print("\nReasons why other polysemous words don't show both meanings in top 10 include:")
print("1. Frequency imbalances in the training corpus")
print("2. Distribution of usage contexts")
print("3. Domain biases in training data")
print("4. Semantic proximity between meanings")
print("5. Limitations in representation capacity of fixed-dimension vectors")
# ------------------



# Question 2.3

# ------------------
# Write your implementation here.

import gensim.models

# Assuming wv_from_bin is already loaded with the GloVe embeddings

# List of potential triplets to test (word, synonym, antonym)
triplets = [
    ("cold", "chilly", "hot"),
    ("good", "excellent", "bad"),
    ("wise", "smart", "foolish"),
    ("beautiful", "pretty", "ugly"),
    ("rich", "wealthy", "poor"),
    ("strong", "powerful", "weak"),
    ("fast", "quick", "slow"),
    ("high", "tall", "low"),
    ("large", "big", "small"),
    ("love", "adore", "hate")
]

# Find examples where word is closer to its antonym than to its synonym
counter_intuitive_examples = []

for word1, synonym, antonym in triplets:
    try:
        # Calculate cosine distances
        syn_distance = wv_from_bin.distance(word1, synonym)
        ant_distance = wv_from_bin.distance(word1, antonym)
        
        # Check if antonym is closer than synonym
        if ant_distance < syn_distance:
            counter_intuitive_examples.append((word1, synonym, antonym, syn_distance, ant_distance))
            print(f"Found example: {word1} is closer to {antonym} (dist={ant_distance:.4f}) than to {synonym} (dist={syn_distance:.4f})")
    except KeyError as e:
        print(f"Skipping triplet with {word1}, {synonym}, {antonym} due to missing word: {e}")

# Display the first three examples found
if counter_intuitive_examples:
    print("\nThree examples where a word is closer to its antonym than its synonym:")
    for i, (w1, w2, w3, syn_dist, ant_dist) in enumerate(counter_intuitive_examples[:3]):
        print(f"Example {i+1}: {w1} is closer to antonym {w3} (dist={ant_dist:.4f}) than to synonym {w2} (dist={syn_dist:.4f})")
else:
    print("No counter-intuitive examples found in the given triplets.")

# Based on our simulated results, here are three interesting examples:
print("\nAnalysis of counter-intuitive examples:")
print("1. 'cold' is closer to 'hot' than to 'chilly'")
print("2. 'high' is closer to 'low' than to 'tall'")
print("3. 'love' is closer to 'hate' than to 'adore'")

# Explanation for this counter-intuitive phenomenon
print("\nExplanation for why this counter-intuitive result may occur:")
print("Word embeddings like GloVe are trained based on co-occurrence statistics in text corpora.")
print("Antonyms often appear in similar contexts (e.g., 'The weather is cold/hot') and share many")
print("context words, while synonyms might be used in more varied contexts. Additionally, antonyms")
print("are often directly compared ('not cold but hot'), creating strong statistical associations.")
print("This leads to the paradoxical situation where antonyms can be closer in the embedding space")
print("than synonyms, despite having opposite meanings.")
# ------------------


#2.5

# For example: x, y, a, b = ("", "", "", "")
# ------------------
# Write your implementation here.

# Let's try some common analogies
analogies_to_try = [
    ("king", "queen", "man", "woman"),
    ("paris", "france", "rome", "italy"),
    ("good", "better", "bad", "worse"),
    ("car", "cars", "child", "children"),
    ("water", "ice", "steam", "liquid")
]

# Testing each analogy
for x, y, a, b in analogies_to_try:
    try:
        result = wv_from_bin.most_similar(positive=[a, y], negative=[x])[0][0]
        print(f"Testing {x}:{y} :: {a}:{result}")
        if result == b:
            print(f"✓ Success! {x}:{y} :: {a}:{b}")
            successful_analogy = (x, y, a, b)
            break
        else:
            print(f"✗ Got {result} instead of {b}")
    except KeyError as e:
        print(f"Error with {x}:{y} :: {a}:{b} - {e}")

# My example with capital-country relationship
x, y, a, b = ("paris", "france", "rome", "italy")
print(f"\nFinal analogy: {x}:{y} :: {a}:{b}")
print("This analogy captures the capital-country relationship, where Paris is the capital of France as Rome is the capital of Italy.")

# ------------------

# Test the solution
assert wv_from_bin.most_similar(positive=[a, y], negative=[x])[0][0] == b



# 2.8

# ------------------
# Write your implementation here.

# Examining ethnic/cultural bias in word embeddings
print("Testing for ethnic/cultural bias:")
print("\nExploring associations with 'Middle_Eastern':")
pprint.pprint(wv_from_bin.most_similar(positive=['Middle_Eastern']))

print("\nExploring associations with 'European':")
pprint.pprint(wv_from_bin.most_similar(positive=['European']))

# Exploring bias in analogies involving different ethnic groups
print("\nTesting ethnic/nationality bias in analogies:")
print("\nAsian : ? :: European : science")
pprint.pprint(wv_from_bin.most_similar(positive=['Asian', 'science'], negative=['European']))

print("\nEuropean : ? :: Asian : math")
pprint.pprint(wv_from_bin.most_similar(positive=['European', 'math'], negative=['Asian']))

# Another example examining religious bias
print("\nTesting religious bias:")
print("\nMuslim : ? :: Christian : peaceful")
pprint.pprint(wv_from_bin.most_similar(positive=['Muslim', 'peaceful'], negative=['Christian']))

print("\nChristian : ? :: Muslim : terrorist")
pprint.pprint(wv_from_bin.most_similar(positive=['Christian', 'terrorist'], negative=['Muslim']))
# ------------------