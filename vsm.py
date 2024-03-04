from collections import defaultdict
from math import log, sqrt
import re
import sys
from tqdm import tqdm
import csv
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


""" 
This program implements a Vector Space Model IR system. The object is the retrieval of cooking recipes, 
by searching the RecipeNLG dataset (https://recipenlg.cs.put.poznan.pl/). 
Recipes are referred to as documents in the following.

Word stemming is adopted, to decrease the impact of vocabulary dimension.

In the algorithm, representation of all documents as vectors of size equal to vocabulary length is avoided.
Instead, data necessary for Cosine Similarity computation is stored in an inverse Tf-Idf index, which only 
stores non null weights. The idea is that, for calculating the scalar product, you only need to know the 
values of non null terms. 

With more than 2 million recipes, the dataset is too large for the scope of the project. A reduced version,
with 100000 recipes, is included instead (mini_dataset.csv).
Moreover, an option is given to only import the first N recipes. 

To avoid re-calculating the index each time (and importing the dataset), you can save the tf-idf index to file, 
together with the list of L2 norms of all documents in the index, and the list of recipe titles.

The provided index was calculated with all 100000 recipes in the reduced dataset.

Here is the structure of the algorithm:
1. Setup:
    -Import dataset as list of lists of words (one list of words per document) (and dictionary of {DocId: Title} pairs).
    -Create (or import existing) inverse indices: idf index and tf-idf index.
2. Query: 
    -Iterate over terms in query to compute Cosine Similarity score of all documents using Tf-Idf index.
    -Return top K scoring documents (K is set to 25).
    -Repeat with new query, go to 3 or end.
3: Relevance Feedback:
    -Calculate the centroids of relevant and non-relevant documents.
    -Create new query with Rocchio formula.
    -Go to step 2.
"""

def import_dataset(max_size):
    """
    This function imports the first max_size recipes in the dataset. 
    It returns a list of recipes and a dictionary of the recipe titles.
    Recipes are represented as lists of words which have been stemmed 
    with the Porter Stemmer provided by the NLTK library.
    """
    # Initialize an empty list to store the recipes and an empty dictionary for the recipe titles.
    recipes = []
    titles = {}

    # Open the csv file in read mode.
    with open('mini_dataset.csv', 'r') as f:
        # Get stop words and initialize the stemmer.
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()

        # Initialize csv reader for file
        recipereader = csv.reader(f, doublequote=True)
        # Skip first row in csv.
        next(recipereader)

        # Iterate over each row in the file. i keeps track of the row so we only read the first max_size.
        i = 0
        for recipe in tqdm(recipereader): # iterate over csv rows
            if i >= max_size:
                break
            try: # Avoids some issues in the csv formatting. Get the directions and title as strings.
                directions = recipe[3] 
                title = recipe[1]
            except:
                continue

            # Replace non-letter characters with nothing, convert to lowercase and split to form a list of words.
            directions = re.sub(r'[^a-zA-Z\s]+', '', directions).lower().split()
            # Convert to list of stemmed words by looking at stop_words list, and append to recipe list.
            directions_filt = [ps.stem(w) for w in directions if not w in stop_words]
            recipes.append(directions_filt)

            # Keep title as is while keeping track of recipe ID.
            titles[int(recipe[0])] = title
            i += 1

    # Return the list of recipes and titles.
    return recipes, titles

def make_positional_index(recipes):
    """
    This function builds a positional inverted index as a dictionary. 
    Here, for each term in the recipes, the index stores a dictionary 
    where the keys are document IDs and the values are lists of positions 
    where the term appears in each document.
    """
    # Create a defaultdict of dictionaries. The outer dictionary holds terms,
    # and the inner dictionary maps document IDs to a list of positions.
    index = defaultdict(dict)

    # Enumerate over the recipes to get both the document ID (docid) and the recipe itself.
    for docid, recipe in enumerate(recipes):
        # Enumerate over each term in the recipe to get both the position (pos) and the term.
        for pos, term in enumerate(recipe):
            try:
                # Try to append the position to the existing list for this term in this document.
                index[term.lower()][docid].append(pos)
            except KeyError:
                # If the term or document ID doesn't exist yet in the index,
                # create a new entry with the current position in a list.
                index[term.lower()][docid] = [pos]

    # Return the constructed positional inverted index.
    return index

def make_idf_index(recipes, p_index):
    """
    This function builds a positional inverted index as a dictionary,  
    where keys are terms in the vocabulary, and values are each term's 
    Inverse Document Frequency.
    """

    # Calculate the total number of documents
    n = len(recipes)

    # Calculate the Inverse Document Frequency (IDF) for each term
    index = {}
    for term in p_index.keys():
        index[term] = log(n / len(p_index[term]))

    return index

def make_tfidf_index(recipes, p_index, idf_index):
    """
    This function calculates the TF-IDF index for each term-document pair.
    It returns a dictionary of dictionaries, where keys in the outer dictionary 
    are terms, and keys in the inner dictionaries are documents. This gives a 
    reduced representation of all recipes as vectors, without assigning space 
    for terms which are not present. 
    It also returns a list containing the L2 norm of each document, for future 
    normalization. This function is suitable for small collections as its space 
    complexity is O(#documents x #terms).
    """
    # Initialize index as dictionary
    index = defaultdict(dict)
    # Calculate the total number of documents
    n = len(recipes)
    # Initialize empty list of document norms for normalization
    norms = [0] * n

    for term in tqdm(p_index.keys()): # Iterate over all terms in the dictionary.
        for docid in range(0, len(recipes)): # Iterate over all recipes in the corpus.
            # Calculate TF-IDF for each term and store it in the dictionary
            try:
                # Term Frequency (TF) is the number of times a term occurs in this document
                # Multiply TF with IDF to get TF-IDF
                tfidf = len(p_index[term][docid]) * idf_index[term]
                # Assign the TF-IDF score to the term-document pair
                index[term][docid] = tfidf
                # Increment this document's norm
                norms[docid] += tfidf**2
            except KeyError:
                # If the term is not in the document, do nothing and continue (don't store null tf-idf values).
                continue
    
    for docid in range(0, len(recipes)): # To have the correct L2 norm, we need to take square roots
        norms[docid] = sqrt(norms[docid])

    return index, norms

def vectorize_query(text):
    """
    This function takes a string of words as input and converts it 
    to vector form (as dictionary), in order to use it for querying. 
    In practice, we only need to store those terms which are present 
    in the query. Note that input text needs to be stemmed beforehand.
    """
    
    # Convert to lower-case and split the input string into words
    words = text.lower().split()

    # Initialize empty dictonary to store vectorized query
    query_dict = {}
    for term in tfidf_index.keys():
        if term in words: # Add an element only for terms in dictionary which are present in the query
            query_dict[term] = 1

    if len(query_dict) != 0: # Normalize when at least one word was matched
        norm = sqrt(sum([x**2 for x in query_dict.values()]))
        for term in query_dict.keys():
            query_dict[term] /= norm
    
    return query_dict

def query(query_vec, k): 
    """
    This function computes cosine similarity score between query 
    and all documents and prints the top k results.
    The input variable query_vec is assumed to be a dictionary of {term: weight} pairs, and normalized.
    """

    if len(query_vec) == 0: 
        raise ValueError('Error: no terms matched')
    
    # Initialize empty dictionary to store results of cosine similarity calculation
    scores = {}

    # Cosine similarity: we only need to iterate over non-zero elements of query.
    for term in query_vec.keys():
        for docid in range(len(titles)):
            try:
                # Compute product of tf-idf indices for this term, between query and current document
                add = query_vec[term] * tfidf_index[term][docid]
                try: 
                    scores[docid] += add
                except: # If document is not already in the dictionary, initialize key-value pair
                    scores[docid] = add
            except: # If the term is not present in the document, skip iteration (remember that null tf-idf values aren't stored)
                continue
    
    # Divide by document norm (query is already normalized) 
    for docid in range(len(titles)):
        try:
            scores[docid] /= norms[docid]
        except: # Key docid may not exist if document doesn't contain any terms in query
            continue

    # Create sorted copy of dictionary by descending similarity score
    scores_sorted = dict(sorted(scores.items(), key=lambda x:x[1], reverse=True))

    # Print top k results as id - score - title
    print(f'\n~~~~~~~~~~~~~~~~~~~~\nTop {k} results:\n')
    i = 0
    for id in scores_sorted.keys():
        if i >= k:
            break
        print(f'recipe {id}: {scores_sorted[id]} ({titles[id].strip()})')
        i += 1
    print('~~~~~~~~~~~~~~~~~~~~')
    
    return

def rocchio(old_query, r, nr):
    """
    This function performs relevance feedback, taking original query and lists of 
    relevant and non-relevant document ids as input, and returning the new updated query.
    old_query is assumed to be a dictionary, while r and nr should be lists.
    """

    thresh = 0.01 # Threshold for eliminating indices in new query

    # Paramenters for rocchio algorithm
    a = 1.0 # Coefficient for original query
    b = 0.75 # Coefficient for relevant documents
    c = 0.15 # Coefficient for non-relevant documents

    # Calculate number of relevant and non-relevant documents
    r_size = len(r)
    nr_size =  len(nr)

    # Calculation of centroid vectors as weighted sum of vectors in sets
    r_centroid = {} # Centroid of relevant vectors
    if r_size > 0:
        for term in tfidf_index.keys():
            for docid in r:
                try: # Try to add weight 
                    try:
                        r_centroid[term] += tfidf_index[term][int(docid)]
                    except:
                        r_centroid[term] = tfidf_index[term][int(docid)]
                except: # If term is not in document, either do nothing or initialize to zero
                    try:
                        r_centroid[term] += 0
                    except:
                        r_centroid[term] = 0
            r_centroid[term] /= r_size
    else: # If no relevant documents provided, set all weights to zero
        for term in tfidf_index.keys():
            r_centroid[term] = 0

    nr_centroid = {} # Centroid of non-relevant vectors
    if nr_size > 0:
        for term in tfidf_index.keys():
            for docid in nr:
                try: 
                    try:
                        nr_centroid[term] += tfidf_index[term][int(docid)]
                    except:
                        nr_centroid[term] = tfidf_index[term][int(docid)]
                except: 
                    try:
                        nr_centroid[term] += 0
                    except:
                        nr_centroid[term] = 0
            nr_centroid[term] /= nr_size
    else: # If no non-relevant documents provided, set all weights to zero
        for term in tfidf_index.keys():
            nr_centroid[term] = 0

    # Initialize empty dictionary for new query
    new_query = {}

    # Rocchio formula
    for term in tfidf_index.keys():
        try: # Check if the term is in old_query
            newvalue = max(a*old_query[term] + b*r_centroid[term] - c*nr_centroid[term], 0)
            # Use max(result, 0) to assign 0 to negative values
        except: # If term is not in old_query, its weight is zero
            newvalue = max(b*r_centroid[term] - c*nr_centroid[term], 0)
        if newvalue > thresh: # Add weight to new query only if greater than fixed threshold
            new_query[term] = newvalue

    if len(new_query) != 0: # Normalize new query
        norm = sqrt(sum([x**2 for x in new_query.values()]))
        for term in new_query.keys():
            new_query[term] /= norm

    return new_query

def new_query():
    """
    Wrapper function for querying. 
    Returns old query as vector, for use in relevance feedback.
    """
    
    text = input('\nEnter new query as free-form text: ')
    text_stem = ""

    # Perform stemming of input string for input
    for word in text.split():
        text_stem += ps.stem(word) + ' '
    print("After stemming:\n", text_stem)

    q = vectorize_query(text_stem)

    try:
        query(q, K)
    except:
        print('\nError: no terms matched in vocabulary')
        new_query()
        
    return q

def relevance_feedback(old_query):
    """
    Wrapper function for relevance feedback. Includes recursion for subsequent rf steps.
    Asks for list of relevant and non-relevant documents based on previous query output.
    """

    # Read strings of relevant and non-relevant docids and split into lists
    r = input('Enter DocIds of relevant documents (space separated numbers): ').split()
    nr = input('Enter DocIds of non-relevant documents (space separated numbers): ').split()

    new_query = rocchio(old_query, r, nr)
    print('\nOLD QUERY:', dict(sorted(old_query.items(), key=lambda x:x[1], reverse=True)))
    print('(~~~~~~~~~~~~~~~~~~~~\nNEW QUERY:', dict(sorted(new_query.items(), key=lambda x:x[1], reverse=True)))

    query(new_query, K)

    rf =  input('\nProceed with relevance feedback (y/n)? ')
    if rf == 'y':
        relevance_feedback(new_query)
    else:
        return
    
def save_files(tfidf_index, norms, titles):
    # Write tf-idf index, document norm list and titles list to files before closing
    # Tf-idf idex
    with open('tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf_index, f)
    # Norms list
    with open('norms.txt', 'w') as f:
        for i in range(len(norms)-1):
            f.write(f'{norms[i]}\n')
        f.write(f'{norms[len(norms)-1]}')
    # Titles list
    with open('titles.txt', 'w') as f:
        for i in range(len(titles)-1):
            f.write(f'{titles[i]}\n')
        f.write(f'{titles[len(titles)-1]}')
    return

def load_files():
    # Load tf-idf index
    with open('tfidf.pkl', 'rb') as f:
        tfidf_index = pickle.load(f)

    # Load document norms
    with open('norms.txt', 'r') as f:
        norms = []
        lines = f.readlines()
        for line in lines:
            norms.append(float(line))

    # Load recipe titles
    with open('titles.txt', 'r') as f:
        titles = []
        lines = f.readlines()
        for line in lines:
            titles.append(line)    

    return tfidf_index, norms, titles


##############################
####### INITIALIZATION #######
##############################

# Check whether to compute inverted index or read from file
while True:
    load = input('Load index from file (y/n)? ')
    if load == 'y':
        try: # Try to read files
            tfidf_index, norms, titles = load_files()
            break
        except IOError: # If files don't exist, quit program
            print('Could not read file(s)')
            sys.exit()

    elif load == 'n':
        # Get number of documents to import
        N = int(input('\nEnter number of documents to import: '))

        # Import dataset and calculate inverted indices
        print(f'Importing dataset ({N} recipes)')
        recipes, titles = import_dataset(N)   
        print('Creating positional index') 
        p_index = make_positional_index(recipes) 
        print('Creating idf index')
        idf_index = make_idf_index(recipes, p_index)
        print('Creating tf-idf index')
        tfidf_index, norms = make_tfidf_index(recipes, p_index, idf_index)

        break
    else:
        print("\nEnter 'y' or 'n'")
        continue
 
# Initialize Porter stemmer
ps = PorterStemmer()
# Set K as number of top documents displayed
K = 25

##########################
####### MAIN CYCLE #######
##########################

while True:
    new = input('\nEnter new query (y/n)? ')
    if new == 'y':
        old_query = new_query()

    else: # Exit program after choosing whether to save index, norms and titles files
        while True:
            save = input('\nSave index to file (y/n)? ')
            if save == 'y':
                save_files(tfidf_index, norms, titles)
                print('\nWritten Tf-Idf index, norms list and titles list to files')
                break
            elif save == 'n':
                break
            else:
                print("\nEnter 'y' or 'n'")
                continue
        sys.exit()

    rf = input('\nProceed with relevance feedback (y/n)? ')
    if rf == 'y':
        relevance_feedback(old_query)