# recipe-vsm
**Vector Space Model Information Retrieval System**

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

# How to run
Download the dataset from https://recipenlg.cs.put.poznan.pl and place it in the same folder as "vsm.py".
The program will ask whether to use existing files (which will not exist yet), or import the dataset. You must choose to import the dataset and specify how many recipes to import. I suggest not importing more than around 100000 recipes.

# Algorithm Structure:
1. **Setup**:\
    -Import dataset as list of lists of words (one list of words per document) (and dictionary of {DocId: Title} pairs).\
    -Create (or import existing) inverse indices: idf index and tf-idf index.
2. **Query**:\
    -Iterate over terms in query to compute Cosine Similarity score of all documents using Tf-Idf index.\
    -Return top K scoring documents (K is set to 25).\
    -Repeat with new query, go to 3 or end.
3. **Relevance Feedback**:\
    -Calculate the centroids of relevant and non-relevant documents.\
    -Create new query with Rocchio formula.\
    -Go to step 2.
