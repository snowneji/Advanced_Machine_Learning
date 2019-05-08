import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'glove.twitter.27B.50D.txt',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    ########################
    #### YOUR CODE HERE ####
    ########################

    import numpy as np
    
    


    print("Loading Glove Model")
    f = open(embeddings_path,'rb')
    embedding_model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        word = str(word, 'utf-8')
        embedding = np.array([float(val) for val in splitLine[1:]])
    #     print(str(word))
        
        embedding_model[str(word)] = embedding
    print("Done.",len(embedding_model)," words loaded!")
    return embedding_model,50

# def question_to_vec(question, embeddings, dim):
#     """Transforms a string to an embedding by averaging word embeddings."""
    
#     # Hint: you have already implemented exactly this function in the 3rd assignment.

#     ########################
#     #### YOUR CODE HERE ####
#     ########################

#     # remove this when you're done
#     raise NotImplementedError(
#         "Open utils.py and fill with your code. In case of Google Colab, download"
#         "(https://github.com/hse-aml/natural-language-processing/blob/master/project/utils.py), "
#         "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")

def question_to_vec(question, embeddings, dim=50):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    
    question_list = question.split(' ')
    question_list = [i for i in question_list if len(i)>0]

    all_embs = [embeddings[w] for w in question_list if w in embeddings]
    
    
    if sum([len(i) for i in all_embs])>0:
      return  np.mean(all_embs,axis=0)
 
    else:
      return np.zeros(dim)

def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
