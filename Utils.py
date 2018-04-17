from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from os import listdir
import numpy as np

def addToDictionary(d, tuple):
    """adds a key value tuple to a dictionary
        Parameters
        ----------
            d : dictionary
            tuple : tuple of (key, value)
    """
    if tuple[0] not in d:
        d[tuple[0]] = tuple[1]

def build_tag_vectors(tag_directory_path):
    """Loads tag files, builds sparse vectors for each song
        Parameters
        ----------
            tag_directory_path : String, path of directory containing tags
        Returns
        -------
            id_vec_mapping : dict (song id => vector)
            dictionary : gensim Dictionary containing all tags and ids
    """
    dictionary = Dictionary()
    lengths = []
    for f in listdir(tag_directory_path):
        with open(tag_directory_path+"/"+f, 'r') as tags:
            tokens = tags.read().split(sep=' ')
            lengths.append(len(tokens))
            dictionary.add_documents([tokens])
    dictionary.filter_extremes(no_below=2)
    dictionary.compactify()
    id_vec_mapping = {}
    for f in listdir(tag_directory_path):
        song_id = f[0:-4]
        with open(tag_directory_path+"/"+f, 'r') as tags:
            tokens = tags.read().split(sep=' ')
        lengths.append(len(tokens))
        sparseVec = dictionary.doc2bow(tokens)
        addToDictionary(id_vec_mapping, (song_id, sparseVec))
    return id_vec_mapping, dictionary

def build_tfidf_vectors(id_vec_mapping):
    """ builds tfidf model, transforms sparse vectors tfidf vectors
        Parameters
        ----------
            id_vec_mapping : dict (song id => sparse tag vector)
        Returns
        -------
            id_vec_mapping : dict (song id => tfidf vector)
    """
    tfidf = TfidfModel(id_vec_mapping.values())
    for id, vec in id_vec_mapping.items():
        id_vec_mapping[id] = tfidf[vec]
    return id_vec_mapping, tfidf

def sparse_to_dense(sparse_vector, size):
    """ turns sparse vector into dense
        Parameters
        ---------- 
            sparse_vector : list of (index, value) tuples
            size : int, dimensionality of dense vector
        Returns
        -------
            dense_vector : dense numpy array
    """
    dense_vec = np.zeros(size)
    for (index, value) in sparse_vector:
        dense_vec[index] = value
    return dense_vec

def load_dataset(valid_songs=None):
    """ Loads data into training, validation, test sets.
        For each user sequence of n items, first n-2 is training, n-1th is validation, and n is testing.
        After hyperparameter tuning, combine training and validation to predict on test
        Parameters
        ----------
            valid_songs = set of song_id valid for dataset (invalid songs could be songs with no tags, for example)
                if None, assumes all songs are valid
        Returns
        -------
            train : dict (userId => list of songIds)
            validation : dict (userId => list of songId)
            test : dict (userId => list of songId)
    """
    train = {}
    validation = {}
    test = {}
    with open("KGRec-dataset/KGRec-dataset/KGRec-music/implicit_lf_dataset.csv", 'r') as data:
        current_user = ""
        current_user_data = []
        for line in data:
            tokens = line.split(sep='\t')
            if tokens[0] != current_user and current_user != "":
                if len(current_user_data) >= 3:
                    train[current_user] = current_user_data[:-2]
                    validation[current_user] = current_user_data[-2]
                    test[current_user] = current_user_data[-1]
                current_user = tokens[0]
                current_user_data = insert_song_into_history(tokens[1], [], valid_songs)
            elif current_user == "":
                current_user = tokens[0]
                current_user_data = insert_song_into_history(tokens[1], [], valid_songs)
            else:
                current_user_data = insert_song_into_history(tokens[1], current_user_data, valid_songs)
    return train, validation, test

def insert_song_into_history(song, history, valid_songs):
    """ helper function to insert song, if valid, into listening history
        Parameters
        ----------
            song : string id
            history : list, history that song will be inserted into
            valid_songs : set of eligible songs, or None
        Returns
        -------
            history : list, updated listening history
            """
    if valid_songs == None or song in valid_songs:
        history.append(song)
    return history






