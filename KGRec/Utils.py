from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from os import listdir
import numpy as np
from keras.layers import Embedding, Input, Reshape, Dense, Dot
from keras import Model
import pickle


def add_to_dictionary(d, tuple):
    """adds a key value tuple to a dictionary
        Parameters
        ----------
            d : dictionary
            tuple : tuple of (key, value)
    """
    if tuple[0] not in d:
        d[tuple[0]] = tuple[1]


def append_to_value_set(key, value, d):
    """ adds a value to a key set in a dict, or adds the key, set(value]) to the dict if key not in dict
        Parameters
        ----------
            key : key for dict
            value : value to be appended to key list (or initialized in list)
            d : dictionary to add to
        Returns
        -------
            d : updated dictionary
    """
    if key in d:
        d[key].add(value)
    else:
        d[key] = set(value)
    return d


def build_coevent_dict(train, window_size):
    """ builds a coevent dict for items (for use when computing cold start metrics)
        Parameters
        ----------
            train : dict, (userID => list of songid listened to)
            window_size : int, window size for training, (to know which pairs have been used for training)
        Returns
        -------
            coevents : dict, (songId => list of songIds that come after key in a listening history)
    """
    coevents = {}
    for _, history in train.items():
        hist_len = len(history)
        for i in range(hist_len):
            for j in range(max(0, i-window_size), min(hist_len, i+window_size)):
                if i != j:
                    coevents = append_to_value_set(history[i], history[j], coevents)
    return coevents


def build_item_indices(train_seqs_list, id_index_dict_file):
    """ maps items to integers and returns dictionary and reverse dict for lookup
        Parameters
        ----------
            train_seqs : list of sequence lists
            id_index_dict_file : id_index_dict_file : str, name of pickle file for id_index dict, if None, construct new
        Returns
        -------
            id_index_dict : dict (item_id => index)
            index_id_dict : dict (index => item_id)
    """
    if id_index_dict_file:
        with open(id_index_dict_file, 'rb') as f:
            id_index_dict = pickle.load(f)
    else:
        id_index_dict = {}
        item_ids = list(set([item for sublist in train_seqs_list for item in sublist]))
        for item_id in item_ids:
            index = len(id_index_dict)
            id_index_dict[item_id] = index
        with open('id_index_file' + '.pkl', 'wb') as f:
            pickle.dump(id_index_dict, f)
    index_id_dict = dict(zip(id_index_dict.values(), id_index_dict.keys()))
    return id_index_dict, index_id_dict


def build_tag_vectors(tag_directory_path):
    """Loads tag files, builds sparse vectors for each song
        Parameters
        ----------
            tag_directory_path : String, path of directory containing tags
        Returns
        -------
            id_vec_mapping : dict (song id => list[tuple(tagId, count)])
            dictionary : gensim Dictionary containing all tags and ids
    """
    dictionary = Dictionary()
    for f in listdir(tag_directory_path):
        with open(tag_directory_path+"/"+f, 'r') as tags:
            tokens = tags.read().split(sep=' ')
            dictionary.add_documents([tokens])
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    dictionary.compactify()
    id_vec_mapping = {}
    for f in listdir(tag_directory_path):
        song_id = f[0:-4]
        with open(tag_directory_path+"/"+f, 'r') as tags:
            tokens = tags.read().split(sep=' ')
        sparse_vec = dictionary.doc2bow(tokens)
        add_to_dictionary(id_vec_mapping, (song_id, sparse_vec))
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


def compute_cold_start_metrics(recs, train, test, window_size=1):
    """ computes hit rate, nDCG for recommendations for query => next pairs unseen in training set
        Parameters
        ----------
            recs : dict, (userID => list of recommendations)
            train : dict, (userID => training items)
            test : dict, dict (userID => next songID)
            window_size : int, window size used for training, for P2V = 5, else = 1
        Returns
        -------
            hit_rate : float, number of hits divided by K (1/K if next song is in recommendations)
            nDCG : float, normalized discounted cumulative gain
                https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
        Notes
        -----
            Since we are only predicting the next event, (only one target item) the ideal DCG = 1: (2^1 - 1)/log(1+1)
    """
    k = float(len(list(recs.values())[0]))
    num_users = len(list(recs.keys()))
    hit_rate = 0.0
    nDCG = 0.0
    coevents = build_coevent_dict(train, 1)
    for user, rec_items in recs.items():
        if not train[user][-1:][0] in coevents:
            if test[user] in rec_items:
                hit_rate += 1.0 / k
                nDCG += 1.0 / np.log2(rec_items.index(test[user]) + 2)
        elif not test[user] in coevents[train[user][-1:][0]]:
            if test[user] in rec_items:
                hit_rate += 1.0 / k
                nDCG += 1.0 / np.log2(rec_items.index(test[user]) + 2)
    return hit_rate / num_users, nDCG / num_users


def compute_metrics(recs, test):
    """ computes hit rate, nDCG for recommendations
        Parameters
        ----------
            recs : dict, (userID => list of recommendations)
            test : dict, dict (userID => next songID)
        Returns
        -------
            hit_rate : float, number of hits divided by K (1/K if next song is in recommendations)
            nDCG : float, normalized discounted cumulative gain 
                https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
        Notes
        -----
            Since we are only predicting the next event, (only one target item) the ideal DCG = 1: (2^1 - 1)/log(1+1)
    """
    k = float(len(list(recs.values())[0]))
    num_users = len(list(recs.keys()))
    hit_rate = 0.0
    nDCG = 0.0
    for user, rec_items in recs.items():
        if test[user] in rec_items:
            hit_rate += 1.0 / k
            nDCG += 1.0 / np.log2(rec_items.index(test[user]) + 2)
    return hit_rate / num_users, nDCG / num_users


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
    if not valid_songs or song in valid_songs:
        history.append(song)
    return history

def test_tag_placement(tag_songs, user_songs):
    for song in tag_songs:
        if song not in user_songs:
            return False
    return True


def load_dataset(valid_songs=None):
    """ Loads data into training, validation, test sets.
        For each user sequence of n items, first n-2 is training, n-1th is validation, and n is testing.
        After hyperparameter tuning, combine training and validation to predict on test
        Parameters
        ----------
            valid_songs = dict of (song_id => _)valid for dataset (invalid songs could be songs with no tags, for example)
                if None, assumes all songs are valid
        Returns
        -------
            train : dict (userId => list of songIds)
            validation : dict (userId => songId)
            test : dict (userId => songId)
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


def sparse_to_dense(sparse_vector, size, norm=False):
    """ turns sparse vector into dense
        Parameters
        ----------
            sparse_vector : list of (index, value) tuples
            size : int, dimensionality of dense vector
            norm : Boolean, if true normalize the resulting vector using 2-norm
        Returns
        -------
            dense_vector : dense numpy array
    """
    dense_vec = np.zeros(size)
    for (index, value) in sparse_vector:
        dense_vec[index] = value
    if norm:
        norm_value = np.linalg.norm(dense_vec)
        return dense_vec / norm_value
    return dense_vec



