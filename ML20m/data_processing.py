import numpy as np


def append_to_value_list(key, value, d):
    if key in d:
        d[key].append(value)
    else:
        d[key] = [value]
    return d


def append_to_value_set(key, value, d):
    if key in d:
        d[key].add(value)
    else:
        d[key] = set(value)
    return d


def load_data(validation=True):
    data_by_user = {}
    with open("../ml-20m/ratings.csv", 'r', encoding="utf8") as ratings:
        for line in ratings:
            values = line.rstrip().split(',')
            if values[0] != "userId":
                # key = user ID, value = list of (movie, timestamp) tuples
                data_by_user = append_to_value_list(values[0], (values[1], int(values[3])), data_by_user)
    test_by_user = {}
    if validation:
        end = -2
    else:
        end = -1
    for user, list_of_movies in data_by_user.items():
        sequence = sorted(list_of_movies, key=lambda x: x[1])
        data_by_user[user] = [x[0] for x in sequence[:end]]
        test_by_user[user] = sequence[end][0]
    train_user_movies = {}
    for user, movies in data_by_user.items():
        train_user_movies[user] = set(movies)
    return data_by_user, train_user_movies, test_by_user


def load_word_embeddings(file_name, vector_size):
    word_embeddings = {}
    with open(file_name, 'r', encoding="utf8") as f:
        for line in f:
            index = line.rindex("[")
            word = line[:index - 1]
            embedding = line[index + 1:].split()
            if len(embedding) == vector_size:
                embedding[-1] = embedding[-1][:-1]
            word_embeddings[word] = np.array(embedding[:vector_size], dtype=float)
    return word_embeddings


def load_word_weights(file_name):
    word_weights = {}
    with open(file_name, 'r', encoding="utf8") as f:
        for line in f:
            tokens = line.split()
            weight = tokens[-1]
            tag = line[:line.index(weight)-1]
            word_weights[tag] = float(weight)
    return word_weights


def load_tags(train, min_count=2):
    # loads tags (and genres as tags), only using tags from training data (user_item_sets)
    movie_tags = {}
    seen_tags = set()
    # first pass: get counts of words (not needed for genres)
    counts = {}
    with open("../ml-20m/tags.csv", 'r', encoding="utf8") as tags:
        for line in tags:
            values = line.rstrip().split(',')
            if values[0] != "userId" and values[1] in train[values[0]]:
                if values[2].lower().replace('"', '') in counts:
                    counts[values[2].lower().replace('"', '')] += 1
                else:
                    counts[values[2].lower().replace('"', '')] = 1
    # second pass: add tags only if count is higher than threshold
    with open("../ml-20m/tags.csv", 'r', encoding="utf8") as tags:
        for line in tags:
            values = line.rstrip().split(',')
            if values[0] != "userId" \
                    and values[1] in train[values[0]] \
                    and counts[values[2].lower().replace('"', '')] > min_count:
                movie_tags = append_to_value_list(values[1], values[2].lower().replace('"', ''), movie_tags)
                seen_tags.add(values[2].lower().replace('"', ''))
    with open("../ml-20m/movies.csv", 'r', encoding="utf8") as genres:
        for line in genres:
            values = line.rstrip().split(',')
            if len(values) == 3 and values[0] != "movieId":
                for genre in values[2].split('|'):
                    movie_tags = append_to_value_list(values[0], genre.lower(), movie_tags)
                    seen_tags.add(genre.lower())
    return movie_tags, seen_tags


def filter_movies_with_no_tags(train, test, movie_tags):
    for user, movie_list in train.items():
        new_movie_list = movie_list
        for movie in movie_list:
            if movie not in movie_tags:
                new_movie_list = list(filter(lambda x: x != movie, new_movie_list))
        train[user] = new_movie_list
    for user, movie in test.items():
        if movie not in movie_tags:
            test[user] = train[user][-1]
            train[user] = train[user][:-1]
    return train, test


def build_index_dictionary(vocab):
    dictionary = dict()
    for i, word in enumerate(vocab):
        dictionary[word] = i
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reversed_dictionary


def build_product_embeddings(word_embeddings, item_tags, weights=None):
    """ weights:
            None : uniform weighting for each word
            "tfidf" : compute tfidf for each product tags, weight by value
            vector of shape (vocab_size) : use learned word weights
    """
    if weights == "tfidf":
        # dict of {itemId => dict{tag => tfidf_value}}
        tfidf_dict = compute_tfidf(item_tags)
    vector_size = len(list(word_embeddings.values())[0])
    item_embeddings = {}
    for item, tags in item_tags.items():
        vector = np.zeros(vector_size)
        for tag in tags:
            if not weights:
                vector += np.array(word_embeddings[tag])
            elif weights == "tfidf":
                vector += np.array(word_embeddings[tag]) * tfidf_dict[item][tag]
            else:
                if tag in word_embeddings and tag in weights:
                    vector += np.array(word_embeddings[tag]) * weights[tag]
                else:
                    print(tag)
                    print(tag in word_embeddings)
                    print(tag in weights)
        norm = np.linalg.norm(vector)
        item_embeddings[item] = vector / norm
    return item_embeddings


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


def compute_tfidf(item_tags):
    """ returns dict of {itemId => dict{tag => tfidf_value}} """
    from gensim.corpora import Dictionary
    from gensim.models import TfidfModel
    dictionary = Dictionary()
    item_bow = {}
    for item, tags in item_tags.items():
        dictionary.add_documents([tags])
    for item, tags in item_tags.items():
        sparse_vec = dictionary.doc2bow(tags)
        item_bow[item] = sparse_vec
    tfidf = TfidfModel(item_bow.values())
    id2token = dictionary.token2id
    id2token = dict(zip(id2token.values(), id2token.keys()))
    item_tfidf = {}
    for item, vec in item_bow.items():
        item_tfidf[item] = {}
        tfidf_vec = tfidf[vec]
        for (indx, value) in tfidf_vec:
            item_tfidf[item][id2token[indx]] = value
    return item_tfidf


def get_movie_vocab(train_sets):
    """
        train_sets : dict of {user_id => set(train movies)}
        returns set of all movie ids
    """
    all_movies = set()
    for _, train_set in train_sets.items():
        all_movies = all_movies.union(train_set)
    return all_movies


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


def compute_cold_start_metrics(recs, train, test, window_size=1):
    """ computes hit rate, nDCG for recommendations for query => next pairs unseen in training set
        Parameters
        ----------
            recs : dict, (userID => list of recommendations)
            train : dict, (userID => training items)
            test : dict, (userID => next songID)
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
    coevents = build_coevent_dict(train, window_size)
    for user, rec_items in recs.items():
        if train[user][-1:][0] in coevents:
            if not test[user] in coevents[train[user][-1:][0]]:
                if test[user] in rec_items:
                    hit_rate += 1.0 / k
                    nDCG += 1.0 / np.log2(rec_items.index(test[user]) + 2)
    return hit_rate / num_users, nDCG / num_users


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
