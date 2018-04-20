from KNearestNeighborsRec import KNearestNeighborsRecModel
from ToVecTrainer import ToVecTrainerModel
import Utils
from keras.preprocessing.sequence import skipgrams
from random import shuffle, randint


class ProdWord2VecModel(KNearestNeighborsRecModel, ToVecTrainerModel):
    def __init__(self, vector_size=100):
        """ initializes prodtext2vec recommendation model
            Parameters
            ----------
                vector_size : int, size of low rank vectors
            Attributes
            ----------
                vector_size
                item_vectors : dict of (song_id => tag vectors)
                item_tfidf_vectors : dict of (song_id => tfidf vectors)
                tag_dict : dict of (tag => index)
                tfidf_model : gensim TfIdf model (for transforming songs not in original training)
                lsi_model : gensim lsi model (for transforming songs not in original training)
                embeddings_model : Keras model to train embeddings
                id_index_dict : dict (item_id => index in embeddings layer)
                    - needed to recover embedding weights from model
        """
        self.vector_size = vector_size
        self.item_vectors = {}
        self.item_tfidf_vectors = {}
        self.tag_dict = {}
        self.tfidf_model = {}
        self.lsi_model = {}
        self.embeddings_model = None
        self.id_index_dict = None

    def compute_tfidf_vectors(self):
        """ computes tfidf vectors
            saves item_tfidf_vectors, tag_vectors, and the tag_id"""
        id_vec_mapping, tag_dict = Utils.build_tag_vectors("KGRec-dataset/KGRec-dataset/KGRec-music/tags")
        self.tag_dict = tag_dict.token2id
        self.item_tfidf_vectors, self.tfidf_model = Utils.build_tfidf_vectors(id_vec_mapping)

    def train_embeddings(self, train_seqs, epochs, prev_model=None):
        """ computes item vectors by training neural net for embeddings using negative sampling and SGD
            Parameters
            ----------
                train_seqs : dict, (userid => interaction history (sequence of items))
                epochs : int, number of epochs to run for neural net
                prev_model : str, name of model to load to continue training, if None, initialize new model
        """
        vocab_size = len(self.tag_dict)
        print("building context and negative sampling pairs")
        data_couples = []
        labels = []
        for seq in train_seqs.values():
            new_seq = []
            for item_id in seq:
                new_seq.append(self.tag_dict[item_id])
            user_couples, user_labels = skipgrams(new_seq, vocabulary_size=vocab_size, window_size=5,
                                                  negative_samples=5.0)
            data_couples += user_couples
            labels += user_labels
        print(str(len(labels)) + " training examples")
        self._train_embeddings_model(data_couples, labels, epochs, prev_model)

    def skipgrams_product_words(self, sequence, item_id_tags, negative_samples=1., num_samples_per_word=1):
        """ build skipgram negative sampling pairs/labels for product words
            - modifies keras.sequencing.skipgrams implementation
        Parameters
        ----------
            sequence : list of item_ids in listening sequence
            item_id_tags : dict of (item_id => bag of words/tags)
            negative_samples : negative_samples: float >= 0. 0 for no negative (i.e. random) samples.
                1 for same number as positive samples.
            num_samples_per_word : int, number of couples to make for each word in a description for each context product
        Returns
        -------
            couples : list of (target, context) tuples
            labels : list of labels for couples
        """
        couples = []
        labels = []
        items_enumerated_list = list(enumerate(sequence))
        for i, item_id_i in items_enumerated_list:
            window_start = max(0, i - 1)
            window_end = min(len(sequence), i + 2)
            for j in range(window_start, window_end):
                if j != i:
                    for word_i in item_id_tags[item_id_i]:
                        context_words = shuffle(item_id_tags[items_enumerated_list[j]])
                        for word_index in range(num_samples_per_word):
                            couples.append([word_i, context_words[word_index]])
                            labels.append(1)
        if negative_samples > 0:
            num_negative_samples = int(len(labels) * negative_samples)
            words = [c[0] for c in couples]
            shuffle(words)
            couples += [[words[i % len(words)],
                         randint(1, len(item_id_tags) - 1)]
                        for i in range(num_negative_samples)]
            labels += [0] * num_negative_samples
        return couples, labels




# TODO: test skipgrams_product_words, implement parallelism




