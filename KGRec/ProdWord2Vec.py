from KNearestNeighborsRec import KNearestNeighborsRecModel
from KGRec.ToVecTrainer import ToVecTrainerModel
from KGRec import Utils
from random import shuffle, randint
import numpy as np


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
                tag_id_dict : dict of (tag => index)
                id_tag_dict : dict of (index => tag)
                embeddings_model : Keras model to train embeddings
                id_index_dict : dict (item_id => index in embeddings layer)
                    - needed to recover embedding weights from model
        """
        self.vector_size = vector_size
        self.item_vectors = {}
        self.item_tfidf_vectors = {}
        self.tag_id_dict = {}
        self.id_tag_dict = {}
        self.embeddings_model = None
        self.id_index_dict = None
        self.id_vec_mapping = {}

    def compute_tfidf_vectors(self):
        """ computes tfidf vectors
            saves item_tfidf_vectors, tag_vectors, and the tag_id"""
        self.id_vec_mapping, tag_dict = Utils.build_tag_vectors("KGRec-dataset/KGRec-dataset/KGRec-music/tags")
        for item_id, bow in self.id_vec_mapping.items():
            self.item_vectors[item_id] = [i[0] for i in bow]
        self.tag_id_dict = tag_dict.token2id
        self.id_tag_dict = {v: k for k, v in self.tag_id_dict.items()}
        self.item_tfidf_vectors, _ = Utils.build_tfidf_vectors(self.id_vec_mapping)

    def build_item_vectors(self, word_vectors, tfidf_weighting=True):
        items = list(self.item_vectors.keys())
        for item in items:
            vector = np.zeros(self.vector_size)
            for tfidf_tuple in self.item_tfidf_vectors[item]:
                # tfidf_tuple = (index, weight)
                if self.id_tag_dict[tfidf_tuple[0]] in word_vectors:
                    if tfidf_weighting:
                        vector += np.array(word_vectors[self.id_tag_dict[tfidf_tuple[0]]]) * tfidf_tuple[1]
                    else:
                        vector += np.array(word_vectors[self.id_tag_dict[tfidf_tuple[0]]])
            if not np.array_equal(vector, np.zeros(self.vector_size)):
                norm = np.linalg.norm(vector)
                self.item_vectors[item] = vector / norm
            else:
                del self.item_vectors[item]

    def discount_invalid_tags(self, train, test, tfidf_weighting):
        tag_songs_dict = {}
        for songId, tag_list in self.id_vec_mapping.items():
            for (tag, _) in tag_list:
                if tag in tag_songs_dict:
                    tag_songs_dict[tag].append(songId)
                else:
                    tag_songs_dict[tag] = [songId]

        invalid_tags = set()
        for user, songs in train.items():
            for tag, tag_songs in tag_songs_dict.items():
                if Utils.test_tag_placement(tag_songs, songs):
                    if tag in invalid_tags:
                        invalid_tags.remove(tag)
                    else:
                        invalid_tags.add(tag)

        for _, songID in test.items():
            vector = np.zeros(self.vector_size)
            for tfidf_tuple in self.item_tfidf_vectors[songID]:
                # tfidf_tuple = (index, weight)
                if self.id_tag_dict[tfidf_tuple[0]] in word_vectors and self.id_tag_dict[tfidf_tuple[0]] in invalid_tags:
                    if tfidf_weighting:
                        vector -= np.array(word_vectors[self.id_tag_dict[tfidf_tuple[0]]]) * tfidf_tuple[1]
                    else:
                        vector -= np.array(word_vectors[self.id_tag_dict[tfidf_tuple[0]]])
            if not np.array_equal(vector, np.zeros(self.vector_size)):
                norm = np.linalg.norm(vector)
                self.item_vectors[songID] = vector / norm
            else:
                if songID in self.item_vectors:
                    del self.item_vectors[songID]



    def train_embeddings(self, train_seqs, epochs, prev_model=None):
        """ computes item vectors by training neural net for embeddings using negative sampling and SGD
            Parameters
            ----------
                train_seqs : dict, (userid => interaction history (sequence of items))
                epochs : int, number of epochs to run for neural net
                prev_model : str, name of model to load to continue training, if None, initialize new model
        """
        print("building context and negative sampling pairs")
        data_couples = []
        labels = []
        for seq in train_seqs.values():
            user_couples, user_labels = self.skipgrams_product_words(seq,
                                                                     negative_samples=1.0,
                                                                     num_samples_per_word=2)
            data_couples += user_couples
            labels += user_labels
            print(user_couples)
            print(user_labels)
        print(str(len(labels)) + " training examples")
        exit(0)
        self._train_embeddings_model(data_couples, labels, epochs, prev_model)

    def skipgrams_product_words(self, sequence, negative_samples=1., num_samples_per_word=2):
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
                    for word_i in self.item_vectors[item_id_i]:
                        context_words = self.item_vectors[items_enumerated_list[j][1]]
                        shuffle(context_words)
                        context_count = len(context_words)
                        for word_index in range(min(context_count, num_samples_per_word)):
                            if word_i != context_words[word_index]:
                                couples.append([word_i, context_words[word_index]])
                                labels.append(1)
        if negative_samples > 0:
            num_negative_samples = int(len(labels) * negative_samples)
            words = [c[0] for c in couples]
            shuffle(words)
            couples += [[words[i % len(words)],
                         randint(1, len(self.tag_dict) - 1)]
                        for i in range(num_negative_samples)]
            labels += [0] * num_negative_samples
        return couples, labels


if __name__ == "__main__":
    print("loading embeddings")
    pw2v_model = ProdWord2VecModel(vector_size=100)
    pw2v_model.compute_tfidf_vectors()
    word_vectors = {}
    with open("embeddings/word_embeddings100.txt", 'r') as f:
        for line in f:
            item_id = line[line.index("(")+2:line.index(",")-1]
            vector = line[line.index("[")+1:-3].split(",")
            vector = [float(x.strip()) for x in vector]
            word_vectors[item_id] = vector
    pw2v_model.build_item_vectors(word_vectors, True)
    train, valid, test = Utils.load_dataset(pw2v_model.item_vectors)
    print("discounting invalid tags")
    pw2v_model.discount_invalid_tags(train, valid, True)
    train, valid, test = Utils.load_dataset(pw2v_model.item_vectors)
    print("getting recommendations")
    recs = pw2v_model.get_recs(train, k=10)
    print("computing metrics")
    hr, ndcg = Utils.compute_metrics(recs, valid)
    print(hr)
    print(ndcg)
    hr_cs, ndcg_cs = Utils.compute_cold_start_metrics(recs, train, test, window_size=1)
    print(hr_cs)
    print(ndcg_cs)


