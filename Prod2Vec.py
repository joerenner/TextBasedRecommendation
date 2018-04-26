from KNearestNeighborsRec import KNearestNeighborsRecModel
from ToVecTrainer import ToVecTrainerModel
import Utils
from keras.preprocessing.sequence import skipgrams
from keras.models import load_model
import numpy as np


class Prod2VecModel(KNearestNeighborsRecModel, ToVecTrainerModel):
    """ Model to train embedding network on product sequences. Neural network is trained and resulting embeddings
    are used to create recommendations
    """
    def __init__(self, vector_size=100):
        """ initializes Prod2Vec recommendation model
            Parameters
            ----------
                vector_size : int, embedding layer size
            Attributes
            ----------
                vector_size
                item_vectors : dict of (song_id => p2v vectors)
                embeddings_model : Keras model to train embeddings
                id_index_dict : dict (item_id => index in embeddings layer)
                    - needed to recover embedding weights from model
        """
        self.vector_size = vector_size
        self.item_vectors = {}
        self.embeddings_model = None
        self.id_index_dict = None

    def train_embeddings(self, train_seqs, epochs, id_index_dict_file, prev_model=None):
        """ computes item vectors by training neural net for embeddings using negative sampling and SGD
            Parameters
            ----------
                train_seqs : dict, (userid => interaction history (sequence of items))
                epochs : int, number of epochs to run for neural net
                id_index_dict_file : str, name of pickle file for id_index dict, if None, construct new
                 -  this is important if prev_model is defined, as the id_index has to be the same to continue training
                prev_model : str, name of model to load to continue training, if None, initialize new model
        """
        self.id_index_dict, _ = Utils.build_item_indices(list(train_seqs.values()), id_index_dict_file)
        vocab_size = len(self.id_index_dict)
        print("building context and negative sampling pairs")
        data_couples = []
        labels = []
        for seq in train_seqs.values():
            new_seq = []
            for item_id in seq:
                new_seq.append(self.id_index_dict[item_id])
            user_couples, user_labels = skipgrams(new_seq, vocabulary_size=vocab_size, window_size=5,
                                                  negative_samples=5.0)
            data_couples += user_couples
            labels += user_labels
        print(str(len(labels)) + " training examples")
        self._train_embeddings_model(data_couples, labels, epochs, prev_model)

    def compute_vectors(self, id_index_dict_file=None, model_file=None):
        """ extracts embedding weights from neural network and saves them in self.item_vectors
            Parameters
            ---------
                id_index_dict_file : str, name of pickle file for id_index dict, if None, use self.id_index_dict
                    - if id_index_dict_file == None and no self.id_index_dict is initialized, error will be raised
                model_file : str or None, if str, loads model from filename, if None, uses self.embeddings_model
                    - if model_file == None and no self.embeddings_model is initialized, error will be raised
        """
        if model_file:
            self.embeddings_model = load_model(model_file)
        elif not self.embeddings_model:
            raise ValueError("Reference to self.embeddings_model before initialization, provide model file name")
        if id_index_dict_file:
            self.id_index_dict, index_id_dict = Utils.build_item_indices([], id_index_dict_file)
        elif not self.id_index_dict:
            raise ValueError("Reference to self.id_index_dict before initialization, provide pickle file name")
        else:
            index_id_dict = dict(zip(self.id_index_dict.values(), self.id_index_dict.keys()))
        embeddings = self.embeddings_model.get_layer("embedding").get_weights()[0]
        for embedding_index, item_id in index_id_dict.items():
            dense_vec = embeddings[embedding_index]
            norm_value = np.linalg.norm(dense_vec)
            self.item_vectors[item_id] = dense_vec / norm_value


train, valid, test = Utils.load_dataset()
print("loading embeddings")
p2v_model = Prod2VecModel(vector_size=25)
with open("p2v_embeddings50.txt", 'r') as f:
    for line in f:
        item_id = line[line.index("(")+2:line.index(",")-1]
        vector = line[line.index("[")+1:-3].split(",")
        vector = [float(x.strip()) for x in vector]
        p2v_model.item_vectors[item_id] = vector

print("getting recommendations")
recs = p2v_model.get_recs(train, k=10)
print("computing metrics")
hr, ndcg = Utils.compute_metrics(recs, valid)
print(hr)
print(ndcg)
hr_cs, ndcg_cs = Utils.compute_cold_start_metrics(recs, train, test, window_size=5)
print(hr_cs)
print(ndcg_cs)

exit(0)
print("computing item vectors")
p2v_model.train_embeddings(train, epochs=1000000,
                           id_index_dict_file="id_index_file.pkl", prev_model="p2vEmbeddingModel.h5")
p2v_model.compute_vectors(id_index_dict_file="id_index_file.pkl", model_file="p2vEmbeddingModel.h5")


