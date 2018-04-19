from KNearestNeighborsRec import KNearestNeighborsRecModel
import Utils
from keras.preprocessing.sequence import skipgrams
from keras.models import load_model
import numpy as np


class Prod2VecModel(KNearestNeighborsRecModel):
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
        """
        self.vector_size = vector_size
        self.item_vectors = {}
        self.embeddings_model = []

    def compute_vectors(self, train_seqs, epochs, id_index_dict_file=None, prev_model=None):
        """ computes item vectors by training neural net for embeddings using negative sampling and SGD
            Parameters
            ----------
                train_seqs : dict, (userid => interaction history (sequence of items))
                epochs : int, number of epochs to run for neural net
                id_index_dict_file : str, name of pickle file for id_index dict, if None, construct new
                 -  this is important if prev_model is defined, as the id_index has to be the same to continue training
                prev_model : str, name of model to load to continue training, if None, initialize new model
        """

        id_index_dict, index_id_dict = Utils.build_item_indices(list(train_seqs.values()), id_index_dict_file)
        vocab_size = len(id_index_dict)
        # data preprocessing for negative sampling
        print("building context and negative sampling pairs")
        data_couples = []
        labels = []
        for seq in train_seqs.values():
            new_seq = []
            for item_id in seq:
                new_seq.append(id_index_dict[item_id])
            user_couples, user_labels = skipgrams(new_seq, vocabulary_size=vocab_size, window_size=5,
                                                  negative_samples=1.0)
            data_couples += user_couples
            labels += user_labels
        print(str(len(labels)) + " training examples")
        if prev_model:
            self.embeddings_model = load_model(prev_model)
        else:
            self.embeddings_model = Utils.build_embeddings_model(vocab_size, self.vector_size)
        target = np.zeros((1,))
        context = np.zeros((1,))
        label = np.zeros((1,))
        print("training embeddings model")
        for cnt in range(epochs):
            i = np.random.randint(0, len(labels) - 1)
            target[0,] = data_couples[i][0]
            context[0,] = data_couples[i][1]
            label[0,] = labels[i]
            loss = self.embeddings_model.train_on_batch([target, context], label)
            if cnt % 1000 == 0:
                print("Iteration {}, loss={}".format(cnt, loss))
                self.embeddings_model.save('p2vEmbeddingModel.h5')


train, valid, test = Utils.load_dataset()
p2v_model = Prod2VecModel(vector_size=100)
print("computing item vectors")
p2v_model.compute_vectors(train, 1000000, id_index_dict_file="id_index_file.pkl", prev_model="p2vEmbeddingModel.h5")
# TODO : find way to extract embeddings from keras model and save in self.item_vectors, train p2v and evaluate
