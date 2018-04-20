import numpy as np
from keras.models import load_model
import Utils


class ToVecTrainerModel:
    """
        Attributes (implemented in subclasses):
        ----------
            vocab_size : int, number of words in vocab
            vector_size : int, size of embedding layer
            embeddings_model : Keras model
    """

    def _train_embeddings_model(self, data_couples, labels, epochs, prev_model=None):
        """ computes item vectors by training neural net for embeddings using negative sampling and SGD
            saves neural net in self.embeddings_model
            Parameters
            ----------
                data_couples : list of (target word, context word)
                labels : list of corresponding labels for data_couples
                epochs : int, number of epochs to run for neural net
                prev_model : str, name of model to load to continue training, if None, initialize new model
        """
        # initialize model
        if prev_model:
            self.embeddings_model = load_model(prev_model)
        else:
            self.embeddings_model = Utils.build_embeddings_model(self.vocab_size, self.vector_size)
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
                self.embeddings_model.save('p2vEmbeddingModel.h5') # TODO: file name as argument (p2v or pw2v)
        self.embeddings_model.save('p2vEmbeddingModel.h5')