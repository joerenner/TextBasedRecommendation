from gensim.models import LsiModel
from KNearestNeighborsRec import KNearestNeighborsRecModel
from scipy.spatial.distance import cosine
import Utils

class LatentSemanticIndexingModel(KNearestNeighborsRecModel):
    def __init__(self, vector_size = 100, k = 5):
        """ initializes LSI recommendation model
            Parameters
            ----------
                vector_size : int, size of low rank vectors
            Attributes
            ----------
                vector_size
                k : int, number of recommendations to generate per user
                vectors : dict of (song_id => lsi vectors)
                user_vectors : dict of (user_id => lsi vectors)
                tag_dict : gensim Dictionary of (tag => index)
                tfidf_model : gensim TfIdf model (for transforming songs not in original training)
                lsi_model : gensim lsi model (for transforming songs not in original training) 
        """
        self.vector_size = vector_size
        self.k = k
        self.vectors = {}
        self.user_vectors = {}
        self.tag_dict = {}
        self.tfidf_model = {}
        self.lsi_model = {}

    def compute_vectors(self):
        """ build sparse vectors, computes tfidf matrix, performs svd, saves low rank vectors """
        id_vec_mapping, self.tag_dict = Utils.build_tag_vectors("KGRec-dataset/KGRec-dataset/KGRec-music/tags")
        id_vec_mapping, self.tfidf_model = Utils.build_tfidf_vectors(id_vec_mapping)
        self.lsi_model = LsiModel(id_vec_mapping.values(), id2word=self.tag_dict, num_topics=self.vector_size)
        for song_id, tfidf_vector in id_vec_mapping.items():
            sparse_vec = self.lsi_model[tfidf_vector]
            if len(sparse_vec) > 0:
                self.vectors[song_id] = Utils.sparse_to_dense(sparse_vec, self.vector_size)

    def get_recs(self, user_vectors, user_history):
        """ generates k recommendations for every vector based on cosine similarity
        Parameters
        ----------
            user_vectors : dict, (user_id => user_vector)
            user_history : dict, (user_id => list of songs in listening history), to filter recommendations
            k : int, length of list of ranked recs for each query
        Returns 
        -------
            user_recs : dict, (user_id => list of ranked song recommendation ids)
        """
        user_recs = {}
        for user, vector in user_vectors.items():
            user_recs[user] = self._get_recs_user(vector, user_history[user])
        return user_recs

    def _get_recs_user(self, user_vector, history):
        """ generates k recommendations for a single vector
        overrides super method, as super assumes query vector is in training set, so precomputes similarity
        Parameters
        ----------
            user_vector : list, dense vector
            user_history : list, (user_id => list of songs in listening history), to filter recommendations
        Returns 
        -------
            user_recs : ranked list of song recommendation ids
        """
        recommendations = []
        rec_length = 0
        for song_id, song_vector in self.vectors.items():
            if song_id not in history:
                sim = 1.0 - cosine(song_vector, user_vector)
                recommendations = self._insert_song_into_recs(song_id, sim, rec_length, recommendations)
        return recommendations

    def build


lsiModel = LatentSemanticIndexingModel(100)
lsiModel.compute_vectors()
# TODO: test dataset loading, get recs super and LSI (implement build user)
train, valid, test = Utils.load_dataset()
recs = lsiModel.get_recs(users, user_hist, 2)
print(recs)