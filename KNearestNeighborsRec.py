from sklearn.neighbors import BallTree


class KNearestNeighborsRecModel:
    """
        Attributes (implemented in subclasses):
        ----------
            item_vectors : dict (song_id => vector)
            k : number of recommendations per user
    """
    def get_recs(self, user_history, k):
        """ generates k recommendations for every vector based on cosine similarity
        Parameters
        ----------
            user_history : dict, (user_id => list of songs in listening history), to filter recommendations
        Returns 
        -------
            user_recs : dict, (user_id => list of ranked song recommendation ids)
        """
        X = list(self.item_vectors.values())
        ids = list(self.item_vectors.keys())
        id_index_list = {}
        num_users = len(ids)
        for i in range(num_users):
            id_index_list[ids[i]] = i
        tree = BallTree(X, leaf_size=40)
        user_recs = {}
        for user, song_ids in user_history.items():
            _, recs_ind = tree.query([self.item_vectors[user_history[user][-1:][0]]], k=k)
            recs = []
            for i in recs_ind[0]:
                recs.append(ids[i])
            user_recs[user] = recs
        return user_recs

