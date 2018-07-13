from sklearn.neighbors import BallTree


class KNearestNeighborsRecModel:
    """
        Attributes (implemented in subclasses):
        ----------
            item_vectors : dict (song_id => vector)
    """
    def get_recs(self, user_history, k):
        """ generates k recommendations for every vector based on cosine similarity
        Parameters
        ----------
            user_history : dict, (user_id => list of songs in listening history), to filter recommendations
            k : int, number of recommendations to make per user
        Returns 
        -------
            user_recs : dict, (user_id => list of ranked song recommendation ids)
        """
        X = list(self.item_vectors.values())
        ids = list(self.item_vectors.keys())
        id_index_list = {}
        num_items = len(ids)
        for i in range(num_items):
            id_index_list[ids[i]] = i
        tree = BallTree(X, leaf_size=40)
        user_recs = {}
        j = 0
        num_users = len(user_history)
        for user, song_ids in user_history.items():
            _, recs_ind = tree.query([self.item_vectors[song_ids[-1:][0]]], k=k+len(song_ids))
            recs = []
            for i in recs_ind[0]:
                # filter items in user history
                if ids[i] not in song_ids:
                    recs.append(ids[i])
            user_recs[user] = recs[:k]
            j += 1
            if j % 1000 == 0:
                print(str(j) + " out of " + str(num_users) + " recs computed")
        return user_recs

