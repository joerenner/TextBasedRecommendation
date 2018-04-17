from scipy.spatial.distance import cosine

class KNearestNeighborsRecModel():
    """
        Attributes:
        ----------
            vectors : dict (song_id => vector)
            k : number of recommendations per user
    """

    def get_recs(self, user_vectors, user_history):
        """ generates k recommendations for every vector based on cosine similarity
        Parameters
        ----------
            user_vectors : dict, (user_id => songId) the last song listened to by the user
            user_history : dict, (user_id => list of songs in listening history), to filter recommendations
            k : int, length of list of ranked recs for each query
        Returns 
        -------
            user_recs : dict, (user_id => list of ranked song recommendation ids)
        """
        similarity_matrix = self.build_sim_matrix
        user_recs = {}
        for user, song_id in user_vectors.items():
            user_recs[user] = self._get_recs_user(song_id, user_history[user], similarity_matrix)
        return user_recs

    def _get_recs_user(self, query_id, history, similarity_matrix):
        """ generates k recommendations for a single vector
        Parameters
        ----------
            query_id : string, id of query song
            user_history : list of songs in listening history to filter recommendations
            similarity_matrix : dict of dicts, precomputed cosine sims between songs
        Returns 
        -------
            user_recs : ranked list of song recommendation ids
        """
        recommendations = []
        rec_length = 0
        for song_id, song_vector in self.vectors.items():
            if song_id not in history:
                sim = similarity_matrix[query_id][song_id]
                recommendations = self._insert_song_into_recs(song_id, sim, rec_length, recommendations)
        return recommendations

    def _insert_song_into_recs(self, song_id, sim, rec_length, recs):
        """ inserts song into recommendations based on similarity value
                Parameters
                ----------
                    song_id : string, id of song
                    sim : float, similarity value
                    rec_length : int, current length of recommendation list
                    recs : list of top k (songid, sim) in order by sim
                Returns 
                -------
                    recs : updated ranked list of song recommendation ids and similarity values
        """
        if rec_length < self.k:
            recs.append((song_id, sim))
            recs.sort(key=lambda x: x[1], reverse=True)
            rec_length += 1
            return recs
        else:
            for i in range(self.k):
                if sim > recs[i][1]:
                    recs.insert(i, (song_id, sim))
                    break
            return recs[:self.k]

    def _build_sim_matrix(self):
        """ precomputes song vs song similarity values
            Returns
            -------
                sim_matrix : dict of dicts for fast lookup, queryid => (songid => sim)
        """
        sim_matrix = {}
        for songid1, vector1 in self.vectors.items():
            for songid2, vector2 in self.vectors.items():
                if songid1 != songid2 and songid2 not in sim_matrix[songid1]:
                    sim = 1.0 - cosine(vector1, vector2)
                    sim_matrix[songid1][songid2] = sim
                    sim_matrix[songid2][songid1] = sim
        return sim_matrix