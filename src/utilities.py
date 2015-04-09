import numpy

class Estimator:
    def __init__(self):
        pass
    
    def predict(self, user, movie):
        return 0
    
    def predict_all(self, user_movie_list):
        """Predict all the data points in the user_movie_list
        
        :param user_movie_list: a list contain (user, movie) pairs
        """
        predict_list = []
        for (user, movie) in user_movie_list:
            predict_list.append(self.predict(user, movie)) 
        return predict_list
    