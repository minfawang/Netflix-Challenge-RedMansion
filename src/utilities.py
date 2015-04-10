import numpy

class Estimator:
    
    def predict(self, user, movie, date):
        return 0
    
    def predict_list(self, input_list):
        """Predict all the data points in the user_movie_list
        
        :param input_list: a list contain (user, movie, date) pairs
        """
        predict_list = []
        for (user, movie, date) in input_list:
            predict_list.append(self.predict(user, movie, date)) 
        return predict_list
