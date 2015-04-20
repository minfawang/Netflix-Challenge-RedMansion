class Estimator:
    
    def train(self, x_list, y_list):
        pass
    
    def predict(self, user, movie, date):
        return 0
    
    def predict_list(self, x_list):
        """Predict all the data points in the user_movie_list
        
        :param x_list: a list contain (user, movie, date) pairs
        """
        g_list = []
        for (user, movie, date) in x_list:
            g_list.append(self.predict(user, movie, date)) 
        return g_list