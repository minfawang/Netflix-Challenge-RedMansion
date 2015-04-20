from Estimator import Estimator
import os
import numpy as np
import numpy.random as rnd


class MF_neicul(Estimator):
    def __init__(self):
        pass
    
    def sgd_gradient(self, y, U, V, A, B, l, mu, i, j):
        gU = l * U
        gV = l * V
        gA = l * A
        gB = l * B
        pFpX = 2 * (y - mu - (np.dot(np.transpose(U[:, i]), V[:, j]) + A[:, i] + B[:, j]))
        
        print pFpX
        
        gU[:, i] = gU[:, i] - pFpX * V[:, j]
        gV[:, j] = gV[:, j] - pFpX * U[:, i]
        gA[:, i] = gA[:, i] - pFpX
        gB[:, j] = gB[:, j] - pFpX
        return gU, gV, gA, gB
    
    def train(self, x_list, y_list, n_iter):
        K = 2
        
        M = np.size(y_list, 0)
        n_users = np.max(x_list[:, 0])
        n_movies = np.max(x_list[:, 1])
        mu = np.mean(y_list)
        l = 0.01 / (1.0 * n_users * n_movies)
        learning_rate = 0.005
        rate =  learning_rate
        
        print 'users:{}, movies:{}'.format(n_users, n_movies)
        
        U = rnd.rand(K, n_users)
        V = rnd.rand(K, n_movies)
        A = rnd.rand(1, n_users)
        B = rnd.rand(1, n_movies)
        
        I = x_list[:, 0]
        J = x_list[:, 1]
        
        for i_loop in xrange(n_iter):
            perm = rnd.choice(M, size=M, replace=True)
            for index in perm:
                i = I[index] - 1
                j = J[index] - 1
                y = y_list[index]
                
                gU, gV, gA, gB = self.sgd_gradient(y, U, V, A, B, l, mu, i, j)
                
                U = U - rate * gU
                V = V - rate * gV
                A = A - rate * gA
                B = B - rate * gB
            
            y_guess = np.zeros(M)
            for index in xrange(M):
                i = I[index] - 1
                j = J[index] - 1
                y_guess[index] = np.dot(np.transpose(U[:, i]), V[:, j]) + A[:, i] + B[:, j] + mu
            err = y_list - y_guess
            err2 = np.mean(err * err)
            print err2        
        pass
    
    def predict(self, user, movie, date):
        pass
    
def main():
    data_dir = '../data/'
    output_dir = '../output/'
    
    main_path = os.path.join(data_dir, 'main_q.npy')
    probe_path = os.path.join(data_dir, 'probe_q.npy')
    qual_path = os.path.join(data_dir, 'qual_q.npy')
    
    main_data = np.load(main_path)
    qual_data = np.load(qual_path)
    
    estimator = MF_neicul()
    estimator.train(main_data[:, 0:2], main_data[:, 3], 3)
    
    qual_y = estimator.predict_list(qual_data[:, 0:3])
    
    output_path = os.path.join(output_dir, 'output.dta')
    with open(output_path, 'w') as output_file:
        output_file.write('\n'.join([str(y) for y in qual_y]))
        
    print 'done...'
    

if __name__ == '__main__':
    main()
