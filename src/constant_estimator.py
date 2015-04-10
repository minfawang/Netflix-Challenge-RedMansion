import utilities as ut
import numpy as np
import os

class constant_estimator(ut.Estimator):
    def __init__(self, c):
        self.c = c
    
    def predict(self, user, movie, date):
        return self.c 

def main():
    data_dir = 'data/'
    output_dir = 'output/'
    
    main_path = os.path.join(data_dir, 'main_q.npy')
    probe_path = os.path.join(data_dir, 'probe_q.npy')
    qual_path = os.path.join(data_dir, 'qual_q.npy')
    
    qual_data = np.load(qual_path)
    
    estimator = constant_estimator(3.5)
    qual_y = estimator.predict_list(qual_data[:, 0:3])
    
    output_path = os.path.join(output_dir, 'output.dta')
    with open(output_path, 'w') as output_file:
        output_file.write('\n'.join([str(y) for y in qual_y]))
        
    print 'done...'
    

if __name__ == '__main__':
    main()
