import utilities as ut
import os

class constant_estimator(ut.Estimator):
    def __init__(self, c):
        self.c = c
    
    def predict(self, user, movie, date):
        return self.c 

def main():
    data_dir = 'data/'
    output_dir = 'output/'
    
    qual_path = os.path.join(data_dir, 'qual.dta')
    
    qual_x = []
    with open(qual_path, 'r') as qual_file:
        for line in qual_file.readlines():
            [user, movie, date] = line.split()
            qual_x.append((user, movie, date))
    
    estimator = constant_estimator(3.5)
    qual_y = estimator.predict_list(qual_x)
    
    output_path = os.path.join(output_dir, 'output.dta')
    with open(output_path, 'w') as output_file:
        output_file.write('\n'.join([str(y) for y in qual_y]))
        
    print 'done...'
    


    

if __name__ == '__main__':
    main()
