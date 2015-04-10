import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import os

def main():
    data_dir = 'data/'
    output_dir = 'output/'
    
    main_path = os.path.join(data_dir, 'main_q.npy')
    probe_path = os.path.join(data_dir, 'probe_q.npy')
    qual_path = os.path.join(data_dir, 'qual_q.npy')
    
    main_data = np.load(main_path, mmap_mode='r')
    probe_data = np.load(probe_path, mmap_mode='r')
    qual_data = np.load(qual_path, mmap_mode='r')
    
    
    for i in xrange(20):
        bin_l = i * 250
        bin_r = (i + 1) * 250
        
        index = main_data[:, 1] == 3
        index = main_data[index, 2] >= bin_l
        index = main_data[index, 2] < bin_r
        
        rates = main_data[index, 3]
        
        plt.figure() 
        if len(rates) > 0:   
                    
            plt.hist(rates, [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], rwidth=0.8)
    plt.show()
    
    
    
if __name__ == '__main__':
    main()
