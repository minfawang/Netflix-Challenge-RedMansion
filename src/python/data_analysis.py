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
    
    
    
    
    
if __name__ == '__main__':
    main()
