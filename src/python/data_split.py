import numpy as np
import os

def main():
    data_dir = 'data/'
    output_dir = 'output/'
    
    idx_path = os.path.join(data_dir, 'all.idx')
    data_path = os.path.join(data_dir, 'all.dta')
    
    print 'loading index...'
    
    with open(idx_path) as idx_file:
        idx = idx_file.read().split()
     
    len_main = 0
    len_probe = 0
    len_qual = 0
       
    for x in idx:
        if x=='1' or x=='2' or x=='3':
            len_main += 1
        elif x == '4':
            len_probe += 1
        else:
            len_qual += 1
    
    print len_main, len_probe, len_qual


    main_path = os.path.join(data_dir, 'main_q')
    probe_path = os.path.join(data_dir, 'probe_q')
    qual_path = os.path.join(data_dir, 'qual_q')
    
    main_data = np.empty((len_main, 4), dtype=int)
    probe_data = np.empty((len_probe, 4), dtype=int)
    qual_data = np.empty((len_qual, 4), dtype=int)
    
    print 'generating main_data, prob_data, qual_data...'
    
    i = 0
    j = 0
    k = 0
    h = 0
    
    with open(data_path) as data_file:    
        for line in data_file:
            if i % 100000 == 0:
                print 'row {}00000'.format(i / 100000)
            row_data_tuple = map(int, line.split())
            row_data = np.array(row_data_tuple, dtype=int)
            if idx[i] == '1' or idx[i] == '2' or idx[i] == '3':
                main_data[j, :] = row_data
                j += 1
            elif idx[i] == '4':
                probe_data[k, :] = row_data
                k += 1
            elif idx[i] == '5':
                qual_data[h, :] = row_data
                h += 1
            i += 1    

    print 'saving data...'

    np.save(main_path, main_data)
    np.save(probe_path, probe_data)
    np.save(qual_path, qual_data)
    
    print 'done'
    
    
if __name__ == '__main__':
    main()
