import random
import numpy as np
import torch

class DataLoader:
    
    def __init__(self, file_path, batch_size=16):
        
        self.batch_size = batch_size
        self.char_to_ix = {
            'x': 0,
            '+': 1,
            '-': 2,
            '*': 3,
            '/': 4,
            '_': 5,
            #'\n': 6
        }
        self.ix_to_char = {v:k for (k,v) in self.char_to_ix.items()}
        self.readFile(file_path)
        self.idx = 0
        
    def __len__(self):
        pass
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.lines)
            
    def next(self):
        
        # iterator edge case
        if self.idx >= self.total_lines:
            raise StopIteration
    
        # figure out end_index based on what is left in the list
        if(self.idx + self.batch_size < self.total_lines):
            end_index = self.idx + self.batch_size
        else:
            end_index = self.total_lines

        # contains list of strings (length is batch_size and each element of this list is math eq string)
        batch_lines = self.lines[self.idx : end_index]
        
        #increment idx (bookeeping for iterator)
        self.idx += self.batch_size
        
        # contains input data to be returned
        all_input_data = []
        # contains target data to be returned
        all_target_data = []
        
        for line in batch_lines:
            # convert char to index (do this for input data and target data)
            # here input data and target data are staggered by one position
            input_data = [self.char_to_ix[c] for c in line]
            # target doesn't contain the first char, add 6 (maps to '\n') to end
            target_data = input_data[1:]
            target_data.append(0)
            
            all_input_data.append(input_data)
            all_target_data.append(target_data)

        # convert to torch long tensor (ready to be used by nn.Embedding)
        all_input_data = torch.from_numpy(np.asarray(all_input_data)).long()
        all_target_data = torch.from_numpy(np.asarray(all_target_data)).long()

        return all_input_data, all_target_data
    
    def readFile(self, file_path):
        with open(file_path, 'r') as f:
            self.lines = f.read().split('\n')
        self.total_lines = len(self.lines)

    def frequency(self, file_path):
        freq_arr = np.zeros((6,6))
        with open(file_path, 'r') as f:
            self.lines = f.read().split('\n')
            chars = list(self.lines)
    
            for i in range(1,len(chars)):
                freq_arr[self.char_to_ix.get(chars[i-1]),self.char_to_ix.get(chars[i])]+=1
        np.save('freq_array.npy',freq_arr/np.sum(freq_arr))


        self.total_lines = len(self.lines)


    def convert_to_char(self, data):
        string_arr = []
        for each_tensor in data:
            string = ''.join([self.ix_to_char[i] for i in each_tensor.data.numpy()])
            string_arr.append(string)
        return string_arr