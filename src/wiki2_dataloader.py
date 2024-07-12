'''
    Project: SDP
    Wiki2 Dataset Dataloader

'''

import os
import time
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import s3fs
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'

'''
constants is a local module that defines the constants used here like SEQUENCE_LENGTH, BATCH_SIZE etc.
'''
import constants as CONST

class Wiki2TextDataset(Dataset):
    '''
    class Wiki2TextDataset implements the torch Dataset class and is used to create training, test, validate dataloaders.
    
    '''
    def __init__(self, file_path, tokenizer, world_size =1, max_length=CONST.SEQUENCE_LENGTH):
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open(file_path, 'r', encoding='utf-8') as fd:
            self.tokens = []
            self.attention_masks = [] # Attention masks
            for line in fd:
                sline = line.strip()
                if len(sline) > 0:
                    tokens = tokenizer.encode(sline, truncation=True, max_length=max_length, padding='max_length')
                    attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in tokens]
                    self.tokens.append(torch.tensor(tokens, dtype=torch.long))
                    self.attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.attention_masks[i]
    
def load_dataset(file_path, tokenizer, rank, world_size, batch_size,
                 shuffle=False, max_length=CONST.SEQUENCE_LENGTH):
    '''
    Load_dataset function loads the wiki2 data from the input file and returns the 
    dataloader object 

    Parameters:
        file_path (str): The full path to the dataset file
        tokenizer (Transformer.Tokenizer*): The Transformers based tokenizer associated with the model
        shuffle (bool) : Shuffle the dataset when True
        max_length (int) : The maximum length of each line 
        batch_size (int): The size of each batch
    '''   
    dataset = Wiki2TextDataset(file_path, tokenizer, max_length=max_length)
    if world_size > 1:
        sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size)
        return DataLoader(dataset, sampler=sampler, batch_size=batch_size, pin_memory=True, shuffle=shuffle)
    else:
        return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle)

def get_wiki2_dataloaders(args, tokenizer, rank=0, world_size=1):
    '''
    get_wiki2_dataloaders function loades the wiki2 train, test, valid and unittest 
    data sets.
    Parameters:
        args (argparse.ArgumentParser): The command line argument parser object
        tokenizer (Transformers.Tokenizer*): The tokenizer object associated with the model
        rank (int): The rank provides the current GPU instance id and is 0 by default.
        world_size (int): World size provides the number of GPUs available for training. When > 1 
                          DDP Torch is used for parallel training. For cpu only system 
                          world_size = 1.
    '''
    root = 's3://differential-privacy-datasets'
    wikitext2_root = root + '/kaggle-wikitext/wikitext-2/'
    train_file = wikitext2_root + 'wiki.train.tokens'
    test_file  = wikitext2_root + 'wiki.test.tokens'
    valid_file = wikitext2_root + 'wiki.valid.tokens'
    unittest_file = wikitext2_root + 'unittest.tokens'

    train_dataloader    = load_dataset(train_file, tokenizer, rank, world_size, args.batch_size, shuffle=True)
    test_dataloader     = load_dataset(test_file, tokenizer, rank, world_size, args.batch_size)
    valid_dataloader    = load_dataset(valid_file, tokenizer, rank, world_size, args.batch_size)
    unittest_dataloader = load_dataset(unittest_file, tokenizer, rank, world_size, args.batch_size)
    return (train_dataloader, test_dataloader, valid_dataloader, unittest_dataloader)
    