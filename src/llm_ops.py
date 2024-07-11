'''
    llm_base_ops class implements the baseline SDP model operations
    for fine-tuning, validating, evaluating and generating text using a 
    pretrained model for baseline operations.
'''
import os
import time
import torch
#from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
#from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
#import s3fs

class llm_base_ops:
    def __init__(self, llm, llm_tokenizer, optimizer, device_type, rank=0, world_size=1):
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.optimizer = optimizer
        self.device_type = device_type
        self.rank = rank
        self.world_size = world_size
        if world_size == 1:
            self.device = torch.device(device_type)
        else:
            self.device = torch.device(f'{device_type}:{rank}')
        print(f'Device set to {self.device}')
        # Send the model to device
        self.llm = self.llm.to(self.device)

        if self.world_size > 1:
            # Wrap the llm with DDP
            self.llm = DDP(sel.llm, device_ids=[self.rank])
        return

    # Training function
    def train(self, dataloader):
        self.llm.train()
        total_loss = 0
        for inputs, attention_mask in tqdm(dataloader, desc="Training model"):
            self.optimizer.zero_grad()
            inputs = inputs.to(self.device)
            attention_mask = attention_mask.to(self.device)
            outputs = self.llm(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.llm.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, attention_mask in tqdm(dataloader, desc="Evaluating llm"):
                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.llm(inputs, attention_mask=attention_mask, labels=inputs)
                loss = outputs.loss
                total_loss += loss.item()
        return total_loss / len(dataloader)

    # Training loop
    def train_loop(self, epochs, train_dataloader, valid_dataloader):
        print(f"Using device: {self.device}")
        self.llm = self.llm.to(self.device)
        st = time.time()
        for epoch in range(1, epochs+1):
            train_loss = self.train(train_dataloader)
            valid_loss = self.evaluate(valid_dataloader)
            print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {valid_loss}")
        en = time.time()
        return (st, en)

    def generate(self, input_text, max_length=256):
        target = torch.device("cpu")
        self.llm.to(target)
        self.llm.eval()
        input_ids = self.llm_tokenizer.encode(input_text, return_tensors='pt').to(target)
        attention_mask = torch.tensor([1] * len(input_ids[0]), dtype=torch.long).unsqueeze(0).to(target)
   
        with torch.no_grad():
            output = self.llm.generate(input_ids, attention_mask=attention_mask, max_length=max_length, 
                                  pad_token_id=self.llm_tokenizer.eos_token_id, do_sample=True,
                                  num_return_sequences=5,
                                  no_repeat_ngram_size=2,
                                  temperature=0.7, 
                                  top_k=50, top_p=0.95)
        gen_text = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)
        return gen_text
        

en = time.time()
