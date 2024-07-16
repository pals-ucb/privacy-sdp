'''
    llm_base_ops class implements the baseline SDP model operations
    for fine-tuning, validating, evaluating and generating text using a 
    pretrained model for baseline operations.
'''
import os
import sys
import time
import constants as CONST
import torch
#from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
#from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
#import s3fs
# Imports for DP
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

class llm_base_ops:
    def __init__(self, args, llm, llm_tokenizer, device_type, train_dataloader, 
                 validate_dataloader, rank, world_size):
        self.args = args
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.optimizer = None
        self.device_type = device_type
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.rank = rank
        self.world_size = world_size
        if world_size == 1:
            self.device = torch.device(device_type)
        else:
            self.device = torch.device(f'{device_type}:{rank}')

        if world_size > 2:
            print(f'llm_ops: Enabling DDP')
            self.enable_DDP()

        #self.enable_device()
        return
        
    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.llm.parameters(), lr=CONST.LEARNING_RATE, eps=CONST.EPSILON)
        if self.args.optimizer == 2:
            self.enable_differential_privacy()
            
    def enable_differential_privacy(self):
        self.llm.train() # put the model in training mode
        self.privacy_engine = PrivacyEngine()
        self.llm, self.optimizer, self.train_dataloader = self.privacy_engine.make_private_with_epsilon(
            module=self.llm,
            optimizer=self.optimizer,
            data_loader=self.train_dataloader,
            target_epsilon=CONST.PRIVACY_EPSILON,
            target_delta=CONST.PRIVACY_DELTA,
            epochs=self.args.epochs, 
            max_grad_norm=CONST.MAX_GRADIENT_NORM,
            batch_first = False
        )
        print(f'Enabled differential privacy')
        return

    def enable_device(self):
        print(f'llm_ops: pushing llm to device {self.device}')
        # Send the model to device
        self.llm = self.llm.to(self.device)
        return

    def enable_DDP(self):
        if self.world_size > 1:
            # Wrap the llm with DDP
            self.llm = DDP(sel.llm, device_ids=[self.rank])
        return

    def freeze_layers(self, layers):
        '''
        freeze_layers function freezes the given list of sublayers and their
        child. This will causes these parameers not to be updated during training.
        
        Parameters:
            layers (list) : List of submodules within the llm
                           
        '''
        total_params  = 0
        frozen_params = 0

        for param in self.llm.parameters():
            param.requires_grad = True
            total_params += param.numel()

        for layer in layers:
            sm = self.llm.get_submodule(layer)
            for _, child in sm.named_modules():
                for _, param in child.named_parameters():
                    if param.requires_grad:
                        param.requires_grad = False
                        frozen_params += param.numel()
        return total_params, frozen_params, total_params - frozen_params

    def get_grad_layers(self, ret_list, child=None, prefix=''):
        """
        Recursively prints the requires_grad flag for each layer in the model.
        """
        module = child if child else self.llm
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            # Check if the module has parameters
            if list(child.parameters()):
                for param_name, param in child.named_parameters(recurse=False):
                    ret_list.append(f"Layer: {full_name}.{param_name} - requires_grad: {param.requires_grad}")
            # Recursively check the child module
            self.get_grad_layers(ret_list, child, full_name)
    # Training function common
    def __train(self, dataloader, description):
        self.llm.train()
        total_loss = 0
        for inputs, attention_mask in tqdm(dataloader, desc=description):
            self.optimizer.zero_grad()
            inputs = inputs.to(self.device)
            attention_mask = attention_mask.to(self.device)
            outputs = self.llm(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss
        
    # Training function
    def train(self, dataloader):
        loss = self.__train(dataloader, "Training model")
        return loss/len(dataloader)

    def train_dp(self, dataloader):
        '''
        with BatchMemoryManager(data_loader=dataloader, 
                                max_physical_batch_size=self.args.batch_size, 
                                optimizer=self.optimizer) as memory_safe_data_loader:
            total_loss = self.__train(memory_safe_data_loader, "Training model with DP")
        '''
        with BatchMemoryManager(data_loader=dataloader, max_physical_batch_size=4, 
                                optimizer=self.optimizer ) as memory_safe_data_loader:
            loss = self.__train(memory_safe_data_loader, "Training model with DP")
        epsilon, best_alpha = self.privacy_engine.get_privacy_spent()
        print(f"Privacy spent (ε): {epsilon:.2f}")
        return loss / len(dataloader)
    
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
    def train_loop(self):
        self.enable_device()
        epochs = self.args.epochs
        st = time.time()
        for epoch in range(1, epochs+1):
            if self.world_size > 1:
                print(self.train_dataloader.sampler)
                self.train_dataloader.sampler.set_epoch(epoch)
            if self.args.optimizer == 1:
                train_loss = self.train(self.train_dataloader)
            elif self.args.optimizer == 2:
                train_loss = self.train_dp(self.train_dataloader)
            else:
                print(f'llm_ops: invalid type of optimizer.')
                sys.exit(-1)
            valid_loss = self.evaluate(self.valid_dataloader)
            print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {valid_loss}")
        en = time.time()
        print(f'llm_ops: training total time: {(en-st)/60} mins')
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
