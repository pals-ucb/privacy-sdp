'''
    llm_base_ops class implements the baseline SDP model operations
    for fine-tuning, validating, evaluating and generating text using a 
    pretrained model for baseline operations.
'''
import sys
import time
import numpy as np
import constants as CONST
import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
# from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import s3fs
# Imports for DP
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager


class llm_base_ops:
    def __init__(self, args, llm, llm_tokenizer, device_type, train_dataloader, 
                 validate_dataloader, rank, world_size):
        self.args = args
        self.llm = llm
        self.llm_original = llm
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
        return

    def create_optimizer(self):
        #self.optimizer = torch.optim.AdamW(self.llm.parameters(), lr=CONST.LEARNING_RATE, eps=CONST.EPSILON)
        self.optimizer = torch.optim.SGD(self.llm.parameters(), lr=CONST.LEARNING_RATE)
        if self.args.optimizer == 2:
            self.enable_differential_privacy()

    def enable_differential_privacy(self):
        self.llm.train()  # put the model in training mode
        self.privacy_engine = PrivacyEngine()
        self.llm, self.optimizer, self.train_dataloader = self.privacy_engine.make_private_with_epsilon(
            module=self.llm,
            optimizer=self.optimizer,
            data_loader=self.train_dataloader,
            target_epsilon=CONST.PRIVACY_EPSILON,
            target_delta=CONST.PRIVACY_DELTA,
            epochs=self.args.epochs, 
            max_grad_norm=CONST.MAX_GRADIENT_NORM,
            batch_first=False
        )
        print('Enabled differential privacy')
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
        total_params = 0
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

    # Training function without dp
    def train(self, dataloader):
        st = time.time()
        for epoch in range(1, self.args.epochs+1):
            if self.world_size > 1:
                dataloader.sampler.set_epoch(epoch)
            self.llm.train()
            epoch_losses = []
            for step, (inputs,
                       attention_mask) in enumerate(tqdm(dataloader,
                                                         desc="Training")
                                                         ):
                self.optimizer.zero_grad()
                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.llm(inputs, attention_mask=attention_mask,
                                   labels=inputs)
                loss = outputs.loss
                loss.backward()
                epoch_losses.append(loss.item())
                self.optimizer.step()
                if step > 0 and step % self.args.epoch_log_interval == 0:
                    train_loss = np.mean(epoch_losses)
                    eval_loss, eval_accuracy = self.evaluate(self.validate_dataloader)
                    print('')
                    print(
                        f"Epoch: {epoch} | "
                        f"Step: {step} | "
                        f"Train loss: {train_loss:.3f} | "
                        f"Eval loss: {eval_loss:.3f} | "
                        f"Eval accuracy: {eval_accuracy:.3f} "
                        )
            train_loss = np.mean(epoch_losses)
            valid_loss = self.evaluate(self.validate_dataloader)
            print(f"Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {valid_loss}")
        en = time.time()
        print(f'llm_ops: training total time: {(en-st)/60} mins')
        return (st, en)

    def train_dp(self, dataloader):
        st = time.time()
        for epoch in range(1, self.args.epochs+1):
            epoch_losses = []
            with BatchMemoryManager(
                data_loader=dataloader, 
                max_physical_batch_size=CONST.BATCH_SIZE_MEM_LIMIT, 
                optimizer=self.optimizer
            ) as msdl:
                for step, (inputs, 
                           attention_mask) in enumerate(tqdm(msdl, 
                                                             desc='DP Training'
                                                            )
                                                        ):
                    self.optimizer.zero_grad()
                    inputs = inputs.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    outputs = self.llm(inputs, attention_mask=attention_mask,
                                       labels=inputs)
                    loss = outputs.loss
                    loss.backward()
                    epoch_losses.append(loss.item())
                    self.optimizer.step()
                    if step > 0 and step % self.args.epoch_log_interval == 0:
                        train_loss = np.mean(epoch_losses)
                        eps = self.privacy_engine.get_epsilon(CONST.PRIVACY_DELTA)
                        eval_loss, eval_accuracy = self.evaluate_dp(self.validate_dataloader)
                        print(
                          f"Epoch: {epoch} | "
                          f"Step: {step} | "
                          f"Train loss: {train_loss:.3f} | "
                          f"Eval loss: {eval_loss:.3f} | "
                          f"Eval accuracy: {eval_accuracy:.3f} | "
                          f"ɛ: {eps:.2f}"
                        )
        en = time.time()
        print(f'llm_ops: DP training total time: {(en-st)/60} mins')
        return (st, en)

    def evaluate(self, dataloader):
        def accuracy(y, y_hat):
            return (y == y_hat).mean()
        self.llm.eval()
        total_loss = 0
        loss_arr = []
        accuracy_arr = []
        for inputs, attention_mask in tqdm(dataloader, 
                                           desc="Evaluating"):
            with torch.no_grad():
                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.llm(inputs,
                                   attention_mask=attention_mask,
                                   labels=inputs)
                loss, logits = outputs[:2]
                preds = np.argmax(logits.detach().cpu().numpy(), axis=2)
                labels = inputs.detach().cpu().numpy()
                loss_arr.append(loss.item())
                accuracy_arr.append(accuracy(preds, labels))
        self.llm.train()
        return np.mean(loss_arr), np.mean(accuracy_arr) 

    def evaluate_dp(self, dataloader):
        def accuracy(y, y_hat):
            return (y == y_hat).mean()
        self.llm.eval()
        loss_arr = []
        accuracy_arr = []
        for inputs, attention_mask in dataloader:
            with torch.no_grad():
                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.llm(inputs,
                                   attention_mask=attention_mask,
                                   labels=inputs)
                loss, logits = outputs[:2]
                preds = np.argmax(logits.detach().cpu().numpy(), axis=2)
                labels = inputs.detach().cpu().numpy()
                loss_arr.append(loss.item())
                accuracy_arr.append(accuracy(preds, labels))
        self.llm.train()
        return np.mean(loss_arr), np.mean(accuracy_arr) 

    def train_start(self):
        print('llm_ops: starting model fine-tuning.')
        self.enable_device()
        if self.args.optimizer == 1:
            self.train(self.train_dataloader)
        elif self.args.optimizer == 2:
            self.train_dp(self.train_dataloader)
        else:
            print('llm_ops: invalid type of optimizer.')
            sys.exit(-1)
        return

    def generate(self, input_text, max_length=256):
        target = torch.device("cpu")
        self.llm.to(target)
        self.llm.eval()
        input_ids = self.llm_tokenizer.encode(input_text, return_tensors='pt').to(target)
        attention_mask = torch.tensor([1] * len(input_ids[0]), dtype=torch.long).unsqueeze(0).to(target)
   
        with torch.no_grad():
            output = self.llm_original.generate(input_ids,
                                                attention_mask=attention_mask,
                                                max_length=max_length,
                                                pad_token_id=self.llm_tokenizer.eos_token_id,
                                                do_sample=True,
                                                num_return_sequences=5,
                                                no_repeat_ngram_size=2,
                                                temperature=0.7,
                                                top_k=50,
                                                top_p=0.95)
        gen_text = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)
        return gen_text
        
    def save_model(self):
        if self.args.save_model:
            if self.args.optimizer == 1:
                self.llm.save_pretrained(self.args.path)
                self.llm_tokenizer.save_pretrained(self.args.path)
            elif self.args.optimizer == 2:
                self.llm_original.save_pretrained(self.args.path)
                self.llm_tokenizer.save_pretrained(self.args.path)
            else:
                print(f'Invalid optimizer: {self.args.optimizer}')
                return
            print(f'Saved model to: {self.args.path}')