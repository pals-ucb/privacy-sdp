import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import constants as CONST
import wiki2_dataloader as w2dl
from llm_ops import llm_base_ops 
import llm_utils

def get_gpt2_model(model_name):
    gpt2_lm = GPT2LMHeadModel.from_pretrained(model_name)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_lm.resize_token_embeddings(len(gpt2_tokenizer)) 
    gpt2_optimizer = torch.optim.AdamW(gpt2_lm.parameters(), lr=CONST.LEARNING_RATE)
    return gpt2_lm, gpt2_tokenizer, gpt2_optimizer

def prepare_model(rank, world_size, device_type):
    '''
    prepare_models function preps the model for privacy qualification.
    It trains the model or loads a pre_trained model with the SDP qualification 
    dataset.

    Parameters:
        rank (int): The GPU node id on which this instance is spawned.
                    if this is 0 then we are on the main instance
        world_size (int): Total number of GPU instances
        
    '''
    model_name = 'gpt2'
    llm, tokenizer, optimizer = get_gpt2_model(model_name)
    train_dataloader, test_dataloader, validate_dataloader, _ = w2dl.get_wiki2_dataloaders(tokenizer, rank, world_size)
    llm_ops = llm_base_ops(llm, tokenizer, optimizer, device_type, rank, world_size)
    print(llm_ops.generate('Robert went on a trip to Las Vegas, and '))
    llm_ops.train_loop(CONST.NUM_EPOCHS, train_dataloader, validate_dataloader)
    print(llm_ops.generate('Robert went on a trip to Las Vegas, and '))


def main():

    world_size, device_type = llm_utils.get_torch_device_info()
    if world_size > 1:
        torch.multiprocessing.spawn(prepare_model, args=(world_size, device_type), nprocs=world_size, join=True)
    else:
        rank = 0
        world_size = 1
        prepare_model(rank, world_size, device_type)
        
if __name__ == "__main__":
    main()
