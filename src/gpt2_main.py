#!python
import os
import sys
import argparse
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
    return gpt2_lm, gpt2_tokenizer


def prepare_model(rank, args, world_size, device_type):
    '''
    prepare_models function preps the model for privacy qualification.
    It trains the model or loads a pre_trained model with the SDP qualification 
    dataset.

    Parameters:
        rank (int): The GPU node id on which this instance is spawned.
                    if this is 0 then we are on the main instance
        world_size (int): Total number of GPU instances

    '''
    if world_size > 1:
        print(f'Setting up DDP for rank: {rank}')
        llm_utils.ddp_setup()
    else:
        print('Skip DDP setup.')

    model_name = 'gpt2'
    llm, tokenizer = get_gpt2_model(model_name)
    train_dataloader, test_dataloader, validate_dataloader, _ = w2dl.get_wiki2_dataloaders(args, tokenizer, rank, world_size)
    llm_ops = llm_base_ops(args, llm, tokenizer, device_type, train_dataloader,
                           validate_dataloader,  rank, world_size)
    if rank == 0 and not args.skip:
        print(llm_ops.generate('Robert went on a trip to Las Vegas, and '))
    if args.optimizer == 2:
        layers_to_freeze = ['transformer.h.0', 'transformer.h.1',
                            'transformer.h.2', 'transformer.h.3',
                            'transformer.h.4', 'transformer.h.5',
                            'transformer.h.6', 'transformer.h.7',
                            'transformer.h.8', 'transformer.h.9',
                            'transformer.ln_f',
                            'transformer.wte', 'transformer.wpe'
                            ]
        r = llm_ops.freeze_layers(layers_to_freeze)
        print(f'Number of parameters           : {r[0]:>10}')
        print(f'Number of Trainable Parameters : {r[2]:>10}')
        print(f'Number of frozen parameters    : {r[1]:>10}')
        '''
        layers = []
        llm_ops.get_grad_layers(layers)
        for e in layers:
            print(e)
         '''
    llm_ops.create_optimizer()
    llm_ops.train_start()
    if rank == 0:
        print(llm_ops.generate('Robert went on a trip to Las Vegas, and '))
    if world_size > 1:
        print(f'Cleaning up DDP for rank: {rank}')
        llm_utils.ddp_cleanup()


def main(args):
    world_size, device_type = llm_utils.get_torch_device_info()
    print(f'world_size: {world_size}, device_type: {device_type}')
    if args.optimizer == 1:
        print('optimizer: Stochastic Gradient Descent')
    elif args.optimizer == 2:
        print('optimizer: Differential Privacy Stochastic Gradient Descent')
    else:
        sys.exit(f'Invalid optimizer provided : {args.optimizer}')
    if world_size > 1:
        torch.multiprocessing.spawn(prepare_model, args=(args, world_size, device_type), nprocs=world_size, join=True)
    else:
        rank = 0
        world_size = 1
        prepare_model(rank, args, world_size, device_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gpt2 llm fine-tuning with optimizers SGD and DP-SGD using wiki2 dataset')
    parser.add_argument('-e', '--epochs', type=int, 
                        default=CONST.NUM_EPOCHS, 
                        help='Total epochs to train the model')
    parser.add_argument('-b', '--batch_size', type=int, 
                        default=CONST.BATCH_SIZE, 
                        help='Set the batch size for training.')
    parser.add_argument('-l', '--epoch_log_interval', type=int, 
                        default=CONST.EPOCH_LOG_INTERVAL, 
                        help='Set the epoch log interval for training.')
    parser.add_argument('-o', '--optimizer', type=int, 
                        default=1, 
                        help='The optimizer type SGD: 1 , DP SGD: 2')
    parser.add_argument('-s', '--skip', action="store_true", 
                        help='Skip the initial testing of the model (before training)')
    args = parser.parse_args()
    main(args)
