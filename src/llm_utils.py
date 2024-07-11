import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def get_torch_device_info():
    if torch.cuda.is_available():
        device_type = "cuda"
        world_size  = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device_type = "mps"
        world_size  = 1
    else:
        device_type = "cpu"
        world_size  = 1
    return world_size, device_type

# Initialize distributed data parallel and setup the process group
def ddp_setup(rank, world_size):
    '''
    ddp_setup function sets up the distributed data parallels and the process group 
    for spawning the process.
    parameters:
        rank (int) : The GPU id for this instance. if 0 then this is the main node
        world_size (int): The number of GPUs on this node
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Cleanup the process group
def ddp_cleanup():
    dist.destroy_process_group()

