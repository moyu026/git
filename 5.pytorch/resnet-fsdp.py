import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
from torchvision import transforms, datasets
from torchvision.models import resnet18

def init_fsdp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def fsdp_generator(seed=112):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g

def fsdp_main(rank, world_size, data_root):
    init_fsdp(rank, world_size)
    torch.manual_seed(0)
    g = fsdp_generator()

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    train_dataset = datasets.ImageFolder(root=data_root, transform=data_transform["train"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler, generator=g)

    torch.cuda.set_device(rank)
    model = resnet18()
    model = model.cuda(rank)
    model = wrap(model)  # 使用 wrap 将模型包装成 FSDP 模型
    model = FSDP(model)  # 使用 FSDP 包装模型

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

    for epoch in range(2):
        train_sampler.set_epoch(epoch)
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            images = images.cuda(rank, non_blocking=True)
            labels = labels.cuda(rank, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            if rank == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    dist.destroy_process_group()

if __name__ == '__main__':
    data_root = '/home/yuzhenchen/pythonwork/dataset/classification/button'
    os.environ['NCCL_P2P_DISABLE'] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    n_gpus = torch.cuda.device_count()
    mp.spawn(fsdp_main, args=(n_gpus, data_root), nprocs=n_gpus, join=True)
