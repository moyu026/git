import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import transforms, datasets
from torchvision.models.resnet import resnet18 as model

def init_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def ddp_generator(seed=112):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g

def ddp_main(rank, world_size, data_root):
    try:
        init_ddp(rank, world_size)

        g = ddp_generator()

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
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler, generator=g)

        net = model().cuda(rank)
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

        for epoch in range(10):
            train_sampler.set_epoch(epoch)
            net.train()
            for step, data in enumerate(train_loader):
                images, labels = data
                images, labels = images.cuda(rank), labels.cuda(rank)
                output = net(images)
                loss = loss_function(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if rank == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

    except Exception as e:
        print(f"An error occurred in rank {rank}: {e}")
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    data_root = '/path/to/your/dataset'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    n_gpus = torch.cuda.device_count()
    mp.spawn(ddp_main, args=(n_gpus, data_root), nprocs=n_gpus, join=True)
