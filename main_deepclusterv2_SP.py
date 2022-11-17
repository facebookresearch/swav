# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans
import random
import networkx as nx
from sklearn.manifold import TSNE

from scipy.sparse.linalg import eigs


import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from apex.parallel.LARC import LARC
from scipy.sparse import csr_matrix


from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from src.multicropdataset import MultiCropDataset
import src.resnet50 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of DeepCluster-v2")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## dcv2 specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=[3000, 3000, 3000], type=int, nargs="+",
                    help="number of prototypes - it can be multihead")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=1e10, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")


def main():
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
        return_index=True,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
    )

    # synchronize batch norm layers
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # wrap model
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]

    # build the memory bank
    mb_path = os.path.join(args.dump_path, "mb" + str(args.rank) + ".pth")
    if os.path.isfile(mb_path):
        mb_ckp = torch.load(mb_path)
        local_memory_index = mb_ckp["local_memory_index"]
        local_memory_embeddings = mb_ckp["local_memory_embeddings"]
    else:
        local_memory_index, local_memory_embeddings = init_memory(train_loader, model)

    cudnn.benchmark = True
    
    # plot for Loss & color for plotting the clusters
    plt_Loss_avg=[[],[]]
    color = ["#00FFFF", "#0000FF", "#8A2BE2", "#8B2323", "#8A360F", "#FF6103", "#7FFF00", "#FFD700"]

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        scores, local_memory_index, local_memory_embeddings, assignments = train(
            train_loader,
            model,
            optimizer,
            epoch,
            lr_schedule,
            local_memory_index,
            local_memory_embeddings,
        )
        training_stats.update(scores)
       
        # plotting the losses per epoch
        plt_Loss_avg[0].append(epoch)
        plt_Loss_avg[1].append(scores[1])
        print("plt_Loss_avg", plt_Loss_avg)

        # k empty cluster Ai
        Cluster = []
        for i in range(8):
            Cluster.append([])
        # fill  cluster Ai with indices of assigned elements and plot
        n = local_memory_embeddings.size(dim=1)
        local_memory_embeddings_np = local_memory_embeddings[0].cpu().numpy()
        local_memory_embeddings_np = np.transpose(local_memory_embeddings_np)
        assign_np = assignments[0].cpu().numpy()   
        
        for i in range(980):
            for q in range(8):
                if assign_np[local_memory_index[i]] == q:
                    Cluster[q].append(local_memory_embeddings_np[:,i])
        
        # plotting the clusters
        for i in range(8):
            Cluster[i] = np.array(Cluster[i])
            Cluster_2d = TSNE(n_components=2, perplexity=30, learning_rate=10).fit_transform(Cluster[i])
            conv_i = "% s" % i                                                                              #convert number of cluster to string, for labeling the plot
            plt.scatter(Cluster_2d[:,0], Cluster_2d[:,1], s=1, c=color[i], label="cluster" + conv_i)
        plt.legend()
        plt.savefig(str(epoch))
        plt.close()
        
        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        torch.save({"local_memory_embeddings": local_memory_embeddings,
                    "local_memory_index": local_memory_index}, mb_path)

    plt.plot(plt_Loss_avg[0],plt_Loss_avg[1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def train(loader, model, optimizer, epoch, schedule, local_memory_index, local_memory_embeddings):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)

    assignments = cluster_memory(epoch, model, local_memory_index, local_memory_embeddings, len(loader.dataset))         
    logger.info('Clustering for epoch {} done.'.format(epoch))

    end = time.time()
    start_idx = 0
    for it, (idx, inputs) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = schedule[iteration]

        # ============ multi-res forward passes ... ============
        emb, output = model(inputs)
        emb = emb.detach()
        bs = inputs[0].size(0)

        # ============ deepcluster-v2 loss ... ============
        loss = 0       
        for h in range(len(args.nmb_prototypes)):
            scores = output[h] / args.temperature                                                                   # scores €(20x8)                                                                 
            targets = assignments[h][idx].repeat(sum(args.nmb_crops)).cuda(non_blocking=True)                       # targets €(20)                                                                                                                                                                                         
            loss += cross_entropy(scores, targets)
        loss /= len(args.nmb_prototypes)
        
        # ============ backward and optim step ... ============
        optimizer.zero_grad()                                                           
        loss.backward()
        
        # cancel some gradients
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        optimizer.step()

        # ============ update memory banks ... ============
        local_memory_index[start_idx : start_idx + bs] = idx
        for i, crop_idx in enumerate(args.crops_for_assign):
            local_memory_embeddings[i][start_idx : start_idx + bs] = \
                emb[crop_idx * bs : (crop_idx + 1) * bs]
        start_idx += bs

        # ============ misc ... ============
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.optim.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg), local_memory_index, local_memory_embeddings, assignments


def init_memory(dataloader, model):
    size_memory_per_process = len(dataloader) * args.batch_size
    local_memory_index = torch.zeros(size_memory_per_process).long().cuda()
    local_memory_embeddings = torch.zeros(len(args.crops_for_assign), size_memory_per_process, args.feat_dim).cuda()
    start_idx = 0
    with torch.no_grad():
        logger.info('Start initializing the memory banks')
        for index, inputs in dataloader:
            nmb_unique_idx = inputs[0].size(0)
            index = index.cuda(non_blocking=True)

            # get embeddings
            outputs = []
            for crop_idx in args.crops_for_assign:
                inp = inputs[crop_idx].cuda(non_blocking=True)
                outputs.append(model(inp)[0])

            # fill the memory bank
            local_memory_index[start_idx : start_idx + nmb_unique_idx] = index
            for mb_idx, embeddings in enumerate(outputs):
                local_memory_embeddings[mb_idx][
                    start_idx : start_idx + nmb_unique_idx
                ] = embeddings
            start_idx += nmb_unique_idx
    logger.info('Initializion of the memory banks done.')
    return local_memory_index, local_memory_embeddings


def cluster_memory(epoch, model, local_memory_index, local_memory_embeddings, size_dataset, nmb_kmeans_iters=10):
    l = 20                                                                                                          # number of NN in similarity graph                                                                                                    
    n = local_memory_embeddings.size(dim=1)                                                                             # 980 (number of features/pictures)
    j = 0                                                                                                   
    assignments = -100 * torch.ones(len(args.nmb_prototypes), size_dataset).long()                                                                                             

    with torch.no_grad():
        for i_K, K in enumerate(args.nmb_prototypes):
           
            # SPECTRAL CLUSTERING           
            # cos similarity matrix S
            S = torch.zeros((n,n))
            dot_products = torch.mm(local_memory_embeddings[j], torch.transpose(local_memory_embeddings[j], 0, 1) )     # S (980x980) cos-similarity between each feature 
            lengths = torch.sqrt(dot_products.diagonal())
            for i in range(n):
                for q in range(n):
                    cosines = dot_products[i][q] / (lengths[i]*lengths[q])
                    S[i,q] = cosines
            S = torch.add(torch.div(S,2), 1/2)                                                                          # skaliere cos similarity to [0,1]           
            
            # W weighted adjacency matrix (undirected l-NN graph) 
            W = torch.zeros((n,n)).cuda(non_blocking=True)                                                   
            for i in range(n):
                v, ind = torch.topk(S[i], l*2)                                                                          # v = l biggest entries (WITH 1) of S & their indices                                   
                for x in range(l):
                    if i == ind[x]:
                        W[i][ind[x]] = 0                                                                   
                    else:
                        W[i][ind[x]] = v[x]
                        W[ind[x]][i] = v[x]
            
            W_np = W.cpu().numpy()
            G = nx.from_numpy_matrix(W_np)
            nx.draw(G, width=0.1, node_size=5)
            plt.draw() 
            plt.savefig(str('l-NN graph' + str(epoch)))
            plt.close()
            
            # sym_Laplacian
            D = torch.zeros((n,n)).cuda(non_blocking=True)
            D_MP = torch.zeros((n,n)).cuda(non_blocking=True)
            for i in range(n):
                D_MP[i][i] = 1 / torch.sqrt(torch.sum(W[i]))                                                            # D^-1/2 (Moore-Penrose-Matrix)
                D[i][i] = torch.sum(W[i])                                                                               # D      (DGREEMATRIX)

            L = (D-W).cuda(non_blocking=True)                                                                                     
            L1 = torch.matmul(D_MP, L).cuda(non_blocking=True)            
            L_sym = torch.matmul(L1, D_MP).cuda(non_blocking=True)
            L_sym = L_sym.cpu().numpy()

            # EV
            EW, EV = np.linalg.eigh(L_sym)
            First_k_EVs = EV[:, 1:K+1]                                                                                  # EV no.1 - no.(K+1)
            First_k_EVs = torch.from_numpy(First_k_EVs)     
            
            # matrix T
            T = torch.zeros((n,K)).cuda(non_blocking=True)
            for i in range(n):
                for h in range(K):
                    T[i][h] = First_k_EVs[i][h] / torch.norm(First_k_EVs[i,:])

            # k-means
            # data
            data_size, dims, num_clusters = n, 128, K
            
            # set device
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')

            # clustering
            cluster_ids_x, cluster_centers = kmeans(X = T, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
            cluster_ids_x = cluster_ids_x.cuda(non_blocking=True)
            
            # gather the assignments
            assignments_all = torch.empty(args.world_size, cluster_ids_x.size(0),
                                          dtype=cluster_ids_x.dtype, device=cluster_ids_x.device)
            assignments_all = list(assignments_all.unbind(0))
            dist_process = dist.all_gather(assignments_all, cluster_ids_x, async_op=True)
            dist_process.wait()
            assignments_all = torch.cat(assignments_all).cpu()
            
            # gather the indexes
            indexes_all = torch.empty(args.world_size, local_memory_index.size(0),
                                      dtype=local_memory_index.dtype, device=local_memory_index.device)
            indexes_all = list(indexes_all.unbind(0))
            dist_process = dist.all_gather(indexes_all, local_memory_index, async_op=True)
            dist_process.wait()
            indexes_all = torch.cat(indexes_all).cpu()
            
            # log assignments
            assignments[i_K][indexes_all] = assignments_all

            # next memory bank to use
            j = (j + 1) % len(args.crops_for_assign)                                                                       
    
    return assignments                                                                                                  # assignments €(987)


def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


if __name__ == "__main__":
    main()