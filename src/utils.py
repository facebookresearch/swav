# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from logging import getLogger
import pickle
import os

import numpy as np
import torch

from .logger import create_logger, PD_Stats

import torch.distributed as dist

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


logger = getLogger()

class SwAV_Args:
    def __init__(
        self,
        data_path="/path/to/imagenet",
        nmb_crops=[2],
        size_crops=[224],
        min_scale_crops=[0.14],
        max_scale_crops=[1],
        crops_for_assign=[0, 1],
        temperature=0.1,
        epsilon=0.05,
        sinkhorn_iterations=3,
        feat_dim=128,
        nmb_prototypes=3000,
        queue_length=0,
        epoch_queue_starts=15,
        epochs=100,
        batch_size=64,
        base_lr=4.8,
        final_lr=0,
        freeze_prototypes_niters=313,
        wd=1e-6,
        warmup_epochs=10,
        start_warmup=0,
        arch="resnet50",
        hidden_mlp=2048,
        checkpoint_freq=25,
        sync_bn="pytorch",
        dump_path=".",
        seed=31
    ) -> None:
        """SwAV arguments.

        A class containing all the default - if not overwritten at initialization -
        SwAV arguments.

        Keyword Args:
            data_path (str): Path to dataset repository.
            nmb_crops (list): List of number of crops (example: [2, 6]).
            size_crops (list): Crops resolutions (example: [224, 96]).
            min_scale_crops (list): Argument in RandomResizedCrop (example: [0.14, 0.05]).
            max_scale_crops (list): Argument in RandomResizedCrop (example: [1., 0.14]).
            crops_for_assign (list): List of crops id used for computing assignments.
            temperature (float): Temperature parameter in training loss.
            epsilon (float): Regularization parameter for Sinkhorn-Knopp algorithm.
            sinkhorn_iterations (int): Number of iterations in Sinkhorn-Knopp algorithm.
            feat_dim (int): Feature dimension.
            nmb_prototypes (int): Number of prototypes.
            queue_length (int): Length of the queue (0 for no queue).
            epoch_queue_starts (int): From this epoch, we start using a queue.
            epochs (int): Number of total epochs to run.
            # batch_size (int): Batch size per gpu, i.e. how many unique instances per gpu.
            # bath_size has to be defined the the dataset_kwargs
            base_lr (float): Base learning rate.
            final_lr (float): Final learning rate.
            freeze_prototypes_niters (int): Freeze the prototypes during this many iterations from the start.
            wd (float): Weight decay.
            warmup_epochs (int): Number of warmup epochs.
            start_warmup (float): Initial warmup learning rate.
            arch (str): Convnet architecture.
            hidden_mlp (int): Hidden layer dimension in projection head.
            checkpoint_freq (int): Save the model periodically.
            sync_bn (str): Synchronize bn.
            dump_path (str): Experiment dump path for checkpoints and log.
            seed (int): Seed.
        """
        for argname, argval in dict(locals()).items():
          setattr(self, argname, argval)
        
    def to_dict(self):
        return {k:v for k,v in vars(self).items()
                if k not in ('train_kwargs', 'self')}

    @property
    def train_kwargs(self):
        return {
            k: getattr(self, k)
            for k in (
                'crops_for_assign', 'nmb_crops', 'temperature', 
                'freeze_prototypes_niters', 'epsilon', 'sinkhorn_iterations'
            )
        }

def get_args(**kwargs):
    """Configure a ``SwAV`` object for training SwAV.

    Keyword args:
        **kwargs: Please see the :class:`slideflow.swav.SwAV_Args` documentation
            for information on available parameters.

    Returns:
        slideflow.swav.SwAV

    """
    return SwAV_Args(**kwargs)

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    return


def initialize_exp(params, *args, dump_params=True):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    # dump parameters
    if dump_params:
        pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create repo to store checkpoints
    params.dump_checkpoints = os.path.join(params.dump_path, "checkpoints")
    if not params.rank and not os.path.isdir(params.dump_checkpoints):
        os.mkdir(params.dump_checkpoints)

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.dump_path, "stats" + str(params.rank) + ".pkl"), args
    )

    # create a logger
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"), rank=params.rank
    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("")
    return logger, training_stats


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(
        ckp_path,
        map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count())
    )

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
