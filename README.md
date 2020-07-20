# Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
This code provides a PyTorch implementation and pretrained models for **SwAV** (**Sw**apping **A**ssignments between **V**iews), as described in the paper [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882).

<div align="center">
  <img width="100%" alt="SwAV Illustration" src="https://dl.fbaipublicfiles.com/deepcluster/animated.gif">
</div>

SwAV is an efficient and simple method for pre-training convnets without using annotations.
Similarly to contrastive approaches, SwAV learns representations by comparing transformations of an image, but unlike contrastive methods, it does not require to compute feature pairwise comparisons.
It makes our framework more efficient since it does not require a large memory bank or an auxiliary momentum network.
Specifically, our method simultaneously clusters the data while enforcing consistency between cluster assignments produced for different augmentations (or “views”) of the same image, instead of comparing features directly.
Simply put, we use a “swapped” prediction mechanism where we predict the cluster assignment of a view from the representation of another view.
Our method can be trained with large and small batches and can scale to unlimited amounts of data.

# Model Zoo

We provide baseline SwAV pre-trained models with ResNet-50 architecture in torchvision format.
Besides SwAV, we also provide models pre-trained with DeepCluster-v2 and SeLa-v2 obtained by applying improvements from the self-supervised community to [DeepCluster](https://arxiv.org/abs/1807.05520) and [SeLa](https://arxiv.org/abs/1911.05371) (see details in the [appendix of our paper](https://arxiv.org/abs/2006.09882)).

| method | epochs | batch-size | multi-crop | ImNet top-1 acc. | url | args | time / ep |
|-------------------|-------------------|-------------------|---------------------|--------------------|--------------------|--------------------|--------------------|
| SwAV | 800 | 4096 | 2x224 + 6x96 | 75.3 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar) | [script](./scripts/swav_800ep_pretrain.sh) | 3min40s |
| SwAV | 400 | 4096 | 2x224 + 6x96 | 74.6 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_pretrain.pth.tar) | [script](./scripts/swav_400ep_pretrain.sh) | 3min40s |
| SwAV | 200 | 4096 | 2x224 + 6x96 | 73.9 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_pretrain.pth.tar) | [script](./scripts/swav_200ep_pretrain.sh) | 3min40s |
| SwAV | 100 | 4096 | 2x224 + 6x96 | 72.1 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_100ep_pretrain.pth.tar) | [script](./scripts/swav_100ep_pretrain.sh) | 3min40s |
| SwAV | 200 | 256 | 2x224 + 6x96 | 72.7 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_bs256_pretrain.pth.tar) | [script](./scripts/swav_200ep_bs256_pretrain.sh) | 52min10s |
| SwAV | 400 | 256 | 2x224 + 6x96 | 74.3 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_bs256_pretrain.pth.tar) | [script](./scripts/swav_400ep_bs256_pretrain.sh) | 52min10s |
| DeepCluster-v2 | 400 | 4096 | 2x160 + 4x96 | 74.3 | [model](https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_400ep_pretrain.pth.tar) | [script](./scripts/deepclusterv2_400ep_pretrain.sh) | 3min13s |
| SeLa-v2 | 400 | 4096 | 2x160 + 4x96 | 71.8 | [model](https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar) | - | - |

# Running SwAV unsupervised training

SwAV is very simple to implement and experiment with.
Our implementation consists in a [main_swav.py](./main_swav.py) file from which are imported the dataset definition [src/multicropdataset.py](./src/multicropdataset.py), the model architecture [src/resnet50.py](./src/resnet50.py) and some miscellaneous training utilities [src/utils.py](./src/utils.py).

For example, to train SwAV baseline on a single node with 8 gpus for 400 epochs, run:
```
python -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
--data_path /path/to/imagenet \
--dist_url /your/distributed/init/method \
--epochs 400 \
--base_lr 0.6 \
--final_lr 0.0006 \
--warmup_epochs 0 \
--batch_size 32 \
--size_crops 224 96 \
--nmb_crops 2 6 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--use_fp16 true \
--freeze_prototypes_niters 5005 \
--queue_length 3840 \
--epoch_queue_starts 15
```
We refer the user to pytorch distributed documentation ([env](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization) or [file](https://pytorch.org/docs/stable/distributed.html#shared-file-system-initialization) or [tcp](https://pytorch.org/docs/stable/distributed.html#tcp-initialization)) for setting the distributed initialization method (parameter `dist_url`) correctly.

## Multinode training
Distributed training is available via Slurm. We provide several [SBATCH scripts](./scripts) to reproduce our SwAV models.
For example, to train SwAV on 8 nodes and 64 GPUs with a batch size of 4096 for 800 epochs run:
```
sbatch ./scripts/swav_800ep_pretrain.sh
```
Note that you might need to remove the copyright header from the sbatch file to launch it.

## Evaluate models: Linear classification on ImageNet
To train a supervised linear classifier on frozen features/weights on a single node with 8 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py \
--data_path /path/to/imagenet \
--dist_url /your/distributed/init/method \
--pretrained /path/to/checkpoints/swav_800ep_pretrain.pth.tar
```

## Requirements
- Python 3.6
- [PyTorch](http://pytorch.org) install > 1.4.0
- torchvision
- CUDA 10.1
- [Apex](https://github.com/NVIDIA/apex) with CUDA extension

## License
See the [LICENSE](LICENSE) file for more details.

## Citation
If you find this repository useful in your research, please cite:
```
@article{caron2020unsupervised,
  title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  journal={arXiv preprint arXiv:2006.09882},
  year={2020}
}
```
