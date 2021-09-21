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

We release several models pre-trained with SwAV with the hope that other researchers might also benefit by replacing the ImageNet supervised network with SwAV backbone.
To load our best SwAV pre-trained ResNet-50 model, simply do:
```python
import torch
model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
```

We provide several baseline SwAV pre-trained models with ResNet-50 architecture in torchvision format.
We also provide models pre-trained with [DeepCluster-v2](./main_deepclusterv2.py) and SeLa-v2 obtained by applying improvements from the self-supervised community to [DeepCluster](https://arxiv.org/abs/1807.05520) and [SeLa](https://arxiv.org/abs/1911.05371) (see details in the [appendix of our paper](https://arxiv.org/abs/2006.09882)).

| method | epochs | batch-size | multi-crop | ImageNet top-1 acc. | url | args |
|-------------------|-------------------|---------------------|--------------------|--------------------|--------------------|--------------------|
| SwAV | 800 | 4096 | 2x224 + 6x96 | 75.3 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar) | [script](./scripts/swav_800ep_pretrain.sh) |
| SwAV | 400 | 4096 | 2x224 + 6x96 | 74.6 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_pretrain.pth.tar) | [script](./scripts/swav_400ep_pretrain.sh) |
| SwAV | 200 | 4096 | 2x224 + 6x96 | 73.9 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_pretrain.pth.tar) | [script](./scripts/swav_200ep_pretrain.sh) |
| SwAV | 100 | 4096 | 2x224 + 6x96 | 72.1 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_100ep_pretrain.pth.tar) | [script](./scripts/swav_100ep_pretrain.sh) |
| SwAV | 200 | 256 | 2x224 + 6x96 | 72.7 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_bs256_pretrain.pth.tar) | [script](./scripts/swav_200ep_bs256_pretrain.sh) |
| SwAV | 400 | 256 | 2x224 + 6x96 | 74.3 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_bs256_pretrain.pth.tar) | [script](./scripts/swav_400ep_bs256_pretrain.sh) |
| SwAV | 400 | 4096 | 2x224 | 70.1 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_2x224_pretrain.pth.tar) | [script](./scripts/swav_400ep_2x224_pretrain.sh) |
| DeepCluster-v2 | 800 | 4096 | 2x224 + 6x96 | 75.2 | [model](https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar) | [script](./scripts/deepclusterv2_800ep_pretrain.sh) |
| DeepCluster-v2 | 400 | 4096 | 2x160 + 4x96 | 74.3 | [model](https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_400ep_pretrain.pth.tar) | [script](./scripts/deepclusterv2_400ep_pretrain.sh) |
| DeepCluster-v2 | 400 | 4096 | 2x224 | 70.2 | [model](https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_400ep_2x224_pretrain.pth.tar) | [script](./scripts/deepclusterv2_400ep_2x224_pretrain.sh) |
| SeLa-v2 | 400 | 4096 | 2x160 + 4x96 | 71.8 | [model](https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar) | - |
| SeLa-v2 | 400 | 4096 | 2x224 | 67.2 | [model](https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_2x224_pretrain.pth.tar) | - |

## Larger architectures
We provide SwAV models with ResNet-50 networks where we multiply the width by a factor ×2, ×4, and ×5.
To load the corresponding backbone you can use:
```python
import torch
rn50w2 = torch.hub.load('facebookresearch/swav:main', 'resnet50w2')
rn50w4 = torch.hub.load('facebookresearch/swav:main', 'resnet50w4')
rn50w5 = torch.hub.load('facebookresearch/swav:main', 'resnet50w5')
```

| network | parameters | epochs | ImageNet top-1 acc. | url | args |
|-------------------|---------------------|--------------------|--------------------|--------------------|--------------------|
| RN50-w2 | 94M | 400 | 77.3 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w2_400ep_pretrain.pth.tar) | [script](./scripts/swav_RN50w2_400ep_pretrain.sh) |
| RN50-w4 | 375M | 400 | 77.9 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w4_400ep_pretrain.pth.tar) | [script](./scripts/swav_RN50w4_400ep_pretrain.sh) |
| RN50-w5 | 586M | 400 | 78.5 | [model](https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w5_400ep_pretrain.pth.tar) | - |

## Running times
We provide the running times for some of our runs:
| method | batch-size | multi-crop | scripts | time per epoch |
|---------------------|--------------------|--------------------|--------------------|--------------------|
| SwAV | 4096 | 2x224 + 6x96 | [\*](./scripts/swav_800ep_pretrain.sh) [\*](./scripts/swav_400ep_pretrain.sh) [\*](./scripts/swav_200ep_pretrain.sh) [\*](./scripts/swav_100ep_pretrain.sh) | 3min40s |
| SwAV | 256 | 2x224 + 6x96 | [\*](./scripts/swav_200ep_bs256_pretrain.sh) [\*](./scripts/swav_400ep_bs256_pretrain.sh) | 52min10s |
| DeepCluster-v2 | 4096 | 2x160 + 4x96 | [\*](./scripts/deepclusterv2_400ep_pretrain.sh) | 3min13s |

# Running SwAV unsupervised training

## Requirements
- Python 3.6
- [PyTorch](http://pytorch.org) install = 1.4.0
- torchvision
- CUDA 10.1
- [Apex](https://github.com/NVIDIA/apex) with CUDA extension (see [how I installed apex](https://github.com/facebookresearch/swav/issues/18#issuecomment-748123838))
- Other dependencies: scipy, pandas, numpy

## Singlenode training
SwAV is very simple to implement and experiment with.
Our implementation consists in a [main_swav.py](./main_swav.py) file from which are imported the dataset definition [src/multicropdataset.py](./src/multicropdataset.py), the model architecture [src/resnet50.py](./src/resnet50.py) and some miscellaneous training utilities [src/utils.py](./src/utils.py).

For example, to train SwAV baseline on a single node with 8 gpus for 400 epochs, run:
```
python -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
--data_path /path/to/imagenet/train \
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

## Multinode training
Distributed training is available via Slurm. We provide several [SBATCH scripts](./scripts) to reproduce our SwAV models.
For example, to train SwAV on 8 nodes and 64 GPUs with a batch size of 4096 for 800 epochs run:
```
sbatch ./scripts/swav_800ep_pretrain.sh
```
Note that you might need to remove the copyright header from the sbatch file to launch it.

**Set up `dist_url` parameter**: We refer the user to pytorch distributed documentation ([env](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization) or [file](https://pytorch.org/docs/stable/distributed.html#shared-file-system-initialization) or [tcp](https://pytorch.org/docs/stable/distributed.html#tcp-initialization)) for setting the distributed initialization method (parameter `dist_url`) correctly. In the provided sbatch files, we use the [tcp init method](https://pytorch.org/docs/stable/distributed.html#tcp-initialization) (see [\*](https://github.com/facebookresearch/swav/blob/master/scripts/swav_800ep_pretrain.sh#L17-L20) for example).

# Evaluating models

## Evaluate models: Linear classification on ImageNet
To train a supervised linear classifier on frozen features/weights on a single node with 8 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py \
--data_path /path/to/imagenet \
--pretrained /path/to/checkpoints/swav_800ep_pretrain.pth.tar
```
The resulting linear classifier can be downloaded [here](https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_eval_linear.pth.tar).

## Evaluate models: Semi-supervised learning on ImageNet
To reproduce our results and fine-tune a network with 1% or 10% of ImageNet labels on a single node with 8 gpus, run:
- 10% labels
```
python -m torch.distributed.launch --nproc_per_node=8 eval_semisup.py \
--data_path /path/to/imagenet \
--pretrained /path/to/checkpoints/swav_800ep_pretrain.pth.tar \
--labels_perc "10" \
--lr 0.01 \
--lr_last_layer 0.2
```
- 1% labels
```
python -m torch.distributed.launch --nproc_per_node=8 eval_semisup.py \
--data_path /path/to/imagenet \
--pretrained /path/to/checkpoints/swav_800ep_pretrain.pth.tar \
--labels_perc "1" \
--lr 0.02 \
--lr_last_layer 5
```

## Evaluate models: Transferring to Detection with DETR
[DETR](https://arxiv.org/abs/2005.12872) is a recent object detection framework that reaches competitive performance with Faster R-CNN while being conceptually simpler and trainable end-to-end. We evaluate our SwAV ResNet-50 backbone on object detection on COCO dataset using DETR framework with full fine-tuning. Here are the instructions for reproducing our experiments:

1. [Install detr](https://github.com/facebookresearch/detr#usage---object-detection) and prepare COCO dataset following [these instructions](https://github.com/facebookresearch/detr#data-preparation).

1. Apply the changes highlighted in [this gist](https://gist.github.com/mathildecaron31/bcd03b8864f7ca1aeb89dfe76a118b14#file-backbone-py-L92-L101) to [detr backbone file](https://github.com/facebookresearch/detr/blob/master/models/backbone.py) in order to load SwAV backbone instead of ImageNet supervised weights.

1. Launch training from `detr` repository with [run_with_submitit.py](https://github.com/facebookresearch/detr/blob/master/run_with_submitit.py).
```
python run_with_submitit.py --batch_size 4 --nodes 2 --lr_backbone 5e-5
```

# Common Issues

For help or issues using SwAV, please submit a GitHub issue.

#### The loss does not decrease and is stuck at ln(nmb_prototypes) (8.006 for 3000 prototypes).
It sometimes happens that the system collapses at the beginning and does not manage to converge.
We have found the following empirical workarounds to improve convergence and avoid collapsing at the beginning:
- use a lower epsilon value (`--epsilon 0.03` instead of the default 0.05)
- carefully tune the hyper-parameters
- freeze the prototypes during first iterations (`freeze_prototypes_niters` argument)
- switch to hard assignment
- remove batch-normalization layer from the projection head
- reduce the difficulty of the problem (less crops or softer data augmentation)

We now analyze the collapsing problem: it happens when all examples are mapped to the same unique representation.
In other words, the convnet always has the same output regardless of its input, it is a constant function.
All examples gets the same cluster assignment because they are identical, and the only valid assignment that satisfy the equipartition constraint in this case is the uniform assignment (1/K where K is the number of prototypes).
In turn, this uniform assignment is trivial to predict since it is the same for all examples.
Reducing epsilon parameter (see Eq(3) of our [paper](https://arxiv.org/abs/2006.09882)) encourages the assignments `Q` to be sharper (i.e. less uniform), which strongly helps avoiding collapse.
However, using a too low value for epsilon may lead to numerical instability.

#### Training gets unstable when using the queue.
The queue is composed of feature representations from the previous batches.
[These lines](./main_swav.py#L305-L306) discard the oldest feature representations from the queue and save the newest one (i.e. from the current batch) through a round-robin mechanism.
This way, the assignment problem is performed on more samples: without the queue we assign `B` examples to `num_prototypes` clusters where `B` is the total batch size while with the queue we assign `(B + queue_length)` examples to `num_prototypes` clusters.
This is especially useful when working with small batches because it improves the precision of the assignment.

If you start using the queue too early or if you use a too large queue, this can considerably disturb training: this is because the queue members are too inconsistent.
After introducing the queue the loss should be lower than what it was without the queue.
On the following loss curve (30 first epochs of this [script](./scripts/swav_200ep_bs256_pretrain.sh)) we introduced the queue at epoch 15.
We observe that it made the loss go more down.
<div align="left">
  <img width="35%" alt="SwAV training loss batch_size=256 during the first 30 epochs" src="https://dl.fbaipublicfiles.com/deepcluster/swav_loss_bs256_30ep.png">
</div>

If when introducing the queue, the loss goes up and does not decrease afterwards you should stop your training and change the queue parameters.
We recommend (i) using a smaller queue, (ii) starting the queue later in training.

## License
See the [LICENSE](LICENSE) file for more details.

## See also
[PyTorch Lightning Bolts](https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#swav): Implementation by the Lightning team.

[SwAV-TF](https://github.com/ayulockin/SwAV-TF): A TensorFlow re-implementation.

## Citation
If you find this repository useful in your research, please cite:
```
@article{caron2020unsupervised,
  title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
