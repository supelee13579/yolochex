import os
from argparse import ArgumentParser
import numpy as np

import torch
import torch.nn as nn
import torch.functional as F
from torchsummary import summary

from copy import deepcopy


def generate_mean_std(opt):
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean_val).cuda()
    std = torch.tensor(std_val).cuda()

    view = [1, len(mean_val), 1, 1]

    mean = mean.view(*view)
    std = std.view(*view)

    if opt.amp:
        mean = mean.half()
        std = std.half()

    return mean, std


def L1_norm(layer):
    weight_copy = layer.weight.data.abs().clone().cpu().numpy()
    norm = np.sum(weight_copy, axis=(1, 2, 3))
    return norm


def Laplacian(layer):
    weight = layer.weight.data.detach()
    x = weight.view(weight.shape[0], -1)
    X_inner = torch.matmul(x, x.t())
    X_norm = torch.diag(X_inner, diagonal=0)
    X_dist_sq = X_norm + torch.reshape(X_norm, [-1, 1]) - 2 * X_inner
    X_dist = torch.sqrt(X_dist_sq)
    laplace = torch.sum(X_dist, dim=0).cpu().numpy()
    return laplace


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = x**2 + y**2
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


class SI(nn.Module):
    def __init__(self, inp, k_sobel):
        super(SI, self).__init__()

        self.inp = inp

        sobel_2D = get_sobel_kernel(k_sobel)
        sobel_2D_trans = sobel_2D.T
        sobel_2D = torch.from_numpy(sobel_2D).cuda().half()
        sobel_2D_trans = torch.from_numpy(sobel_2D_trans).cuda().half()
        sobel_2D = sobel_2D.unsqueeze(0).repeat(inp, 1, 1, 1)
        sobel_2D_trans = sobel_2D_trans.unsqueeze(0).repeat(inp, 1, 1, 1)

        self.vars = nn.ParameterList()
        self.vars.append(nn.Parameter(sobel_2D, requires_grad=False))
        self.vars.append(nn.Parameter(sobel_2D_trans, requires_grad=False))

    def forward(self, x):
        grad_x = F.conv2d(x, self.vars[0], bias=None, stride=1, padding=1, dilation=1, groups=self.inp)
        grad_y = F.conv2d(x, self.vars[1], bias=None, stride=1, padding=1, dilation=1, groups=self.inp)
        value = torch.sqrt(grad_x**2 + grad_y**2)
        # value = 1/1.4142 * (torch.abs(grad_x) + torch.abs(grad_y))
        denom = value.shape[2] * value.shape[3]
        out = torch.sum(value**2, dim=(2, 3)) / denom - (torch.sum(value, dim=(2, 3)) / denom) ** 2
        return out**0.5


def SI_pruning(model, data_loader, mean, std):
    model = deepcopy(model.feature_extractor)

    list_conv = []

    def conv_hook(self, input, output):
        SIfeature = SI(output.shape[1], 3)
        list_conv.append(SIfeature(output))

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            if idx >= 100:
                break
            img = data[0][0][0]
            img.sub_(mean).div_(std)
            img = img.cuda()
            model(img)
            if idx == 0:
                score = [torch.mean(m, dim=0, keepdim=True) for m in list_conv]
            else:
                temp = [torch.mean(m, dim=0, keepdim=True) for m in list_conv]
                score = [x + y for x, y in zip(score, temp)]
            list_conv = []
    full_score = [m.squeeze(0).detach().cpu().numpy().tolist() for m in score]
    full_rank = [np.argsort(m) for m in full_score]

    l1 = [2, 6, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 41, 44, 48, 51]
    l2 = (np.asarray(l1) + 1).tolist()
    l3 = (np.asarray(l2) + 1).tolist()
    skip = [5, 15, 28, 47]
    layer_id = 1
    score = []
    rank = []
    for m in model:
        # if str(m) == "FeatureConcat()" or str(m) == "FeatureConcat_l()" or \
        #       str(m) == "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)":
        #     continue
        # for m_ in m:
        if isinstance(m, nn.Conv2d):
            if layer_id in l1 + l2 + skip:
                score.append(full_score[layer_id - 1])
                rank.append(full_rank[layer_id - 1])
                layer_id += 1
                continue
            layer_id += 1
    return score, rank


def get_layer_ratio(model, sparsity):
    model = model.feature_extractor
    l1 = [2, 6, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 41, 44, 48, 51]
    l2 = (np.asarray(l1) + 1).tolist()
    l3 = (np.asarray(l2) + 1).tolist()
    skip = [5, 15, 28, 47]
    total = 0
    bn_count = 1
    for m in model:
        # if str(m) == "FeatureConcat()" or str(m) == "FeatureConcat_l()" or \
        #       str(m) == "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)":
        #     continue
        # for m_ in m:
        if isinstance(m, nn.BatchNorm2d):
            # print(f"m : {m} , m : {m}")
            if bn_count in l2 + l2 + skip:
                # print(f"m : {m} , m : {m} , bn_count : {bn_count}")
                total += m.weight.data.shape[0]
                # print(f'total : {total}')
                bn_count += 1
                continue
            bn_count += 1
            # print(f"m : {m} , m : {m} , bn_count : {bn_count}")
        bn = torch.zeros(total)
        index = 0
        bn_count = 1
    # print("completed first for_loop")
    for m in model:
        # if str(m) == "FeatureConcat()" or str(m) == "FeatureConcat_l()" or \
        #       str(m) == "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)":
        #     continue
        # for m_ in m:
        if isinstance(m, nn.BatchNorm2d):
            if bn_count in l1 + l2 + skip:
                size = m.weight.data.shape[0]
                bn[index : (index + size)] = m.weight.data.abs().clone()
                index += size
                bn_count += 1
                continue
            bn_count += 1
        y, i = torch.sort(bn)
        thre_index = int(total * sparsity)
        thre = y[thre_index]
        layer_ratio = []
        bn_count = 1
    # print("completed second for_loop")
    for m in model:
        # if str(m) == "FeatureConcat()" or str(m) == "FeatureConcat_l()" or \
        #       str(m) == "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)":
        #     continue
        # for m_ in m:
        if isinstance(m, nn.BatchNorm2d):
            if bn_count in l1 + l2 + skip:
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                layer_ratio.append((mask.shape[0] - torch.sum(mask).item()) / mask.shape[0])
                # print(layer_ratio)
                # print("!!!!")
                bn_count += 1
                continue
            bn_count += 1
    # print("completed third for_loop")
    return layer_ratio


def regrow_allocation(model, delta_sparsity, layer_ratio_down):
    model = model.feature_extractor
    l1 = [2, 6, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 41, 44, 48, 51]
    l2 = (np.asarray(l1) + 1).tolist()
    l3 = (np.asarray(l2) + 1).tolist()
    skip = [5, 15, 28, 47]
    bn_count = 1
    idx = 0
    layer_ratio = []
    for m in model:
        # if str(m) == "FeatureConcat()" or str(m) == "FeatureConcat_l()" or \
        #       str(m) == "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)":
        #     continue
        # for m_ in m:
        if isinstance(m, nn.BatchNorm2d):
            out_channel = m.weight.data.shape[0]
            if bn_count in l1 + l2 + skip:
                num_remain = out_channel * (1 - layer_ratio_down[idx])
                num_regrow = int(delta_sparsity * out_channel)
                num_prune = out_channel - num_remain - num_regrow
                if num_prune <= 0:
                    num_prune = 0
                layer_ratio.append(num_prune / out_channel)
                idx += 1
                bn_count += 1
                continue
            bn_count += 1
    return layer_ratio


def init_mask(model, ratio):
    model = model.feature_extractor
    
    print(model)

    prev_model = deepcopy(model)
    l1 = [2, 6, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 41, 44, 48, 51]
    l2 = (np.asarray(l1) + 1).tolist()
    l3 = (np.asarray(l2) + 1).tolist()
    skip = [5, 15, 28, 47]
    layer_id = 1
    cfg_mask = []
    for m in model:
        # if str(m) == "FeatureConcat()" or str(m) == "FeatureConcat_l()" or \
        #       str(m) == "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)":
        #     continue
        # for m_ in m:
        # print(m)
        if isinstance(m, nn.Conv2d):
            # print(f'm.weight.data.shape : {m.weight.data.shape}')
            out_channels = m.weight.data.shape[0]
            if layer_id in l1 + l2 + skip:
                num_keep = int(out_channels * (1 - ratio))
                rank = np.argsort(L1_norm(m))
                arg_max_rev = rank[::-1][:num_keep]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                # print(f'cfg_mask : {cfg_mask}')
                layer_id += 1
                continue
            layer_id += 1
    return cfg_mask, prev_model


def update_mask(model, layer_ratio_up, layer_ratio_down, old_model, Rank_):
    model = model.feature_extractor
    l1 = [2, 6, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 41, 44, 48, 51]
    l2 = (np.asarray(l1) + 1).tolist()
    l3 = (np.asarray(l2) + 1).tolist()
    skip = [5, 15, 28, 47]
    layer_id = 1
    idx = 0
    cfg_mask = []
    for [m, m0] in zip(model, old_model):
        # if str(m) == "FeatureConcat()" or str(m) == "FeatureConcat_l()" or \
        #       str(m) == "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)":
        #     continue
        # for m_ in m:
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in l1:
                num_keep = int(out_channels * (1 - layer_ratio_down[idx]))
                num_free = int(out_channels * (1 - layer_ratio_up[idx])) - num_keep
                rank = Rank_[idx]
                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                grow = np.random.permutation(freedom)[:num_free]
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask.append(mask)

                # most recently used weights copy
                copy_idx = np.where(L1_norm(m) == 0)[0]
                w = m0.weight.data[copy_idx.tolist(), :, :, :].clone()
                m.weight.data[copy_idx.tolist(), :, :, :] = w.clone()

                layer_id += 1
                idx += 1
                continue
            if layer_id in l2:
                num_keep = int(out_channels * (1 - layer_ratio_down[idx]))
                num_free = int(out_channels * (1 - layer_ratio_up[idx])) - num_keep
                rank = Rank_[idx]
                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                grow = np.random.permutation(freedom)[:num_free]
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask.append(mask)

                # most recently used weights copy
                prev_copy_idx = deepcopy(copy_idx)
                copy_idx = np.where(L1_norm(m) == 0)[0]
                w = m0.weight.data[:, prev_copy_idx.tolist(), :, :].clone()
                m.weight.data[:, prev_copy_idx.tolist(), :, :] = w.clone()
                w = m0.weight.data[copy_idx.tolist(), :, :, :].clone()
                m.weight.data[copy_idx.tolist(), :, :, :] = w.clone()

                layer_id += 1
                idx += 1
                continue
            if layer_id in l3:
                # most recently used weights copy
                w = m0.weight.data[:, copy_idx.tolist(), :, :].clone()
                m.weight.data[:, copy_idx.tolist(), :, :] = w.clone()

                layer_id += 1
                continue
            if layer_id in skip:
                num_keep = int(out_channels * (1 - layer_ratio_down[idx]))
                num_free = int(out_channels * (1 - layer_ratio_up[idx])) - num_keep
                rank = Rank_[idx]
                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                grow = np.random.permutation(freedom)[:num_free]
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask.append(mask)

                # most recently used weights copy
                copy_idx = np.where(L1_norm(m) == 0)[0]
                w = m0.weight.data[copy_idx.tolist(), :, :, :].clone()
                m.weight.data[copy_idx.tolist(), :, :, :] = w.clone()

                layer_id += 1
                idx += 1
                continue
            layer_id += 1
        elif isinstance(m, nn.BatchNorm2d):
            if layer_id - 1 in l1 + l2 + skip:
                w = m0.weight.data[copy_idx.tolist()].clone()
                m_.weight.data[copy_idx.tolist()] = w.clone()
                b = m0.bias.data[copy_idx.tolist()].clone()
                m_.bias.data[copy_idx.tolist()] = b.clone()
                rm = m0.running_mean[copy_idx.tolist()].clone()
                m_.running_mean[copy_idx.tolist()] = rm.clone()
                rv = m0.running_var[copy_idx.tolist()].clone()
                m_.running_var[copy_idx.tolist()] = rv.clone()
                continue
    prev_model = deepcopy(model)
    return cfg_mask, prev_model


def apply_mask(model, cfg_mask):
    model = model.feature_extractor
    l1 = [2, 6, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 41, 44, 48, 51]
    l2 = (np.asarray(l1) + 1).tolist()
    l3 = (np.asarray(l2) + 1).tolist()
    skip = [5, 15, 28, 47]
    layer_id_in_cfg = 0
    conv_count = 1
    for m in model:  
        # if str(m) == "FeatureConcat()" or str(m) == "FeatureConcat_l()" or \
        #       str(m) == "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)":
        #     continue
        # for m_ in m:
        if isinstance(m, nn.Conv2d):
            if conv_count in l1:
                # print(f'conv_count_l1 : {conv_count}')
                # print(f'm : {str(m)}')                
                mask = cfg_mask[layer_id_in_cfg].float().cuda()
                mask = mask.view(m_.weight.data.shape[0], 1, 1, 1)
                m_.weight.data.mul_(mask)
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count in l2:
                # print(f'conv_count_l2 : {conv_count}')
                # print(f'm : {str(m)}')                
                mask = cfg_mask[layer_id_in_cfg].float().cuda()
                mask = mask.view(m.weight.data.shape[0], 1, 1, 1)
                m.weight.data.mul_(mask)
                prev_mask = cfg_mask[layer_id_in_cfg - 1].float().cuda()
                # print(f'prev_mask : {prev_mask}')
                prev_mask = prev_mask.view(1, m.weight.data.shape[1], 1, 1)
                # print(f'prev_mask : {prev_mask}')
                m.weight.data.mul_(prev_mask)
                # print(f'm_.weight.data.mul_ : {m_.weight.data.mul_(prev_mask)}')
                layer_id_in_cfg += 1
                # print(f'layer : {conv_count}')
                conv_count += 1
                continue
            if conv_count in l3:
                # print(f'conv_count_l3 : {conv_count}')
                # print(f'm : {str(m)}')                
                prev_mask = cfg_mask[layer_id_in_cfg - 1].float().cuda()
                prev_mask = prev_mask.view(1, m.weight.data.shape[1], 1, 1)
                m.weight.data.mul_(prev_mask)
                conv_count += 1
                continue
            if conv_count in skip:
                mask = cfg_mask[layer_id_in_cfg].float().cuda()
                mask = mask.view(m.weight.data.shape[0], 1, 1, 1)
                m.weight.data.mul_(mask)
                layer_id_in_cfg += 1
                conv_count += 1
                continue
            conv_count += 1
        elif isinstance(m, nn.BatchNorm2d):
            if conv_count in l2:
                mask = cfg_mask[layer_id_in_cfg - 1].float().cuda()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                continue
            if conv_count in l3:
                mask = cfg_mask[layer_id_in_cfg - 1].float().cuda()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                continue
            if conv_count - 1 in skip:
                mask = cfg_mask[layer_id_in_cfg - 1].float().cuda()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                continue

def detect_channel_zero(model):
    model = model.feature_extractor
    l1 = [2, 6, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 41, 44, 48, 51]
    l2 = (np.asarray(l1) + 1).tolist()
    l3 = (np.asarray(l2) + 1).tolist()
    skip = [5, 15, 28, 47]
    total_zero = 0
    total_c = 0
    conv_count = 1
    for m in model:
        # if str(m) == "FeatureConcat()" or str(m) == "FeatureConcat_l()" or \
        #       str(m) == "MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)":
        #     continue
        # for m_ in m:
        if isinstance(m, nn.Conv2d):
            if conv_count in l1 + l2 + skip:
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                norm = np.sum(weight_copy, axis=(1, 2, 3))
                total_zero += len(np.where(norm == 0)[0])
                total_c += m.weight.data.shape[0]
                conv_count += 1
                continue
            conv_count += 1
        return total_zero / total_c


def make_parser():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector" " on COCO")
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="../../coco2017",
        required=True,
        help="path to test and training data files",
    )
    parser.add_argument("--epochs", "-e", type=int, default=65, help="number of epochs for training")
    parser.add_argument("--batch-size", "--bs", type=int, default=32, help="number of examples for each iteration")
    parser.add_argument(
        "--eval-batch-size",
        "--ebs",
        type=int,
        default=32,
        help="number of examples for each evaluation iteration",
    )
    parser.add_argument("--no-cuda", action="store_true", help="use available GPUs")
    parser.add_argument("--seed", "-s", type=int, help="manually set random seed for torch")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to model checkpoint file")
    parser.add_argument("--save", type=str, default=None, help="save model checkpoints in the specified directory")
    parser.add_argument(
        "--mode",
        type=str,
        default="training",
        choices=["training", "evaluation", "benchmark-training", "benchmark-inference"],
    )
    parser.add_argument(
        "--evaluation",
        nargs="*",
        type=int,
        default=[21, 31, 37, 42, 48, 53, 59, 64],
        help="epochs at which to evaluate",
    )
    parser.add_argument(
        "--multistep",
        nargs="*",
        type=int,
        default=[43, 54],
        help="epochs at which to decay learning rate",
    )

    # Hyperparameters
    parser.add_argument("--learning-rate", "--lr", type=float, default=2.6e-3, help="learning rate")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="momentum argument for SGD optimizer")
    parser.add_argument(
        "--weight-decay",
        "--wd",
        type=float,
        default=0.0005,
        help="momentum argument for SGD optimizer",
    )

    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=20,
        metavar="N",
        help="Run N iterations while benchmarking (ignored when training and validation)",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=20,
        metavar="N",
        help="Number of warmup iterations for benchmarking",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    )
    parser.add_argument(
        "--backbone-path",
        type=str,
        default=None,
        help="Path to chekcpointed backbone. It should match the"
        " backbone model declared with the --backbone argument."
        " When it is not provided, pretrained model from torchvision"
        " will be downloaded.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default=None,
        help="If provided, the json summary will be written to" "the specified file.",
    )

    # Distributed
    parser.add_argument(
        "--local_rank",
        default=os.getenv("LOCAL_RANK", 0),
        type=int,
        help="Used for multi-process training. Can either be manually set "
        + "or automatically set by using 'python -m multiproc'.",
    )

    return parser
