import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import random
import time
import copy

from quant import (
    block_reconstruction,
    layer_reconstruction,
    set_act_quant_block,
    set_act_quant,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
)
from build_model import build_model
from build_dataset import *

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def get_train_samples(train_loader, num_samples):
    train_data, target = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        target.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples], torch.cat(target, dim=0)[:num_samples]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general parameters for data and model
    parser.add_argument("--model", default="deit_small",
                        choices=['vit_small', 'vit_base',
                            'deit_tiny', 'deit_small', 'deit_base', 
                            'swin_tiny', 'swin_small', 'swin_base'],
                        help="model")
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='model name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument("--calib-batchsize", default=32,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--val-batchsize", default=200,
                        type=int, help="batchsize of validation set")
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--dataset', default='/datasets-to-imagenet', type=str, help='path to ImageNet data')

    # quantization parameters
    parser.add_argument('--n_bits_w', default=4, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', default=True, help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--outlier', default=5, type=float, help='outlier value for per-token quantization')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=5000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--keep_cpu', action='store_true', help='keep the calibration data on cpu')

    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')

    # activation calibration parameters
    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate for LSQ')

    parser.add_argument('--init_wmode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for weight')
    parser.add_argument('--init_amode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for activation')

    parser.add_argument('--prob', default=0.5, type=float)
    parser.add_argument('--input_prob', default=0.5, type=float)
    parser.add_argument('--lamb_r', default=0.1, type=float, help='hyper-parameter for regularization')
    parser.add_argument('--T', default=4.0, type=float, help='temperature coefficient for KL divergence')
    parser.add_argument('--bn_lr', default=1e-3, type=float, help='learning rate for DC')
    parser.add_argument('--lamb_c', default=0.02, type=float, help='hyper-parameter for DC')
    args = parser.parse_args()

    seed_all(args.seed)
    
    print("Building dataset ...")
    train_loader, test_loader = build_dataset(args)
    model_zoo = {
        'vit_small' : 'vit_small_patch16_224',
        'vit_base' : 'vit_base_patch16_224',

        'deit_tiny' : 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base' : 'deit_base_patch16_224',

        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base' : 'swin_base_patch4_window7_224',
    }
    
    # load model
    print('Building model ...')
    model = build_model(model_zoo[args.model])
    model.cuda()
    model.eval()
  
    fp_model = copy.deepcopy(model)
    fp_model.cuda()
    fp_model.eval()
    
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'outlier': args.outlier}
    
    fp_model = QuantModel(model=fp_model, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    fp_model.cuda()
    fp_model.eval()
    for name,module in fp_model.named_modules():
        if name=="model.layers.0.blocks.0" or name == "model.layers.0.blocks.1" or name == "model.layers.1.blocks.0" or name == "model.layers.1.blocks.1":
            print(f"{name},{module.window_size},{module.shift_size}")
    fp_model.set_quant_state(False, False)

    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()
    qnn.disable_network_output_quantization()
    print('the quantized model is below!')
    print(qnn)
    cali_data, cali_target = get_train_samples(train_loader, num_samples=args.num_samples)
    device = next(qnn.parameters()).device

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight,
                b_range=(args.b_start, args.b_end), warmup=args.warmup, opt_mode='mse',
                lr=args.lr, input_prob=args.input_prob, keep_gpu=not args.keep_cpu, 
                lamb_r=args.lamb_r, T=args.T, bn_lr=args.bn_lr, lamb_c=args.lamb_c)


    '''init weight quantizer'''
    set_weight_quantize_params(qnn)
    # cali_data.to(device)
    # qnn.set_quant_state(weight_quant=True, act_quant=True)
    # with torch.no_grad():
    #     _ = qnn(cali_data)

    def set_weight_act_quantize_params(module, fp_module):
        if isinstance(module, QuantModule):
            layer_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            block_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
        else:
            raise NotImplementedError
        
    def set_act_quantize_params(module):
        if isinstance(module, QuantModule):
            set_act_quant(qnn, module, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            set_act_quant_block(qnn, module, **kwargs)
        else:
            raise NotImplementedError
        
    def recon_model(model: nn.Module, fp_model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for (name, module), (_, fp_module) in zip(model.named_children(), fp_model.named_children()):
            if isinstance(module, QuantModule):
                print('Reconstruction for layer {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            elif isinstance(module, BaseQuantBlock):
                print('Reconstruction for block {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            else:
                recon_model(module, fp_module)
                
    def calib_model(model: nn.Module):
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                print('calib for layer {}'.format(name))
                set_act_quantize_params(module)
            elif isinstance(module, BaseQuantBlock):
                print('calib for block {}'.format(name))
                set_act_quantize_params(module)
            else:
                calib_model(module)
    # Start calibration
    recon_model(qnn, fp_model)
    # calib_model(qnn)

    print(f"=========================== quant process costs {(e1-s1):.2f} s")

    qnn.set_quant_state(weight_quant=True, act_quant=True)
    print('Full quantization (W{}A{}) accuracy: {} for model {}'.format(args.n_bits_w, args.n_bits_a,
                                                           validate_model(test_loader, qnn), args.model))
