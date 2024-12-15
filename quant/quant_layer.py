import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np

class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    :param prob: for qdrop;
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False,
                 scale_method: str = 'minmax',
                 leaf_param: bool = False, prob: float = 1.0, outlier: float = None):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        if self.sym:
            raise NotImplementedError
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = 1.0
        self.zero_point = 0.0
        self.inited = True

        '''if leaf_param, use EMA to set scale'''
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

        '''mse params'''
        self.scale_method = 'mse'
        self.one_side_dist = None
        self.num = 100

        '''for activation quantization'''
        self.running_min = None
        self.running_max = None

        '''do like dropout'''
        self.prob = prob
        self.is_training = False

    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited

    def update_quantize_range(self, x_min, x_max):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        return self.running_min, self.running_max

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)

        # start quantization
        # print(f"!!!x shape={x.shape}, delta shape={self.delta.shape}")
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        if self.is_training and self.prob < 1.0:
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        else:
            x_ans = x_dequant
        return x_ans

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.channel_wise:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = 0, self.n_levels - 1
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, self.eps)
        zero_point = quant_min - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    def quantize(self, x: torch.Tensor, x_max, x_min):
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        if self.channel_wise:
            new_shape = [1] * len(x.shape)
            new_shape[0] = x.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def perform_2D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
            # may also have the one side distribution in some channels
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / (2 ** self.n_bits - 1)
            # enumerate zp
            for zp in range(0, self.n_levels):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                x_q = self.quantize(x, new_max, new_min)
                score = self.lp_loss(x, x_q, 2.4)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    def perform_1D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else thres
            x_q = self.quantize(x, new_max, new_min)
            score = self.lp_loss(x, x_q, 2.4)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def get_x_min_x_max(self, x):
        if self.scale_method != 'mse':
            raise NotImplementedError
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
        if self.one_side_dist != 'no' or self.sym:  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.perform_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.perform_2D_search(x)
        if self.leaf_param:
            return self.update_quantize_range(best_min, best_max)
        return best_min, best_max

    def init_quantization_scale_channel(self, x: torch.Tensor):
        x_min, x_max = self.get_x_min_x_max(x)
        return self.calculate_qparams(x_min, x_max)

    def init_quantization_scale(self, x_clone: torch.Tensor, channel_wise: bool = False):
        if channel_wise:
            # determine the scale and zero point channel-by-channel
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(x_clone.shape)
            new_shape[0] = x_clone.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    @torch.jit.export
    def extra_repr(self):
        return 'bit={}, is_training={}, inited={}'.format(
            self.n_bits, self.is_training, self.inited
        )


class FC2Quantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, outlier: float = None):
        super(FC2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.inited = False
        self.channel_wise = channel_wise
        
    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited
        
    def forward(self, x: torch.Tensor):

        x_min = x.min()
        x = x - x_min + 1e-6
        if self.inited is False:
            self.delta = self.init_quantization_scale(x)
            self.inited = True

        # start quantization
        x_dequant = self.quantize(x, self.delta)
        x_dequant = x_dequant + x_min - 1e-6
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor):
        delta = None
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e+10
        for pct in [0.999, 0.9999, 0.99999]: #
            try:
                new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                new_delta = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
            x_q = self.quantize(x_clone, new_delta)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                delta = new_delta

        return delta
    
    def quantize(self, x, delta):      
        from math import sqrt
        x_int = torch.round(-1 * (x/delta).log2() * 2)
        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        odd_mask = (x_quant%2) * (sqrt(2)-1) + 1
        x_float_q = 2**(-1 * torch.ceil(x_quant/2)) * odd_mask * delta
        x_float_q[mask] = 0
        
        return x_float_q


class TokenQuantizer(nn.Module):
    def __init__(self, n_bits: int = 4, token_wise: bool = True, symmetric: bool = False, qkv_flag: bool=False, outlier: float = None):
        super(TokenQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.token_wise = token_wise
        self.outlier_alpha = outlier
        self.qkv_flag = qkv_flag
    
    def __repr__(self):
        s = super(TokenQuantizer, self).__repr__()
        s = "(" + s + " inited={}, token_wise={}, outlier={})".format(self.inited, self.token_wise, self.outlier_alpha)
        return s
    
    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited

    def forward(self, x: torch.Tensor):
        x_q = x.clone().detach()
        if self.outlier_alpha:
            indices = torch.abs(x_q) > self.outlier_alpha
            if self.qkv_flag:
                num_outliers = torch.sum(indices)
                outlier_ratio = num_outliers.float() / torch.numel(x_q)
                print(f'=========== outlier_alpha is {self.outlier_alpha}, num_outliers is {num_outliers}, total num is {torch.numel(x_q)}, ratio is {outlier_ratio}')
                self.qkv_flag = False
            values = torch.masked_select(x_q, indices)
            x_q[indices] = 0

        if self.inited is False:
            self.delta, self.zero_point  = self.init_quantization_scale(x_q, self.token_wise)
            self.inited = True

        x_int = torch.round(x_q / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        if self.outlier_alpha:
            x_dequant[indices] = values

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, token_wise: bool = False):
        delta, zero_point = None, None
        
        if token_wise:
            x_clone = x.clone().detach()
            n_token = x_clone.shape[1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                # print("44444444")
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                x_min = x_clone.abs().min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 2:
                # print("22222222")
                x_max = x_clone.abs().max(dim=-1)[0]
                x_min = x_clone.abs().min(dim=-1)[0]               
            elif len(x.shape) == 3:
                # print("33333333")
                x_max = x_clone.abs().max(dim=0)[0].max(dim=1)[0]
                x_min = x_clone.abs().min(dim=0)[0].min(dim=1)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # diff = delta - x_min
            # print("diff_shape", diff.shape)
            # print("diff:", diff)
            # diff=diff.cpu().numpy().tolist()
            # res=[str(x_i) for x_i in diff]
            # res=",".join(res)
            # f.write(res)
            # f.write("\n")
            # determine the scale and zero point channel-by-channel
            for t in range(n_token):
                if len(x.shape) == 3:
                    # print("3!!!!!!!!")
                    delta[t], zero_point[t] = self.init_quantization_scale(x_clone[:,t,:], token_wise=False)
                else:
                    # print("eeee!!!!!")
                    delta[t], zero_point[t] = self.init_quantization_scale(x_clone[t], token_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
                
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
                
            elif len(x.shape) == 3:
                delta = delta.view(1, -1, 1)
                zero_point = zero_point.view(1, -1, 1)
                
            else:
                raise NotImplementedError
           
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            
            best_score = 1e+10
            for pct in [0.999, 0.9999, 0.99999]:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   
                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point
    
    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class OutlierQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, symmetric: bool = False):
        super(OutlierQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.outlier_alpha = 2.0
    
    def __repr__(self):
        s = super(OutlierQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s
    
    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited

    def forward(self, x: torch.Tensor):
        # shape = x.shape
        # x = x.reshape(-1)
        # left = 0.01
        # left_num = int(len(x) * left / 2)
        # value1, indices1 = torch.topk(x, left_num, largest=False)
        # value2, indices2 = torch.topk(x, left_num, largest=True)
        # values = torch.cat((value1, value2), dim=0)
        # indices = torch.cat((indices1, indices2), dim=0)

        # x = x.index_fill_(0, indices, 0)

        # x = x.reshape(shape)
        # if self.inited is False:
        #     self.delta, self.zero_point  = self.init_quantization_scale(x, self.channel_wise)
        #     self.inited = True

        # x_int = torch.round(x / self.delta) + self.zero_point
        # x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        # x_dequant = (x_quant - self.zero_point) * self.delta

        # x_dequant = x_dequant.reshape(-1)
        # x_dequant[indices] = values
        # x_dequant = x_dequant.reshape(shape)
        # return x_dequant

        # print(x)
        # torch.save(x, "fc1_tensor1.pt")
        # condition = torch.gt(x, 2)
        # indices = torch.nonzero(condition)
        # print(indices.shape)
        # print(indices)
        # print(x.shape)
        # # torch.save(x, "fc1_tensor.pt")
        # # values = x[indices]
        # x[indices] = 0

        indices = torch.abs(x) > self.outlier_alpha
        values = torch.masked_select(x, indices)
        x[indices] = 0
        # condition = torch.abs(x) > 2
        # values = torch.masked_select(x, condition)
        # indices = torch.nonzero(condition)
        # x[indices] = 0

        if self.inited is False:
            self.delta, self.zero_point  = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        x_dequant[indices] = values

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                x_min = x_clone.abs().min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
                x_min = x_clone.abs().min(dim=-1)[0]               
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                x_min = x_clone.abs().min(dim=0)[0].min(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # diff = delta - x_min
            # print("diff_shape", diff.shape)
            # print("diff:", diff)
            # diff=diff.cpu().numpy().tolist()
            # res=[str(x_i) for x_i in diff]
            # res=",".join(res)
            # f.write(res)
            # f.write("\n")
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
                
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
                
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
                
            else:
                raise NotImplementedError
           
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            
            best_score = 1e+10
            for pct in [0.999, 0.9999, 0.99999]:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   
                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point
    
    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, symmetric: bool = False, outlier: float = None):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
    
    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s
    
    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.delta, self.zero_point  = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        # start quantization
        x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[-1] if len(x.shape) == 3 else x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                x_min = x_clone.abs().min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 2:
                x_max = x_clone.abs().max(dim=-1)[0]
                x_min = x_clone.abs().min(dim=-1)[0]               
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=0)[0].max(dim=0)[0]
                x_min = x_clone.abs().min(dim=0)[0].min(dim=0)[0]
            else:
                raise NotImplementedError

            delta = x_max.clone()
            zero_point = x_max.clone()
            # diff = delta - x_min
            # print("diff_shape", diff.shape)
            # print("diff:", diff)
            # diff=diff.cpu().numpy().tolist()
            # res=[str(x_i) for x_i in diff]
            # res=",".join(res)
            # f.write(res)
            # f.write("\n")
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                if len(x.shape) == 3:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[:,:,c], channel_wise=False)
                else:
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
                
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
                
            elif len(x.shape) == 3:
                delta = delta.view(1, 1, -1)
                zero_point = zero_point.view(1, 1, -1)
                
            else:
                raise NotImplementedError
           
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            
            best_score = 1e+10
            for pct in [0.999, 0.9999, 0.99999]:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   
                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point
    
    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q
    

class LogSqrt2Quantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False, outlier: float = None):
        super(LogSqrt2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.inited = False
        self.channel_wise = channel_wise

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.delta = self.init_quantization_scale(x)
            self.inited = True

        # start quantization
        x_dequant = self.quantize(x, self.delta)
        return x_dequant
    
    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited
        
    def init_quantization_scale(self, x: torch.Tensor):
        delta = None
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e+10
        for pct in [0.999, 0.9999, 0.99999]: #
            try:
                new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                new_delta = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
            x_q = self.quantize(x_clone, new_delta)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                delta = new_delta

        return delta

    def quantize(self, x, delta):      
        from math import sqrt
        x_int = torch.round(-1 * (x/delta).log2() * 2)
        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        odd_mask = (x_quant%2) * (sqrt(2)-1) + 1
        x_float_q = 2**(-1 * torch.ceil(x_quant/2)) * odd_mask * delta
        x_float_q[mask] = 0   
        
        return x_float_q


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant=False):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.outlier_flag = False
        # initialize quantizer
        self.weight_quantizer = UniformQuantizer(**weight_quant_params)
        if "fc2_flag" in act_quant_params.keys():
            act_quant_params.pop("fc2_flag")
            self.act_quantizer = FC2Quantizer(**act_quant_params)
        elif "fc1_flag" in act_quant_params.keys():
            act_quant_params.pop("fc1_flag")
            act_quant_params.pop("channel_wise")
            self.act_quantizer = TokenQuantizer(**act_quant_params)
        elif "qkv_flag" in act_quant_params.keys():
            act_quant_params.pop("qkv_flag")
            act_quant_params.pop("channel_wise")
            self.act_quantizer = TokenQuantizer(**act_quant_params)
            self.act_quantizer.outlier_alpha = 5
        else:
            self.act_quantizer = UniformQuantizer(**act_quant_params)

        self.norm_function = StraightThrough()
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.disable_act_quant = disable_act_quant
        self.trained = False


    def forward(self, input: torch.Tensor):
        # print("orign weight shape:", self.weight.shape)
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        if self.use_act_quant:
            input = self.act_quantizer(input)
        # print("input shape:", input.shape)
        # print("weight shape:", weight.shape)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        out = self.norm_function(out)
        out = self.activation_function(out)
        # if self.disable_act_quant:
        #     return out
        # if self.use_act_quant:
        #     out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    @torch.jit.export
    def extra_repr(self):
        return 'wbit={}, abit={}, disable_act_quant={}'.format(
            self.weight_quantizer.n_bits, self.act_quantizer.n_bits, self.disable_act_quant
        )


class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


class QuantMatMul(nn.Module):
    """
    Class to quantize weights of given Linear layer
    """
    def __init__(self, org_module: MatMul,
                 act_quant_params: dict = {}, disable_act_quant=False):
        super(QuantMatMul, self).__init__()
         # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        # self.quantizer_A = UniformAffineQuantizer(**act_quant_params)
        if "mm2_flag" in act_quant_params.keys():
            act_quant_params.pop('mm2_flag')
            self.quantizer_A = LogSqrt2Quantizer(**act_quant_params)
            # self.quantizer_A = UniformQuantizer(**act_quant_params)
        else:
            self.quantizer_A = UniformAffineQuantizer(**act_quant_params)
        self.quantizer_B = UniformAffineQuantizer(**act_quant_params)

        self.ignore_reconstruction = False
        self.disable_act_quant = disable_act_quant
        self.trained = False
        
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    # @torch.jit.export
    # def extra_repr(self):
    #     return 'wbit={}, abit={}, disable_act_quant={}'.format(
    #         self.weight_quantizer.n_bits, self.act_quantizer.n_bits, self.disable_act_quant
    #     )  
    
    def forward(self, A, B):
        if self.use_act_quant:
            A = self.quantizer_A(A)
            B = self.quantizer_B(B)
        out = A @ B
        # if self.use_act_quant:
        #     out = self.quantizer_A(out)
        return out 