import torch.nn as nn
from .quant_layer import QuantModule, UniformAffineQuantizer, QuantMatMul

from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention
from timm.models.layers.mlp import Mlp
from timm.models.layers import DropPath
from copy import deepcopy

class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        self.ignore_reconstruction = False
        self.trained = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantModule, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant)


# Original Attention -> QuantAttention
class QuantAttention(BaseQuantBlock):
    """
    Implementation of Quantized AttentionBlock used in vit.
    """
    def __init__(self, basic_atten:Attention, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        
        act_quant_params_qkv = deepcopy(act_quant_params)
        act_quant_params_proj = deepcopy(act_quant_params)
        act_quant_params_qkv["qkv_flag"] = True
        weight_quant_params_qkv = deepcopy(weight_quant_params)
        act_quant_params_matmul2= deepcopy(act_quant_params)
        act_quant_params_matmul2['mm2_flag'] = True
        self.qkv = QuantModule(basic_atten.qkv, weight_quant_params_qkv, act_quant_params_qkv)
        self.attn_drop = basic_atten.attn_drop
        self.proj = QuantModule(basic_atten.proj, weight_quant_params, act_quant_params_proj)
        self.proj_drop = basic_atten.proj_drop
        self.matmul1 = QuantMatMul(basic_atten.matmul1, act_quant_params)
        self.matmul2 = QuantMatMul(basic_atten.matmul2, act_quant_params_matmul2)
        self.num_heads = basic_atten.num_heads
        self.scale = basic_atten.scale
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # del q, k

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
        # del attn, v
        x = self.proj(x)
        out = self.proj_drop(x)
        # if self.use_act_quant:
        #     out = self.act_quantizer(out)
        return (out, attn)


class QuantWindowAttention(BaseQuantBlock):
    """
    Implementation of Quantized AttentionBlock used in vit.
    """
    def __init__(self, basic_atten:WindowAttention, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        act_quant_params_qkv = deepcopy(act_quant_params)
        act_quant_params_qkv["qkv_flag"] = True
        act_quant_params_qkv['channel_wise']  = False
        act_quant_params_matmul2= deepcopy(act_quant_params)
        act_quant_params_matmul2['mm2_flag'] = True
        self.qkv = QuantModule(basic_atten.qkv, weight_quant_params, act_quant_params_qkv)
        self.attn_drop = basic_atten.attn_drop
        self.proj = QuantModule(basic_atten.proj, weight_quant_params, act_quant_params)
        self.proj_drop = basic_atten.proj_drop
        self.matmul1 = QuantMatMul(basic_atten.matmul1, act_quant_params)
        self.matmul2 = QuantMatMul(basic_atten.matmul2, act_quant_params_matmul2)
        self.num_heads = basic_atten.num_heads
        self.scale = basic_atten.scale
        self.softmax = basic_atten.softmax
        self.window_size = basic_atten.window_size
        self.relative_position_index = basic_atten.relative_position_index
        self.relative_position_bias_table = basic_atten.relative_position_bias_table
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x, mask = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))
        attn = self.matmul1(q, k.transpose(-2,-1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return (x, attn)
    

class QuantMlp(BaseQuantBlock):
    """
    Implementation of Quantized MlpBlock used in vit.
    """
    def __init__(self, basic_mlp: Mlp, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        act_quant_params_fc1 = deepcopy(act_quant_params)
        weight_quant_params_fc1 = deepcopy(weight_quant_params)
        act_quant_params_fc1["fc1_flag"] = True
        act_quant_params_fc2 = deepcopy(act_quant_params)
        weight_quant_params_fc2 = deepcopy(weight_quant_params)
        act_quant_params_fc2['fc2_flag'] = True

        self.fc1 = QuantModule(basic_mlp.fc1, weight_quant_params_fc1, act_quant_params_fc1)
        self.act = basic_mlp.act
        self.fc2 = QuantModule(basic_mlp.fc2, weight_quant_params_fc2, act_quant_params_fc2)
        # self.fc2 = basic_mlp.fc2
        self.drop1 = basic_mlp.drop
        self.drop2 = basic_mlp.drop
        # self.norm = basic_mlp.norm
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        # x = self.norm(x)
        x = self.fc2(x)
        out = self.drop2(x)
        return out
    

specials = {
    Attention: QuantAttention,
    Mlp: QuantMlp,
    WindowAttention: QuantWindowAttention,
    
}

specials_unquantized = [nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.Dropout, nn.LayerNorm, nn.AdaptiveAvgPool1d, DropPath,]
