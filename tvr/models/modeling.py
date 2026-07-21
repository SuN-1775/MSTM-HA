import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import CrossModel, Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, ArcCrossEn, KL
import math
import numpy as np
# from .query_cross_att import DecoderLayer
# from .transformer_block import EncoderLayer

# ============= MSTE Mamba模块导入 =============
try:
    from mamba_ssm import Mamba
    from einops import rearrange
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm or einops not installed. MSTE mode will not work.")
    print("Install with: pip install mamba-ssm einops")
# ============= 结束导入 =============

# ============= HCA几何库导入 =============
try:
    import geoopt
    from geoopt import ManifoldParameter
    from geoopt.manifolds import PoincareBall
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False
    PoincareBall = None
    ManifoldParameter = None
    print("Warning: geoopt not installed. HCA branch will not be available.")
    print("Install with: pip install geoopt")
# ============= 结束HCA几何库导入 =============

allgather = AllGather.apply
allgather2 = AllGather2.apply

def compute_selfattention(transformer_encoder,x,mask,src_key_padding_mask,i_layer,d_model,num_heads):
    h = F.linear(x, transformer_encoder.layers[i_layer].self_attn.in_proj_weight, bias=transformer_encoder.layers[i_layer].self_attn.in_proj_bias)
    qkv = h.reshape(x.shape[0], x.shape[1], num_heads, 3 * d_model//num_heads)
    qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
    q, k, v = qkv.chunk(3, dim=-1) # [Batch, Head, SeqLen, d_head=d_model//num_heads]
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) # [Batch, Head, SeqLen, SeqLen]
    d_k = q.size()[-1]
    attn_probs = attn_logits / math.sqrt(d_k)
    # combining src_mask e.g. upper triangular with src_key_padding_mask e.g. columns over each padding position
    combined_mask = torch.zeros_like(attn_probs)
    if mask is not None:
        combined_mask += mask.float() # assume mask of shape (seq_len,seq_len)
    if src_key_padding_mask is not None:
        combined_mask += src_key_padding_mask.float().unsqueeze(1).unsqueeze(1).repeat(1,num_heads,x.shape[1],1)
        # assume shape (batch_size,seq_len), repeating along head and line dimensions == "column" mask
    combined_mask = torch.where(combined_mask>0,torch.zeros_like(combined_mask)-float("inf"),torch.zeros_like(combined_mask))
    # setting masked logits to -inf before softmax
    attn_probs += combined_mask
    attn_probs = F.softmax(attn_probs, dim=-1)
    return attn_logits,attn_probs

def extract_selfattention_maps(transformer_encoder,x,mask,src_key_padding_mask):
    attn_logits_maps = []
    attn_probs_maps = []
    num_layers = transformer_encoder.num_layers
    d_model = transformer_encoder.layers[0].self_attn.embed_dim
    num_heads = transformer_encoder.layers[0].self_attn.num_heads
    norm_first = transformer_encoder.layers[0].norm_first
    with torch.no_grad():
        for i in range(num_layers):
            # compute attention of layer i
            h = x.clone()
            if norm_first:
                h = transformer_encoder.layers[i].norm1(h)
            # attn = transformer_encoder.layers[i].self_attn(h, h, h,attn_mask=mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
            # attention_maps.append(attn) # of shape [batch_size,seq_len,seq_len]
            attn_logits,attn_probs = compute_selfattention(transformer_encoder,h,mask,src_key_padding_mask,i,d_model,num_heads)
            attn_logits_maps.append(attn_logits) # of shape [batch_size,num_heads,seq_len,seq_len]
            attn_probs_maps.append(attn_probs)
            # forward of layer i
            x = transformer_encoder.layers[i](x,src_mask=mask,src_key_padding_mask=src_key_padding_mask)
    return attn_logits_maps,attn_probs_maps

# ============================================================================ #
#  HCA几何工具类 (从 Geo-Sign 项目移植)
# ============================================================================ #
if GEOOPT_AVAILABLE and PoincareBall is not None:
    
    class HCAProjection(nn.Module):
        """
        将欧氏空间特征投影到庞加莱球（Poincaré Ball）HCA空间。
        
        流程: Linear -> tangent-space -> exp-map to Poincaré ball
        自动混合精度安全: matmul在权重dtype，几何运算在fp32
        
        关键修改: 使用保守的权重初始化 (std=0.01) 防止训练初期梯度震荡
        """
        def __init__(self, dim_in: int, dim_out: int, manifold: PoincareBall):
            super().__init__()
            if not isinstance(manifold, PoincareBall):
                raise TypeError("manifold must be geoopt.manifolds.PoincareBall")
            self.manifold = manifold
            self.proj = nn.Linear(dim_in, dim_out, bias=True)
            self.log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            
            # ===== 关键修改: 保守初始化，防止破坏CLIP预训练特征 =====
            # 使用非常小的标准差初始化权重，确保训练初期HCA分支几乎不影响主干
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
            # ============================================================

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: 欧氏空间特征 [B, D_in]
            Returns:
                out: 庞加莱球上的点 [B, D_out]
            """
            w_dtype = self.proj.weight.dtype
            # 线性投影到切空间
            y_tan = self.proj(x.to(w_dtype)) * self.log_scale.to(w_dtype).exp()
            # 从原点exp映射到HCA流形
            out = self.manifold.expmap0(y_tan.float(), project=True)
            return out.to(x.dtype)

    class HCAContrastiveLoss(nn.Module):
        """
        在庞加莱球上计算对比学习损失。
        
        使用测地距离（geodesic distance）而非欧氏距离，
        利用HCA空间的层次结构特性进行对比学习。
        """
        def __init__(self, manifold: PoincareBall, label_smoothing: float = 0.1):
            super().__init__()
            if not isinstance(manifold, PoincareBall):
                raise TypeError("manifold must be geoopt.manifolds.PoincareBall")
            self.manifold = manifold
            # 可学习的温度参数和margin
            self.temp = nn.Parameter(torch.tensor(1.0))
            self.margin_base = nn.Parameter(torch.tensor(0.3))
            self.loss_fct = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing, 
                ignore_index=-100
            )

        def pair_loss(self, p: torch.Tensor, t: torch.Tensor):
            """
            计算成对的HCA对比损失。
            
            Args:
                p: 第一组HCA特征 [B, D] (例如视频特征)
                t: 第二组HCA特征 [B, D] (例如文本特征)
            
            Returns:
                包含loss和统计信息的字典
            """
            bsz = p.shape[0]
            if bsz == 0:
                return {
                    "loss": torch.tensor(0.0, device=p.device, requires_grad=True),
                    "sim_mean": torch.tensor(0.0, device=p.device),
                    "margin": self.margin_base.detach(),
                    "temp": torch.sigmoid(self.temp).detach()
                }
            
            # 计算测地距离矩阵 [B, B]
            dist = self.manifold.dist(p.unsqueeze(1), t.unsqueeze(0))
            # 转换为相似度（距离越小，相似度越高）
            sims = -dist
            
            # 温度缩放
            tau = torch.sigmoid(self.temp) * 1.99 + 0.01  # 映射到 (0.01, 2.0)
            logits = sims / tau
            
            # 添加margin到负样本
            eye = torch.eye(bsz, device=logits.device, dtype=torch.bool)
            margin_cuda = self.margin_base.to(logits.dtype)
            logits = logits + margin_cuda * (~eye)
            
            # 对角线是正样本，计算交叉熵
            targets = torch.arange(bsz, device=p.device)
            loss = self.loss_fct(logits, targets)
            
            # 统计信息
            sim_mean_pos = sims.diag().mean().detach()
            
            return {
                "loss": loss,
                "sim_mean": sim_mean_pos,
                "margin": self.margin_base.detach(),
                "temp": tau.detach()
            }


class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x


# ============= MSTE多尺度Mamba时序建模模块 (从MSTE项目移植) =============

class LayerNorm_conv(nn.LayerNorm):
    """适用于卷积特征的LayerNorm"""
    def __init__(self, normalized_shape):
        super().__init__(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)  # [B,C,H,W] -> [B,H,W,C]
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type).permute(0, 3, 1, 2)  # -> [B,C,H,W]


class Mamba_head(nn.Module):
    """Mamba时序建模头
    
    使用Mamba状态空间模型进行高效的时序建模
    输入: [B, SeqLen, C]
    输出: [B, SeqLen, C]
    """
    def __init__(self, embed_dim, layer_num=0.1):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for MSTE mode. "
                "Install with: pip install mamba-ssm einops"
            )
        
        self.embed_dim = embed_dim
        # 核心Mamba状态空间模型
        self.mamba = Mamba(
            self.embed_dim, 
            d_conv=4,           # 卷积核大小
            bimamba_type='v2',  # 双向Mamba
            use_fast_path=True, # 使用快速路径
            expand=2            # 扩展因子
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.proj_drop = nn.Dropout(layer_num)
        self.temporal_fc = nn.Linear(self.embed_dim, self.embed_dim)
        
        # 零初始化，确保训练初期稳定
        nn.init.constant_(self.temporal_fc.weight, 0.)
        nn.init.constant_(self.temporal_fc.bias, 0.)

    def forward(self, hidden_states: torch.Tensor, 
                attention_mask=None, causal_attention_mask=None):
        """
        改进的mask处理：在关键位置都应用mask
        
        Args:
            hidden_states: [B, SeqLen, C]
            attention_mask: [B, SeqLen] where 1=valid, 0=padding
            
        Returns:
            output: [B, SeqLen, C]
        """
        # 准备mask
        if attention_mask is not None:
            # attention_mask: [B, SeqLen], 1=valid, 0=padding
            mask_expanded = attention_mask.unsqueeze(-1).to(
                dtype=hidden_states.dtype, 
                device=hidden_states.device
            )  # [B, SeqLen, 1]
        else:
            mask_expanded = None
        
        # 1. 输入端Mask（防止padding噪声）
        if mask_expanded is not None:
            hidden_states = hidden_states * mask_expanded
        
        # 保存残差（已经是masked的）
        residual = hidden_states
        
        # 2. LayerNorm
        hidden_states = self.layer_norm1(hidden_states)
        
        # 3. LayerNorm后再次Mask（确保归一化后的padding也是0）
        if mask_expanded is not None:
            hidden_states = hidden_states * mask_expanded
        
        # 4. Mamba处理
        hidden_states = self.mamba(hidden_states)
        
        # 5. Mamba输出后Mask（关键：清除padding位置的状态传递）
        if mask_expanded is not None:
            hidden_states = hidden_states * mask_expanded
        
        # 6. Dropout和Linear
        res_temporal = self.proj_drop(hidden_states.contiguous())
        res_temporal = self.temporal_fc(res_temporal)
        
        # 7. 残差连接
        output = residual + res_temporal
        
        # 8. 最终Mask（确保输出padding为0，不影响后续计算）
        if mask_expanded is not None:
            output = output * mask_expanded
        
        return output

# ============= 结束MSTE模块 =============


class DiCoSA(nn.Module):
    def __init__(self, config):
        super(DiCoSA, self).__init__()

        self.config = config
        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CLIP model not found at {model_path}")
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        # for n, p in self.clip.named_parameters():
        #     if "clip.visual" in n:
        #         p.requires_grad = False
        
        if torch.cuda.is_available():
            convert_weights(self.clip)  # fp16

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config
        
        width = int(transformer_width // self.config.center)
        self.weight_fc = nn.Sequential(
                    nn.Linear(2*width, 4*width), nn.ReLU(inplace=True),
                    nn.Linear(4*width, 1))
            
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)
        
        # ========== MSTE多尺度Mamba初始化 ==========
        if self.agg_module == "MSTE":
            if not MAMBA_AVAILABLE:
                raise ImportError(
                    "mamba-ssm and einops are required for MSTE mode. "
                    "Install with: pip install mamba-ssm einops"
                )
            
            print(f"[MSTE] Initializing multi-scale Mamba temporal modeling...")
            
            # 多尺度特征提取模块: U = {3, 7, 14}
            self.mamba_stages = nn.ModuleList()
            mste_scales = [3, 7, 14]  # coarse, base, fine
            dim = transformer_width
            
            for target_grid in mste_scales:
                layers = []
                out_dim = dim
                out_channels = dim
                
                # 输入为 7x7 patch map，对应输出 3x3、7x7、14x14 三个尺度
                if target_grid == 3:
                    layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
                elif target_grid == 7:
                    layers = []
                elif target_grid == 14:
                    layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                    out_dim = dim // 2
                else:
                    raise NotImplementedError(f"MSTE scale {target_grid} is not supported.")
                
                # 后接特征融合卷积层
                layers.extend([
                    nn.Conv2d(out_dim, out_channels, kernel_size=1),
                    LayerNorm_conv(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    LayerNorm_conv(out_channels)
                ])
                
                self.mamba_stages.append(nn.Sequential(*layers))
            
            # Mamba时序建模模块
            depth = getattr(config, 'num_hidden_layers', 4)
            dpr = np.linspace(0, 0.1, depth)  # Stochastic depth decay rule
            self.MS_mamba = nn.ModuleList([
                Mamba_head(transformer_width, dpr[i]) 
                for i in range(depth)
            ])
            
            self.mste_scales = mste_scales
            self.patch_counts = [scale * scale for scale in self.mste_scales]
            
            # 尺度门控权重 (初始化为0，确保初始状态不破坏预训练特征)
            for i in range(len(self.patch_counts)):
                self.register_parameter(f'scale_gate_{i}', nn.Parameter(torch.zeros(1)))
            
            print(f"[MSTE] Initialized: scales={self.mste_scales}, {depth} Mamba layers, scale-aware cross-attention fusion")
        # ========== 结束MSTE初始化 ==========

        self.loss_fct = CrossEn(config)
        self.apply(self.init_weights)  # random init must before loading pretrain
        self.clip.load_state_dict(state_dict, strict=False)

        ## ===> Initialization trick [HARD CODE]
        new_state_dict = OrderedDict()
                
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue

        self.num_queries = self.config.query_number
        self.noun_queries = nn.Parameter(torch.rand(self.num_queries, transformer_width))
        self.spatial_queries = nn.Parameter(torch.rand(self.num_queries, transformer_width))
        # self.spatial_queries = self.noun_queries.clone()

        self.decoder_layer_noun = nn.TransformerDecoderLayer(d_model=transformer_width, nhead=8)
        self.transformer_decoder_noun = nn.TransformerDecoder(self.decoder_layer_noun, num_layers=self.config.cross_att_layer)

        self.decoder_layer_spatial = nn.TransformerDecoderLayer(d_model=transformer_width, nhead=8)
        self.transformer_decoder_spatial = nn.TransformerDecoder(self.decoder_layer_spatial, num_layers=self.config.cross_att_layer)

        # self.compact_s = nn.Linear(transformer_width, transformer_width // 8)
        # self.compact_n = nn.Linear(transformer_width, transformer_width // 8)
        
        # self.transformer_decoder_noun = nn.ModuleList(
        #     [EncoderLayer(transformer_width, transformer_width, 8, transformer_width, transformer_width) for _ in range(self.config.cross_att_layer)])

        # self.transformer_decoder_spatial = nn.ModuleList(
        #     [EncoderLayer(transformer_width, transformer_width, 8, transformer_width, transformer_width) for _ in range(self.config.cross_att_layer)])

        self.load_state_dict(new_state_dict, strict=False)  # only update new state (seqTransf/seqLSTM/tightTransf)
        ## <=== End of initialization trick
        
        # ========== HCA辅助分支初始化 ==========
        # 从 config 读取HCA几何相关配置
        use_hca = getattr(config, 'use_hca', True)
        self.hca_alpha = getattr(config, 'hca_alpha', 0.5)
        
        if GEOOPT_AVAILABLE and PoincareBall is not None and use_hca:
            print(f"[HCA Branch] Initializing HCA auxiliary branch...")
            
            # 定义庞加莱球流形 (曲率 c=1.0, 可学习)
            self.manifold = PoincareBall(c=1.0, learnable=True)
            
            # 定义投影层：将欧氏特征投影到HCA空间
            # transformer_width 是 CLIP 的特征维度 (通常为512)
            self.hca_proj_text = HCAProjection(
                dim_in=transformer_width,
                dim_out=transformer_width,
                manifold=self.manifold
            )
            self.hca_proj_video = HCAProjection(
                dim_in=transformer_width,
                dim_out=transformer_width,
                manifold=self.manifold
            )
            
            # 定义HCA对比损失函数 (使用标签平滑)
            self.hca_loss_fct = HCAContrastiveLoss(
                manifold=self.manifold,
                label_smoothing=0.1
            )
            
            print(f"[HCA Branch] Initialized successfully!")
            print(f"  - Manifold: PoincareBall(c={self.manifold.c.item():.3f}, learnable=True)")
            print(f"  - Projection dim: {transformer_width} -> {transformer_width}")
            print(f"  - Loss weight (alpha): {self.hca_alpha}")
        else:
            # 如果 geoopt 不可用或未启用HCA分支，设置标志位
            self.manifold = None
            self.hca_proj_text = None
            self.hca_proj_video = None
            self.hca_loss_fct = None
            if use_hca and not GEOOPT_AVAILABLE:
                print("[HCA Branch] Warning: use_hca=True but geoopt not available.")
            elif not use_hca:
                print("[HCA Branch] Disabled (use_hca=False).")
        # ========== 结束HCA辅助分支初始化 ==========
        
    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # B x N_v x 3 x H x W - >  (B x N_v) x 3 x H x W
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat, video_feat, cls = self.get_text_video_feat(text_ids, text_mask, video, video_mask, shaped=True)

        if self.training:
            if torch.cuda.is_available():  # batch merge here
                idx = allgather(idx, self.config)
                text_feat = allgather(text_feat, self.config)
                video_feat = allgather(video_feat, self.config)
                text_mask = allgather(text_mask, self.config)
                video_mask = allgather(video_mask, self.config)
                cls = allgather(cls, self.config)
                torch.distributed.barrier()  # force sync

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
            logit_scale = self.clip.logit_scale.exp()
            loss = 0.

            # ========== 步骤1: 主损失计算（欧氏空间检索）==========
            M_t2v_logits, M_v2t_logits, ssr_loss = self.get_similarity_logits(text_feat, cls, video_feat,
                                                                               text_mask, video_mask, shaped=True)
            
            M_loss_t2v = self.loss_fct(M_t2v_logits * logit_scale)
            M_loss_v2t = self.loss_fct(M_v2t_logits * logit_scale)
            M_loss = (M_loss_t2v + M_loss_v2t) / 2
            
            # ========== 步骤2: HCA辅助损失计算（独立分支）==========
            hca_loss = M_loss.new_zeros(())
            if GEOOPT_AVAILABLE and self.manifold is not None:
                # 关键：在 FP32 精度下执行HCA几何运算，确保数值稳定性
                with torch.cuda.amp.autocast(enabled=False):
                    # 文本特征：直接使用 CLIP 的 [CLS] token (已经是 [B, D])
                    # 视频特征：video_feat 形状为 [B, T, D]，需要进行 mask-aware 平均池化
                    
                    # 构建 mask：[B, T] -> [B, T, 1]
                    video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
                    
                    # Mask-aware 平均池化：忽略 padding 位置
                    # 分子：sum(video_feat * mask, dim=1) -> [B, D]
                    # 分母：sum(mask, dim=1).clamp_min(1) -> [B, 1]
                    video_feat_pooled = (video_feat * video_mask_un).sum(dim=1) / \
                                       video_mask_un.sum(dim=1).clamp_min(1)
                    
                    # 投影到庞加莱球 (切空间 -> HCA空间)
                    hca_text = self.hca_proj_text(cls.float())
                    hca_video = self.hca_proj_video(video_feat_pooled.float())
                    
                    # 计算HCA对比损失 (基于庞加莱球上的测地距离)
                    hca_loss_dict = self.hca_loss_fct.pair_loss(hca_video, hca_text)
                    hca_loss = hca_loss_dict["loss"]
            # ========== 结束HCA辅助损失计算 ==========
            
            # ========== 步骤3: 最终损失融合 ==========
            hca_loss = self.hca_alpha * hca_loss
            loss = M_loss + hca_loss + ssr_loss

            return loss, M_loss, hca_loss, ssr_loss
        else:
            return None
        
    def get_similarity_logits(self, text_feat, cls, video_feat, text_mask, video_mask, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        M_t2v_logits, M_v2t_logits, ssr_loss = self.similarity(text_feat, cls, video_feat, text_mask, video_mask)
        
        return M_t2v_logits, M_v2t_logits, ssr_loss
    
    def similarity(self, text_feat, cls, video_feat, text_mask, video_mask):
        '''
        text_feat: torch.Size([128, 32, 512])
        cls: torch.Size([128, 512])
        video_feat: torch.Size([128, 12, 512])
        print(text_feat.shape, cls.shape, video_feat.shape)
        self.noun_queries: torch.rand(self.num_queries, transformer_width)
        self.spatial_queries: torch.rand(self.num_queries, transformer_width)
        video_feat: torch.Size([128, 12, 512])
        print(cls.shape, video_feat.shape)
        '''
        
        v_weight = torch.einsum('ad,bvd->abv', [cls, video_feat]) # bs 512, bs 12 512 -> bs bs 12
        v_weight = torch.softmax(v_weight / self.config.temp, dim=-1)
        v_weight = torch.einsum('abv,bv->abv', [v_weight, video_mask])  
        video_feat_t_cond = torch.einsum('abv,bvd->abd', [v_weight, video_feat]) # bs bs 12, bs 12 512 -> bs bs 512

        a, b = cls.size(0), video_feat_t_cond.size(1)
        cls, video_feat_t_cond = cls.contiguous(), video_feat_t_cond.contiguous()
        t_feat = cls.view(a, self.config.center, -1)
        v_feat = video_feat_t_cond.view(a, b, self.config.center, -1)
        d = t_feat.size(2)
        
        temp = torch.cat([t_feat.unsqueeze(1).repeat(1, b, 1, 1), v_feat], dim=-1) # bs bs 1 512*2
        weight = self.weight_fc(temp).squeeze(3)  # a b c 2d-> a b c
        
        _t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        _v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)
        
        # ========== 步骤1: 全局欧氏相似度计算（Cosine Similarity）==========
        # 基于归一化的文本和视频特征，计算余弦相似度
        retrieve_logits = torch.einsum('acd,abcd->abc', [_t_feat, _v_feat]).squeeze()

        # ========== 步骤2: 局部相似度融合（训练时）==========
        if self.training:
            # 通过 spatial 和 noun queries 计算细粒度相似度
            s, spatial_out, noun_out = self._score(text_feat, cls, video_feat, text_mask, video_mask)

            # [关键] 将局部相似度加权融合到主相似度矩阵
            # 这是纯欧氏空间的相似度融合，不涉及HCA几何
            retrieve_logits += self.config.loss2_weight * s

            sim_metric = torch.einsum('abc,adc->abd', [spatial_out, noun_out])
            sim_metric = sim_metric.sum(0) / spatial_out.shape[1]
            ssr_delta = getattr(self.config, 'ssr_delta', 0.75)
            ssr_beta = getattr(self.config, 'ssr_beta', 0.07)
            r = torch.diag(sim_metric).add(-ssr_delta).pow(2).sum()
            
            ssr_loss = r * ssr_beta

        else:   
            ssr_loss = retrieve_logits.new_zeros(())

        return retrieve_logits, retrieve_logits.T, ssr_loss

    def _score(self, text_feat, cls, video_feat, text_mask, video_mask):
        if self.config.query_share:
            spatial_q = self.spatial_queries.expand(video_feat.size(0),-1,-1)
            noun_q = self.spatial_queries.expand(text_feat.size(0),-1,-1)
        else:
            spatial_q = self.spatial_queries.expand(video_feat.size(0),-1,-1)
            noun_q = self.noun_queries.expand(text_feat.size(0),-1,-1)
        
        tgt = spatial_q.permute(1, 0, 2) # NLD -> LND
        memory = noun_q.permute(1, 0, 2) # NLD -> LND
        
        if self.config.cross_att_share:
            # spatial cross-attention
            spatial_out = self.transformer_decoder_noun(tgt, video_feat.permute(1, 0, 2)).permute(1, 0, 2) # spatial as query
            # noun cross-attention
            noun_out = self.transformer_decoder_noun(memory, text_feat.permute(1, 0, 2)).permute(1, 0, 2) # noun as query
        else:
            # spatial cross-attention
            spatial_out = self.transformer_decoder_spatial(tgt, video_feat.permute(1, 0, 2)).permute(1, 0, 2) # spatial as query
            # noun cross-attention
            noun_out = self.transformer_decoder_noun(memory, text_feat.permute(1, 0, 2)).permute(1, 0, 2) # noun as query

        # spatial_out = spatial_q
        # noun_out = noun_q
        # if self.config.cross_att_share:
        #     # spatial cross-attention
        #     for i in range(self.config.cross_att_layer):
        #         spatial_out = self.transformer_decoder_noun[i](spatial_out, video_feat, video_feat) # spatial as query
        #         # noun cross-attention
        #         noun_out = self.transformer_decoder_noun[i](noun_out, text_feat, text_feat) # noun as query
        # else:
        #     # spatial cross-attention
        #     spatial_out = self.transformer_decoder_spatial[i](spatial_out, video_feat, video_feat) # spatial as query
        #     # noun cross-attention
        #     noun_out = self.transformer_decoder_noun[i](noun_out, text_feat, text_feat) # noun as query
        
        # normalization
        spatial_out = spatial_out / spatial_out.norm(dim=-1, keepdim=True)  # batch x num_query x dim
        noun_out = noun_out / noun_out.norm(dim=-1, keepdim=True) # batch x num_query x dim

        s = torch.matmul(noun_out.permute(1, 0 ,2), spatial_out.permute(1, 2, 0)) # num_query x batch x dim, num_query x dim x batch
        s = s.sum(0) / self.num_queries
        
        return s, spatial_out, noun_out

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        cls, text_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        cls, text_feat = cls.float(), text_feat.float()
        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1))
        cls = cls.view(bs_pair, -1, cls.size(-1)).squeeze(1)
        return text_feat, cls

    def get_video_feat(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        
        # MSTE需要完整的tokens（包括CLS + patches），其他方法只需要CLS token
        if self.agg_module == "MSTE":
            # encode_image returns (cls_token, all_tokens)
            # all_tokens shape: [bs_pair * n_v, L, C] where L=50 for ViT-B/32
            _, video_feat_all = self.clip.encode_image(video, return_hidden=True)
            video_feat = video_feat_all.float()
            # video_feat: [bs_pair * n_v, L, C]
            # reshape to: [bs_pair, n_v * L, C]
            video_feat = video_feat.view(bs_pair, -1, video_feat.size(-1))
        else:
            # 其他方法只需要CLS token
            video_feat = self.clip.encode_image(video, return_hidden=True)[0].float()
            video_feat = video_feat.float().view(bs_pair, -1, video_feat.size(-1))
        
        video_feat = self.agg_video_feat(video_feat, video_mask, self.agg_module)
        return video_feat

    def get_text_video_feat(self, text_ids, text_mask, video, video_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat, cls = self.get_text_feat(text_ids, text_mask, shaped=True)
        video_feat = self.get_video_feat(video, video_mask, shaped=True)

        return text_feat, video_feat, cls

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    def get_text_sep_feat(self, text_feat, text_mask):
        text_feat = text_feat.contiguous()
        text_feat = text_feat[torch.arange(text_feat.shape[0]), torch.sum(text_mask, dim=-1) - 1, :]
        text_feat = text_feat.unsqueeze(1).contiguous()
        return text_feat

    def agg_video_feat(self, video_feat, video_mask, agg_module):
        video_feat = video_feat.contiguous()
        if agg_module == "None":
            pass
        elif agg_module == "seqLSTM":
            # Sequential type: LSTM
            video_feat_original = video_feat
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
                                              batch_first=True, enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            if self.training: self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            video_feat = torch.cat(
                (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqTransf":
            # Sequential type: Transformer Encoder
            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_feat = video_feat.permute(1, 0, 2)  # NLD -> LND
            video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = video_feat.permute(1, 0, 2)  # LND -> NLD
            video_feat = video_feat + video_feat_original
        elif agg_module == "MSTE":
            # ========== MSTE多尺度Mamba时序建模 ==========
            B, TL, C = video_feat.shape
            
            # 验证特征格式 (ViT-B/32: L=50)
            L = 50  # 1 CLS token + 49 patch tokens (7×7)
            if TL % L != 0:
                raise ValueError(
                    f"[MSTE] video_feat shape {video_feat.shape} is not compatible with L={L}. "
                    f"Expected TL to be divisible by {L}. "
                    f"Check CLIP output format."
                )
            T = TL // L  # 帧数
            H = W = int(math.sqrt(L - 1))  # H=W=7 (49个patches)
            
            # Step 1: 重塑特征 [B, T*L, C] -> [B, T, L, C]
            video_feat = video_feat.view(B, T, L, C)
            
            # Step 2: 分离CLS tokens和patch tokens
            visual_output_original = video_feat[:, :, 0, :]  # CLS tokens [B, T, C]
            visual_mamba = video_feat[:, :, 1:, :]  # Patch tokens [B, T, 49, C]
            
            # Step 3: 转换为空间格式 [B*T, C, H, W]
            visual_mamba = visual_mamba.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
            
            # Step 4: 多尺度特征提取
            visual_mamba_ms = []
            for stage in self.mamba_stages:
                # stage输出: [B*T, C, H', W']
                stage_out = stage(visual_mamba)
                # 重塑为: [B, T, C, H'*W'] -> [B, T, H'*W', C]
                stage_out = stage_out.view(B, T, C, -1).permute(0, 1, 3, 2)
                visual_mamba_ms.append(stage_out)
            
            # Step 5: 按帧拼接 CLS 和多尺度 patch tokens
            # [B, T, 1 + sum(num_patches), C] -> [B, T * tokens_per_frame, C]
            visual_mamba_ms = torch.cat(visual_mamba_ms, dim=2)
            tokens_per_frame = 1 + sum(self.patch_counts)
            visual_mamba_output = torch.cat(
                (visual_output_original.unsqueeze(2), visual_mamba_ms),
                dim=2
            ).view(B, T * tokens_per_frame, C)
            
            # === 新增：构造超长 Mask ===
            # 原始 video_mask: [B, T] (1=Valid, 0=Pad)
            # Mamba 输入 visual_mamba_output: [B, T * (1 + sum_patches), C]
            # 我们需要把 mask 扩展到对应的长度
            if video_mask is not None:
                # 1. 获取序列结构信息
                # patches 总数 = 1 (CLS) + 9 (3x3) + 49 (7x7) + 196 (14x14) = 255
                # 2. 扩展 Mask
                # 逻辑：如果某一帧是 Padding，那么这一帧产生的所有 255 个 token 都是 Padding
                # [B, T] -> [B, T, 1] -> [B, T, 255] -> [B, T*255]
                extended_mamba_mask = video_mask.unsqueeze(-1).repeat(1, 1, tokens_per_frame)
                extended_mamba_mask = extended_mamba_mask.view(B, -1)  # [B, T*255]
            else:
                extended_mamba_mask = None
            
            # Step 6: Mamba时序建模 (4层)
            for layer in self.MS_mamba:
                # 传入构造好的 extended_mamba_mask
                visual_mamba_output = layer(visual_mamba_output, attention_mask=extended_mamba_mask)
            
            # Step 7: 尺度感知交叉注意力融合 (Scale-Aware Cross-Attention)
            # visual_mamba_output: [B, T * (1 + sum_patches), C]
            visual_mamba_output = visual_mamba_output.contiguous().view(B, T, tokens_per_frame, C)
            cls_tokens = visual_mamba_output[:, :, 0, :]  # [B, T, C] -> 作为 Query
            multi_scale_tokens = visual_mamba_output[:, :, 1:, :]  # [B, T, sum_patches, C] -> 作为 Key/Value
            
            # 准备融合容器
            fused_details = torch.zeros_like(cls_tokens)  # [B, T, C]
            start_idx = 0
            B, T, C = cls_tokens.shape
            
            # 将 CLS 视为 Query: [B*T, 1, C]
            # 确保 q 的形状是 [B*T, 1, C]，避免广播问题
            q = cls_tokens.contiguous().view(B * T, C).unsqueeze(1)  # [B, T, C] -> [B*T, C] -> [B*T, 1, C]
            # 确保 q 的形状正确
            assert q.shape == (B * T, 1, C), f"q shape should be [{B*T}, 1, {C}], but got {q.shape}"
            
            # 准备mask用于attention（如果存在）
            if video_mask is not None:
                # video_mask: [B, T] -> [B*T, 1] 用于mask无效帧的attention
                frame_mask = video_mask.contiguous().view(B * T, 1).to(dtype=q.dtype)  # [B*T, 1]
            else:
                frame_mask = None
            
            # 遍历每个尺度进行 Attention (比全局 Attention 更能保留尺度特性)
            for i, count in enumerate(self.patch_counts):
                end_idx = start_idx + count
                
                # 提取当前尺度的 Patches
                # multi_scale_tokens[:, :, start_idx:end_idx, :] 形状: [B, T, count, C]
                scale_patches = multi_scale_tokens[:, :, start_idx:end_idx, :]
                k_v = scale_patches.contiguous().view(B * T, count, C)  # [B*T, count, C]
                # 确保 k_v 的形状正确
                assert k_v.shape == (B * T, count, C), f"k_v shape should be [{B*T}, {count}, {C}], but got {k_v.shape}"
                
                # --- 轻量级 Cross-Attention ---
                # Score: [B*T, 1, C] @ [B*T, C, count] -> [B*T, 1, count]
                # 使用 bmm 确保逐帧计算，避免广播问题
                k_v_t = k_v.transpose(-2, -1)  # [B*T, C, count]
                attn_score = torch.bmm(q, k_v_t) / math.sqrt(C)  # [B*T, 1, C] @ [B*T, C, count] -> [B*T, 1, count]
                # 确保 attn_score 的形状正确
                assert attn_score.shape == (B * T, 1, count), f"attn_score shape should be [{B*T}, 1, {count}], but got {attn_score.shape}"
                
                # 应用mask（如果存在）：将padding帧的attention score设为-inf
                if frame_mask is not None:
                    # 对于padding帧（frame_mask=0），将attention score设为-inf
                    # frame_mask: [B*T, 1] -> [B*T, 1, 1] 以匹配 attn_score: [B*T, 1, count]
                    frame_mask_expanded = frame_mask.unsqueeze(-1)  # [B*T, 1, 1]
                    attn_score = attn_score + (frame_mask_expanded - 1.0) * 1e9
                
                attn_probs = F.softmax(attn_score, dim=-1)  # [B*T, 1, count]
                # 确保 attn_probs 的形状正确
                assert attn_probs.shape == (B * T, 1, count), f"attn_probs shape should be [{B*T}, 1, {count}], but got {attn_probs.shape}"
                
                # Weighted Sum: [B*T, 1, count] @ [B*T, count, C] -> [B*T, 1, C]
                # 确保 attn_probs 和 k_v 的形状正确
                # attn_probs: [B*T, 1, count], k_v: [B*T, count, C]
                # 使用 bmm (batch matrix multiplication) 确保维度正确
                scale_out = torch.bmm(attn_probs, k_v)  # [B*T, 1, count] @ [B*T, count, C] -> [B*T, 1, C]
                # 确保 scale_out 的形状正确
                assert scale_out.shape == (B * T, 1, C), f"scale_out shape should be [{B*T}, 1, {C}], but got {scale_out.shape}"
                
                # 恢复形状 [B, T, C]
                # 先 squeeze 移除中间维度，再 reshape
                scale_out = scale_out.squeeze(1)  # [B*T, 1, C] -> [B*T, C]
                assert scale_out.shape == (B * T, C), f"After squeeze, scale_out shape should be [{B*T}, {C}], but got {scale_out.shape}"
                scale_out = scale_out.contiguous().view(B, T, C)  # [B*T, C] -> [B, T, C]
                
                # --- 尺度门控/权重 ---
                # 让模型学习每个尺度的重要性
                gate = torch.tanh(getattr(self, f'scale_gate_{i}'))
                
                # 应用mask：确保padding帧的融合细节为0
                if video_mask is not None:
                    mask_expanded = video_mask.unsqueeze(-1).to(dtype=scale_out.dtype)  # [B, T, 1]
                    scale_out = scale_out * mask_expanded
                
                fused_details = fused_details + gate * scale_out
                
                start_idx = end_idx
            
            # 最终融合: 残差连接
            video_feat = cls_tokens + fused_details
            
            # 最终mask确保输出padding为0
            if video_mask is not None:
                mask_expanded = video_mask.unsqueeze(-1).to(dtype=video_feat.dtype)  # [B, T, 1]
                video_feat = video_feat * mask_expanded
            
            video_feat = video_feat.contiguous()
            # ========== MSTE处理结束 ==========
        else:
            raise ValueError(f"Unknown agg_module: {agg_module}")
        
        return video_feat


    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
