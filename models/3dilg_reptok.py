import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import weight_norm

from timm.models.layers import drop_path, trunc_normal_

from torch_cluster import fps, knn
from torch_scatter import scatter_max


def embed(input, basis):
    projections = torch.einsum('bnd,de->bne', input, basis)
    embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
    return embeddings


class PointConv(torch.nn.Module):
    def __init__(self, local_nn=None, global_nn=None):
        super(PointConv, self).__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn

    def forward(self, pos, pos_dst, edge_index, basis=None):
        row, col = edge_index

        out = pos[col] - pos_dst[row]

        if basis is not None:
            embeddings = torch.einsum('bd,de->be', out, basis)
            embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=1)
            out = torch.cat([out, embeddings], dim=1)


        if self.local_nn is not None:
            out = self.local_nn(out)
        
        out, _ = scatter_max(out, col, dim=0, dim_size=col.max().item() + 1)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class DropPath(nn.Module):
    """
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """
        DiNO Vision Transformer from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        and https://github.com/facebookresearch/dino/blob/main/vision_transformer.py modified for point cloud feature input
    """
    def __init__(self, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., # positional encoding drop rate
                 attn_drop_rate=0., # transformer block drop rate
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 **kwargs
                 ):
        super().__init__()
        
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # class tokenizer
        # position embedding/ positional encoding implicitly through point cloud encoding: the following not needed here
        # self.pos_embded = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate) # positional encoding drop

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (for transformer block drop)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=kwargs["init_values"]) # additionally initial values
            for i in range(depth)])

        self.norm =  norm_layer(embed_dim)

        # classifier head not required here

        trunc_normal_(self.cls_token, std=0.02) # initialize class tokenizer weights

        self.apply(self._init_weights) # initialize (rest of the) model weights

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)
    
    def prepare_tokens(self, x, pos_embed):
        B, _, _ = x.shape

        # add the [CLS] token to the the input
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)


    def forward(self, x, pos_embed):
        # x = self.patch_embed(x)
        B, _, _ = x.size()

        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x


class MLPOccupancy(nn.Module):
    def __init__(self, space_dim=3, latent_dim=128, pos_enc_dim=48):
        super(MLPOccupancy, self).__init__()
        # self.register_buffer('B', torch.randn((128, 3)) * 2)

        e = torch.pow(2, torch.arange(pos_enc_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(pos_enc_dim // 6),
                      torch.zeros(pos_enc_dim // 6)]),
            torch.cat([torch.zeros(pos_enc_dim // 6), e,
                      torch.zeros(pos_enc_dim // 6)]),
            torch.cat([torch.zeros(pos_enc_dim // 6),
                      torch.zeros(pos_enc_dim // 6), e]),
        ])
        self.register_buffer('basis', e)

        self.l1 = weight_norm(nn.Linear(space_dim + latent_dim + pos_enc_dim, 512))
        self.l2 = weight_norm(nn.Linear(512, 512))
        self.l3 = weight_norm(nn.Linear(512, 512))
        self.l4 = weight_norm(nn.Linear(512, 512 - space_dim - latent_dim - pos_enc_dim))
        self.l5 = weight_norm(nn.Linear(512, 512))
        self.l6 = weight_norm(nn.Linear(512, 512))
        self.l7 = weight_norm(nn.Linear(512, 512))
        self.l_out = weight_norm(nn.Linear(512, 1))

    def forward(self, x, z):
        # x: B x N x 3
        # z: B x N x 192
        # input = torch.cat([x[:, :, None].expand(-1, -1, z.shape[1], -1), z[:, None].expand(-1, x.shape[1], -1, -1)], dim=-1)
        # print(x.shape, z.shape)

        embeddings = embed(x, self.basis)

        input = torch.cat([x, embeddings, z], dim=2)

        h = F.relu(self.l1(input))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = torch.cat((h, input), axis=2)
        h = F.relu(self.l5(h))
        h = F.relu(self.l6(h))
        h = F.relu(self.l7(h))
        h = self.l_out(h)
        return h


class Encoder(nn.Module):
    def __init__(self, num_points, enc_dim=128, pos_enc_dim=48, num_subsample=1024, num_neighbors= 32, space_dim=3):
        super().__init__()

        self.space_dim = space_dim

        self.ratio = num_subsample / num_points
        self.k = num_neighbors

        e = torch.pow(2, torch.arange(pos_enc_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(pos_enc_dim // 6),
                      torch.zeros(pos_enc_dim // 6)]),
            torch.cat([torch.zeros(pos_enc_dim // 6), e,
                      torch.zeros(pos_enc_dim // 6)]),
            torch.cat([torch.zeros(pos_enc_dim // 6),
                      torch.zeros(pos_enc_dim // 6), e]),
        ])
        self.register_buffer('basis', e)

        self.point_conv = PointConv(
            local_nn=nn.Sequential(weight_norm(nn.Linear(space_dim + pos_enc_dim, 256)), nn.ReLU(True), weight_norm(nn.Linear(256, 256))),
            global_nn=nn.Sequential(weight_norm(nn.Linear(256, 256)), nn.ReLU(True), weight_norm(nn.Linear(256, enc_dim)))
        )
        self.point_encoder = nn.Sequential(nn.Linear(space_dim + pos_enc_dim, enc_dim))

        self.transformer = VisionTransformer(embed_dim=enc_dim, 
                                            depth=6,
                                            num_heads=6, 
                                            mlp_ratio=4., 
                                            qkv_bias=True, 
                                            qk_scale=None, 
                                            drop_rate=0., 
                                            attn_drop_rate=0.,
                                            drop_path_rate=0.1, 
                                            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                            init_values=0.,
                                            )

    def forward(self, pc):
        B, N, D = pc.shape # batch_size x num_points x space dimension

        flattened_pc = pc.view(B * N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        points = flattened_pc

        fps_idx = fps(points, batch, ratio=self.ratio)

        row, col = knn(points, points[fps_idx], self.k, batch, batch[fps_idx])
        edge_index = torch.stack([row, col], dim=0)

        x = self.point_conv(points, points[fps_idx], edge_index, self.basis)
        points, batch = points[fps_idx], batch[fps_idx]

        x = x.view(B, -1, x.shape[-1])
        points = points.view(B, -1, D)

        embeddings = embed(points, self.basis)
        embeddings = self.point_encoder(torch.cat([points, embeddings], dim=2))

        out = self.transformer(x, embeddings)

        return out, points


class Decoder(nn.Module):
    def __init__(self, enc_dim=128, pos_enc_dim=48, space_dim=3):
        super().__init__()

        self.space_dim = space_dim

        e = torch.pow(2, torch.arange(pos_enc_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(pos_enc_dim // 6),
                      torch.zeros(pos_enc_dim // 6)]),
            torch.cat([torch.zeros(pos_enc_dim // 6), e,
                      torch.zeros(pos_enc_dim // 6)]),
            torch.cat([torch.zeros(pos_enc_dim // 6),
                      torch.zeros(pos_enc_dim // 6), e]),
        ])
        self.register_buffer('basis', e)

        self.point_encoder = nn.Sequential(nn.Linear(space_dim + pos_enc_dim, enc_dim))

        self.transformer = VisionTransformer(embed_dim=enc_dim, 
                                            depth=6,
                                            num_heads=6, 
                                            mlp_ratio=4., 
                                            qkv_bias=True, 
                                            qk_scale=None, 
                                            drop_rate=0., 
                                            attn_drop_rate=0.,
                                            drop_path_rate=0.1, 
                                            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                            init_values=0.,
                                            )

        self.log_sigma = nn.Parameter(torch.FloatTensor([3.0]))
        self.mlp = MLPOccupancy(latent_channel=enc_dim)
        
    def forward(self, latents, centers, points_to_predict):
        embeddings = embed(centers, self.basis)
        embeddings = self.point_encoder(torch.cat([centers, embeddings], dim=2)) # why are embeddings computed again?
        latents = self.transformer(latents, embeddings) # why are the latents created twice (twice transformer)?

        pdist = (points_to_predict[:, :, None] - centers[:, None]).square().sum(dim=3) # need to know what points to predict are

        sigma = torch.exp(self.log_sigma)
        weight = F.softmax(-pdist * sigma, dim=2)

        latents = torch.sum(weight[:, :, :, None] * latents[:, None, :, :], dim=2)
        preds = self.mlp(points_to_predict, latents).squeeze(2)
        
        return preds, sigma



class AutoEncoder(nn.Module):
    def __init__(self, num_points, enc_dim=128, num_subsample=1024, num_neighbors= 32):
        super().__init__()

        self.encoder = Encoder(num_points=num_points, enc_dim=enc_dim, num_subsample=num_subsample, num_neighbors=num_neighbors)

    def encode(self, x):
        # B, _, _ = x.shape
        
        z, centers = self.encoder(x)

        return z, centers
    
    def forward(self, x, points_to_predict):

        z, centers = self.encode(x)