import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List


ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    # Create meshgrid - compatible with older PyTorch versions
    grid_y, grid_x = torch.meshgrid(ys, xs)
    indices = torch.stack([grid_x, grid_y], 0)                            # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w

    def get_prior(self):
        return self.learned_features


class CrossAttentionEAF(nn.Module):
    """
    Cross-view attention with Epipolar Attention Field (EAF).

    입력:
      q: [B, D, H_bev, W_bev]          # BEV features
      k: [B, N, D, h, w]               # per-view image features (after proj)
      v: [B, N, D, h, w]
      W_logits: [B, Q, NK]             # EAF weights (Q = H_bev*W_bev, NK = N*h*w)
      skip: [B, D, H_bev, W_bev] or None

    출력:
      z: [B, D, H_bev, W_bev]
    """
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head

        self.q_norm = norm(dim)
        self.k_norm = norm(dim)
        self.v_norm = norm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)

        self.proj = nn.Linear(inner_dim, dim)

        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.postnorm = norm(dim)

    def forward(self, q, k, v, W_logits, skip=None):
        """
        q: [B, D, H_bev, W_bev]
        k: [B, N, D, h, w]
        v: [B, N, D, h, w]
        W_logits: [B, Q, NK]
        """
        B, D, H_bev, W_bev = q.shape
        Bk, N, Dk, h, w = k.shape
        assert B == Bk and D == Dk, "q/k shape mismatch"

        Q = H_bev * W_bev
        K_per = h * w
        NK = N * K_per

        # ---- reshape ----
        # q: [B, Q, D]
        q = rearrange(q, 'b d H W -> b (H W) d')

        # k,v: [B, NK, D]
        k = rearrange(k, 'b n d h w -> b (n h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # ---- norm ----
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # ---- projections ----
        q = self.to_q(q)   # [B, Q, H*Dh]
        k = self.to_k(k)   # [B, NK, H*Dh]
        v = self.to_v(v)   # [B, NK, H*Dh]

        # ---- split heads ----
        q = rearrange(q, 'b Q (h d) -> b h Q d', h=self.heads, d=self.dim_head)
        k = rearrange(k, 'b K (h d) -> b h K d', h=self.heads, d=self.dim_head)
        v = rearrange(v, 'b K (h d) -> b h K d', h=self.heads, d=self.dim_head)

        # ---- scaled dot product ----
        # logits: [B, H, Q, NK]
        logits = torch.einsum('b h Q d, b h K d -> b h Q K', q, k) * self.scale

        # ---- apply Epipolar weights (multiplicative, as in W ⊙ (QK^T/√d)) ----
        # W_logits: [B, Q, NK] -> [B, 1, Q, NK] -> [B, H, Q, NK]
        W = W_logits.unsqueeze(1).expand(-1, self.heads, -1, -1)
        logits = logits * W

        # ---- softmax over keys ----
        attn = torch.softmax(logits, dim=-1)   # [B, H, Q, NK]

        # ---- aggregate values ----
        out = torch.einsum('b h Q K, b h K d -> b h Q d', attn, v)  # [B,H,Q,Dh]
        out = rearrange(out, 'b h Q d -> b Q (h d)')                # [B,Q,inner]
        out = self.proj(out)                                       # [B,Q,D]

        # ---- reshape back to BEV map ----
        z = rearrange(out, 'b (H W) d -> b d H W', H=H_bev, W=W_bev)

        # ---- residual & FFN ----
        if skip is not None:
            z = z + skip

        z = self.prenorm(rearrange(z, 'b d H W -> b (H W) d'))
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H_bev, W=W_bev)

        return z


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


class CrossViewAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
        eaf_lambda: float = 1.0,
        eaf_zmax: float = 4.0,
        use_adaptive_lambda: bool = True,
        min_sigma: float = 1.0,
        max_sigma: float = 8.0,
    ):
        super().__init__()

        # 이미지 feature grid (키 좌표용; positional encoding 용도 X)
        image_plane = generate_grid(feat_height, feat_width)         # [1,3,h,w]
        image_plane[:, 0] *= image_width
        image_plane[:, 1] *= image_height
        self.register_buffer('image_plane', image_plane, persistent=False)

        # 값/키용 feature projection
        self.val_proj = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False),
        )

        if no_image_features:
            self.key_proj = None
        else:
            self.key_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False),
            )

        self.cross_attend = CrossAttentionEAF(dim, heads, dim_head, qkv_bias)
        self.skip = skip

        # EAF params
        self.eaf_lambda = eaf_lambda
        self.eaf_zmax = eaf_zmax
        self.use_adaptive_lambda = use_adaptive_lambda
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    # -------- Adaptive λ_{q,i} --------
    def _compute_lambda_qi(self, bev: BEVEmbedding, E_inv: torch.Tensor):
        """
        Adaptive λ_{q,i}
        bev.grid: [3, H_bev, W_bev] (ego-plane)
        E_inv:    [B, N, 4, 4]      (cam->world)

        return: [B, N, Q, 1]
        """
        device = E_inv.device
        grid = bev.grid.to(device)
        _, H_bev, W_bev = grid.shape
        Q = H_bev * W_bev

        xy = rearrange(grid[:2], 'c H W -> (H W) c')  # [Q,2]

        cam_center = E_inv[..., :3, 3]                # [B,N,3], cam in world
        cam_xy = cam_center[..., :2]                  # [B,N,2]

        # [B,N,Q,2]
        vec = xy.unsqueeze(0).unsqueeze(0) - cam_xy.unsqueeze(2)
        dist = torch.norm(vec, dim=-1) + 1e-6         # [B,N,Q]

        dist_max = dist.amax(dim=(-2, -1), keepdim=True) + 1e-6
        dist_norm = (dist / dist_max).clamp(0., 1.)

        # 가까울수록 σ 큼(완만), 멀수록 σ 작음(날카로움)
        sigma_qi = self.max_sigma - dist_norm * (self.max_sigma - self.min_sigma)
        lambda_qi = 1.0 / (sigma_qi + 1e-6)

        return lambda_qi.unsqueeze(-1)                # [B,N,Q,1]

    # -------- EAF W_{q,k} 계산 --------
    def _compute_eaf_weights(self, bev: BEVEmbedding, I_inv, E_inv):
        """
        Returns:
          W_logits: [B, Q, NK]  with NK = N * (feat_h * feat_w)
        """
        device = I_inv.device
        B, N, _, _ = I_inv.shape

        grid = bev.grid.to(device)
        _, H_bev, W_bev = grid.shape
        Q = H_bev * W_bev

        xy = rearrange(grid[:2], 'c H W -> (H W) c')          # [Q,2]
        ones = torch.ones(Q, 1, device=device, dtype=grid.dtype)
        zeros = torch.zeros(Q, 1, device=device, dtype=grid.dtype)

        # 두 점으로 epipolar line 정의 (z=0, z=zmax)
        P0 = torch.cat([xy, zeros, ones], dim=-1)            # [Q,4]
        P1 = torch.cat([xy, self.eaf_zmax * ones, ones], dim=-1)

        P0 = P0.unsqueeze(0).unsqueeze(0).expand(B, N, Q, 4) # [B,N,Q,4]
        P1 = P1.unsqueeze(0).unsqueeze(0).expand(B, N, Q, 4)

        # world->cam
        E = torch.inverse(E_inv)                             # [B,N,4,4]
        P0_cam = torch.einsum('bnij,bnqj->bnqi', E, P0)
        P1_cam = torch.einsum('bnij,bnqj->bnqi', E, P1)

        # intrinsics
        I = torch.inverse(I_inv)                             # [B,N,3,3]
        p0 = torch.einsum('bnij,bnqj->bnqi', I, P0_cam[..., :3])
        p1 = torch.einsum('bnij,bnqj->bnqi', I, P1_cam[..., :3])

        p0 = p0 / (p0[..., 2:3] + 1e-8)
        p1 = p1 / (p1[..., 2:3] + 1e-8)

        # epipolar line (homogeneous line)
        l = torch.cross(p0, p1, dim=-1)                      # [B,N,Q,3]
        denom = torch.clamp(torch.sqrt(l[...,0]**2 + l[...,1]**2), min=1e-8)
        l_hat = l / denom.unsqueeze(-1)                      # [B,N,Q,3]

        # 이미지 feature 좌표
        _, _, h, w = self.image_plane.shape
        K_per = h * w
        x_img = rearrange(self.image_plane, '1 c h w -> (h w) c').to(device)  # [K_per,3]

        # 거리 d_{q,i,k}: [B,N,Q,K_per]
        d = torch.einsum('bnqj,kj->bnqk', l_hat, x_img).abs()

        # Adaptive λ_{q,i}
        if self.use_adaptive_lambda:
            lambda_qi = self._compute_lambda_qi(bev, E_inv)  # [B,N,Q,1]
        else:
            lambda_qi = torch.ones((B, N, Q, 1), device=device, dtype=d.dtype)

        lam = self.eaf_lambda
        scale = (lam * lambda_qi)**2                         # [B,N,Q,1]

        W = torch.exp(- scale * (d ** 2))                    # [B,N,Q,K_per]

        # 카메라+픽셀 축 flatten → [B,Q,NK]
        W = rearrange(W, 'b n Q K -> b Q (n K)')
        return W

    # -------- forward --------
    def forward(
        self,
        x: torch.FloatTensor,          # [B, D, H_bev, W_bev]
        bev: BEVEmbedding,
        feature: torch.FloatTensor,    # [B, N, C_in, h, w]
        I_inv: torch.FloatTensor,      # [B, N, 3, 3]
        E_inv: torch.FloatTensor,      # [B, N, 4, 4]
    ):
        B, N, C_in, h, w = feature.shape

        # 키/값 feature proj
        feat_flat = rearrange(feature, 'b n c h w -> (b n) c h w')
        val_flat = self.val_proj(feat_flat)                  # (B*N, D, h, w)

        if self.key_proj is not None:
            key_flat = self.key_proj(feat_flat)              # (B*N, D, h, w)
        else:
            key_flat = val_flat

        # reshape back to [B,N,D,h,w]
        v = rearrange(val_flat, '(b n) d h w -> b n d h w', b=B, n=N)
        k = rearrange(key_flat, '(b n) d h w -> b n d h w', b=B, n=N)

        # EAF weights
        W_logits = self._compute_eaf_weights(bev, I_inv, E_inv)  # [B,Q,NK]

        # Cross-attention (no positional enc; only EAF guides)
        out = self.cross_attend(
            q=x,
            k=k,
            v=v,
            W_logits=W_logits,
            skip=x if self.skip else None,
        )
        return out




class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w
        I_inv = batch['intrinsics'].inverse()           # b n 3 3
        E_inv = batch['extrinsics'].inverse()           # b n 4 4

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)

        return x
