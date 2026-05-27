import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np
from math import sqrt


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class MultiLagVariateEmbedding(nn.Module):
    """Generate K lagged versions of variate-level embeddings for exogenous variables.

    V2 improvements:
    - Fuses x_mark (time features) with exogenous data before projection (like DataEmbedding_inverted)
    - Uses truncation + learnable padding instead of zero-padding to avoid noise
    """

    def __init__(self, seq_len, d_model, num_lags, lag_step, dropout=0.1):
        super(MultiLagVariateEmbedding, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_lags = num_lags
        self.lag_step = lag_step
        # Main projection: both the plain and mark-augmented branches project along
        # the time axis, so the in_features are fixed at seq_len.
        self.value_embedding = nn.Linear(seq_len, d_model)
        # Learnable padding token for truncated sequences [1, 1, d_model]
        self.padding_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(p=dropout)
        self._mark_proj = nn.Linear(seq_len, d_model)
        nn.init.xavier_uniform_(self._mark_proj.weight)
        nn.init.zeros_(self._mark_proj.bias)

    def _get_mark_proj(self, total_dim, device):
        """Return the mark-augmented projection layer."""
        if self._mark_proj.in_features != total_dim:
            self._mark_proj = nn.Linear(total_dim, self.d_model).to(device)
            nn.init.xavier_uniform_(self._mark_proj.weight)
            nn.init.zeros_(self._mark_proj.bias)
        return self._mark_proj

    def forward(self, x, x_mark_enc=None):
        """Forward pass with optional time feature fusion.

        Args:
            x: [B, T, C] exogenous variables
            x_mark_enc: [B, T, M] time features (optional)
        Returns:
            [B, C*K, d_model] multi-lag variate embeddings
        """
        B, T, C = x.shape
        lag_embeds = []
        for k in range(self.num_lags):
            shift = k * self.lag_step
            if shift == 0:
                x_trunc = x  # [B, T, C]
                mark_trunc = x_mark_enc  # [B, T, M] or None
            elif shift < T:
                # Truncation: take x[:, :T-shift, :] (remove the latest `shift` steps)
                x_trunc = x[:, :T - shift, :]  # [B, T-shift, C]
                mark_trunc = x_mark_enc[:, :T - shift, :] if x_mark_enc is not None else None
            else:
                # All data shifted out — use only padding tokens
                pad_embed = self.padding_token.expand(B, C, -1)  # [B, C, d_model]
                lag_embeds.append(pad_embed)
                continue

            L_trunc = x_trunc.shape[1]

            if mark_trunc is not None:
                # Time features are fused later by concatenating mark channels along the
                # variate axis and projecting along the time axis, consistent with the
                # DataEmbedding_inverted style used by TimeXer. No separate pre-pass is needed here.
                pass

            # Project each variate's time series -> d_model
            x_var = x_trunc.permute(0, 2, 1)  # [B, C, L_trunc]

            if L_trunc == self.seq_len:
                # Full length — use standard projection
                if mark_trunc is not None:
                    # Fuse mark: each exo var gets mark features appended to its time series
                    # mark_trunc: [B, L_trunc, M] -> [B, M, L_trunc]
                    mark_var = mark_trunc.permute(0, 2, 1)  # [B, M, L_trunc]
                    # Concat: [B, C+M, L_trunc]
                    x_aug = torch.cat([x_var, mark_var], dim=1)  # [B, C+M, seq_len]
                    proj = self._get_mark_proj(self.seq_len, x.device)
                    x_proj_all = proj(x_aug)  # [B, C+M, d_model]
                    x_lag = x_proj_all[:, :C, :]  # Take only the C exo vars [B, C, d_model]
                else:
                    x_lag = self.value_embedding(x_var)  # [B, C, d_model]
            else:
                # Truncated — embed then pad with learnable tokens
                if mark_trunc is not None:
                    mark_var = mark_trunc.permute(0, 2, 1)  # [B, M, L_trunc]
                    x_aug = torch.cat([x_var, mark_var], dim=1)  # [B, C+M, L_trunc]
                    # Pad time dimension to seq_len
                    pad_size = self.seq_len - L_trunc
                    x_aug_padded = F.pad(x_aug, (0, pad_size), value=0.0)  # [B, C+M, seq_len]
                    proj = self._get_mark_proj(self.seq_len, x.device)
                    x_proj_all = proj(x_aug_padded)  # [B, C+M, d_model]
                    x_lag_proj = x_proj_all[:, :C, :]  # [B, C, d_model]
                else:
                    # Pad time dimension with zeros, then project
                    pad_size = self.seq_len - L_trunc
                    x_var_padded = F.pad(x_var, (0, pad_size), value=0.0)  # [B, C, seq_len]
                    x_lag_proj = self.value_embedding(x_var_padded)  # [B, C, d_model]

                # Blend with learnable padding token based on truncation ratio
                trunc_ratio = L_trunc / self.seq_len
                pad_token = self.padding_token.expand(B, C, -1)  # [B, C, d_model]
                x_lag = trunc_ratio * x_lag_proj + (1.0 - trunc_ratio) * pad_token

            lag_embeds.append(x_lag)
        out = torch.cat(lag_embeds, dim=1)  # [B, C*K, d_model]
        return self.dropout(out)


class CausalAdjacencyModule(nn.Module):
    """Learnable causal adjacency matrix for exogenous variables with lag structure.

    Implements joint (C+1) × (C+1) NOTEARS-style DAG constraint over:
      - node 0: endogenous target
      - nodes 1..C: C exogenous variables
    Edges target←exo via noisy-OR aggregation of K lag gates (A_raw).
    Edges exo_i←exo_j via B_raw (inter-exogenous adjacency).
    """

    def __init__(self, n_exo_vars, num_lags, temp_init=1.0, ablation='none', gate_init=2.0):
        super(CausalAdjacencyModule, self).__init__()
        self.n_exo_vars = n_exo_vars
        self.num_lags = num_lags
        self.ablation = ablation

        if ablation == 'bypass_causal':
            # Fix gate to all-1: sigmoid(10.0) ≈ 1.0, no gradient
            self.A_raw = nn.Parameter(torch.full((n_exo_vars, num_lags), 10.0), requires_grad=False)
        else:
            self.A_raw = nn.Parameter(torch.full((n_exo_vars, num_lags), float(gate_init)))

        self.temperature = temp_init

        # exo-exo adjacency — NOT registered as a Parameter here because it lives
        # inside ExoExoGraphMixer; we expose a setter so the parent Model can inject
        # the reference after construction.
        self._B_raw = None

    def set_exo_exo_params(self, B_raw_param):
        """Call after ExoExoGraphMixer is constructed so DAG loss sees B_raw."""
        self._B_raw = B_raw_param

    def get_causal_gate(self):
        return torch.sigmoid(self.A_raw)

    def get_lag_distribution(self):
        return F.softmax(self.A_raw / self.temperature, dim=-1)

    def get_causal_gate_flat(self):
        return self.get_causal_gate().reshape(-1)

    def _build_W_matrix(self):
        """Build (C+1)×(C+1) joint adjacency W with:
          - W[0,:] = 0 (target has no outgoing edges)
          - W[c+1,0] = noisy-OR over K lag gates (exo_c → target)
          - W[i+1,j+1] = B_raw[i,j] for i≠j (exo-exo)
          - diagonal = 0
        """
        C = self.n_exo_vars
        K = self.num_lags
        device = self.A_raw.device

        G = self.get_causal_gate()            # [C, K] in (0,1)
        # noisy-OR: 1 - ∏_k (1 - g_{c,k})
        noisy_or = 1.0 - torch.prod(1.0 - G + 1e-8, dim=-1)  # [C]

        W = torch.zeros(C + 1, C + 1, device=device)

        # Row 0: target has no outgoing edges
        W[0, :] = 0.0
        # Column 0: exo→target edges via noisy-OR
        W[1:, 0] = noisy_or

        # Inter-exogenous edges
        if self._B_raw is not None:
            B = torch.sigmoid(self._B_raw)   # [C, C]
            # Zero diagonal (no self-loops)
            B = B * (1.0 - torch.eye(C, device=device))
            W[1:, 1:] = B

        return W

    def compute_dag_constraint(self):
        """NOTEARS-style acyclicity: h(W) = Tr(e^{W∘W}) - (C+1) ≥ 0, =0 iff DAG."""
        if self.ablation == 'no_dag':
            return self.A_raw.new_zeros(())

        W = self._build_W_matrix()
        # Hadamard product W∘W
        WW = W * W
        # matrix exponential via torch (stable for C+1 ≤ ~50)
        try:
            h = torch.trace(torch.matrix_exp(WW)) - float(W.shape[0])
        except RuntimeError:
            # Fallback: very large C where matrix_exp is numerically unstable;
            # fall back to spectral radius estimate (rough but differentiable)
            eigenvalues = torch.linalg.eigvalsh(WW)
            h = torch.sum(torch.relu(eigenvalues)) - float(W.shape[0])
        return h

    def compute_sparse_loss(self):
        # L1 on gates + off-diagonal exo-exo edges
        loss = self.get_causal_gate().sum()
        if self._B_raw is not None:
            B = torch.sigmoid(self._B_raw)
            C = self.n_exo_vars
            B_no_diag = B * (1.0 - torch.eye(C, device=B.device))
            loss = loss + 0.5 * B_no_diag.sum()
        return loss


class CausalCrossAttention(nn.Module):
    """Cross-attention with causal gating via log-space additive masking."""

    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(CausalCrossAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, causal_gate=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if causal_gate is not None:
            gate_log = torch.log(causal_gate + 1e-8).reshape(1, 1, 1, S)
            scores = scores + gate_log
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class CausalAttentionLayer(nn.Module):
    """Attention layer wrapper that passes causal_gate to inner CausalCrossAttention."""

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(CausalAttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, causal_gate=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(
            queries, keys, values, attn_mask,
            tau=tau, delta=delta, causal_gate=causal_gate
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class CausalEncoderLayer(nn.Module):
    """Encoder layer with standard self-attention and causal cross-attention."""

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(CausalEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None, causal_gate=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x, attn_mask=x_mask, tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn, attn_weights = self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask, tau=tau, delta=delta,
            causal_gate=causal_gate
        )
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        # V2: add dropout to cross-attention output before residual (align with TimeXer EncoderLayer)
        x_glb = x_glb_ori + self.dropout(x_glb_attn)
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y), attn_weights


class CausalEncoder(nn.Module):
    """Encoder that collects cross-attention weights from each layer."""

    def __init__(self, layers, norm_layer=None):
        super(CausalEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None, causal_gate=None):
        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask,
                                    tau=tau, delta=delta, causal_gate=causal_gate)
            if attn_weights is not None:
                attn_weights_list.append(attn_weights)
        if self.norm is not None:
            x = self.norm(x)
        return x, attn_weights_list


class ExoExoGraphMixer(nn.Module):
    """Optional exo-exo mixing on variate tokens via a learnable adjacency.

    This module is intentionally lightweight and can be toggled off for ablation.
    It mixes only the original variate tokens (first C tokens) and leaves any
    extra tokens (e.g., time feature tokens produced by DataEmbedding_inverted)
    unchanged.
    """

    def __init__(self, n_vars: int):
        super().__init__()
        self.n_vars = int(n_vars)
        init = torch.eye(self.n_vars) * 2.0
        self.B_raw = nn.Parameter(init)

    def get_adj(self):
        # [C, C] in (0,1); row-normalize to keep scale stable
        B = torch.sigmoid(self.B_raw)
        B = B / (B.sum(dim=-1, keepdim=True) + 1e-8)
        return B

    def forward(self, ex_embed: torch.Tensor, base_var_count: int, num_lags: int):
        """Mix exogenous embeddings.

        Args:
            ex_embed: [B, S, D]
            base_var_count: number of original variates C
            num_lags: K for multi-lag mode (ignored if shape doesn't match C*K)
        """
        Bsz, S, D = ex_embed.shape
        C = int(base_var_count)
        if C <= 1:
            return ex_embed

        adj = self.get_adj()  # [C, C]

        # Multi-lag case: S == C*K
        if num_lags is not None and num_lags > 0 and S == C * int(num_lags):
            K = int(num_lags)
            x = ex_embed.view(Bsz, K, C, D)
            x = torch.einsum('ij,bkjd->bkid', adj, x)
            return x.reshape(Bsz, S, D)

        # TimeXer embedding case: tokens may be [C + M]
        if S >= C:
            x_vars = ex_embed[:, :C, :]
            x_tail = ex_embed[:, C:, :] if S > C else None
            x_vars = torch.einsum('ij,bjd->bid', adj, x_vars)
            if x_tail is None:
                return x_vars
            return torch.cat([x_vars, x_tail], dim=1)

        return ex_embed


class Model(nn.Module):
    """CausalExoFormer: TimeXer with causal discovery in cross-attention."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in

        # V2: ablation mode support
        self.ablation = getattr(configs, 'ablation', 'none')

        self.num_lags = getattr(configs, 'num_lags', 14)
        # V2: single_lag ablation forces num_lags=1
        if self.ablation == 'single_lag':
            self.num_lags = 1
        self.lag_step = getattr(configs, 'lag_step', 1)
        self.lambda_sparse = getattr(configs, 'lambda_sparse', 0.01)
        self.lambda_dag = getattr(configs, 'lambda_dag', 1.0)
        self.lambda_consist = getattr(configs, 'lambda_consist', 0.1)
        self.causal_top_k = int(getattr(configs, 'causal_top_k', 0) or 0)

        if configs.features == 'MS':
            self.n_exo_vars = configs.enc_in - 1
        else:
            self.n_exo_vars = configs.enc_in

        # Optional exo-exo graph mixer (default off for backward compatibility)
        self.use_exo_exo_graph = bool(getattr(configs, 'use_exo_exo_graph', 0))
        if self.ablation == 'no_exo_exo_graph':
            self.use_exo_exo_graph = False
        self.exo_exo_mixer = ExoExoGraphMixer(self.n_exo_vars) if self.use_exo_exo_graph else None

        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)

        # V2: ablation - use DataEmbedding_inverted when requested
        if self.ablation == 'timexer_embed':
            self.ex_embedding = DataEmbedding_inverted(
                configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
            self._use_timexer_embed = True
        else:
            self.ex_embedding = MultiLagVariateEmbedding(
                seq_len=configs.seq_len, d_model=configs.d_model,
                num_lags=self.num_lags, lag_step=self.lag_step, dropout=configs.dropout
            )
            self._use_timexer_embed = False

        self.causal_adj = CausalAdjacencyModule(
            n_exo_vars=self.n_exo_vars, num_lags=self.num_lags,
            temp_init=getattr(configs, 'temp_init', 1.0),
            ablation=self.ablation,
            gate_init=getattr(configs, 'causal_gate_init', 2.0)
        )
        if self.exo_exo_mixer is not None:
            self.causal_adj.set_exo_exo_params(self.exo_exo_mixer.B_raw)
        self.encoder = CausalEncoder(
            [
                CausalEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    CausalAttentionLayer(
                        CausalCrossAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=True),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)
        self.use_fft_residual = bool(getattr(configs, 'fft_residual', 0))
        self.use_multiscale_residual = bool(getattr(configs, 'multi_scale_residual', 0))
        self.use_linear_residual = bool(getattr(configs, 'linear_residual', 0)) or self.use_multiscale_residual or self.use_fft_residual
        self.use_seasonal_residual = bool(getattr(configs, 'linear_residual_seasonal', 0))
        self.two_stage_residual = bool(getattr(configs, 'two_stage_residual', 0))
        residual_init = float(getattr(configs, 'linear_residual_init', 0.1))
        if self.use_linear_residual:
            if self.use_fft_residual:
                self.fft_top_k = int(getattr(configs, 'fft_residual_top_k', 5))
                self.fft_trend_linear = nn.Linear(configs.seq_len, configs.pred_len)
                self.fft_seasonal_linear = nn.Linear(configs.seq_len, configs.pred_len)
                self._init_average_linear(self.fft_trend_linear, configs.seq_len)
                self._init_average_linear(self.fft_seasonal_linear, configs.seq_len)
                self.fft_trend_alpha = nn.Parameter(torch.tensor(residual_init))
                self.fft_seasonal_alpha = nn.Parameter(torch.tensor(residual_init))
            elif self.use_multiscale_residual:
                self.ms_windows = self._parse_multiscale_windows(getattr(configs, 'multi_scale_windows', '13,25,48,96'))
                self.ms_trend_lines = nn.ModuleList([nn.Linear(configs.seq_len, configs.pred_len) for _ in self.ms_windows])
                self.ms_seasonal_lines = nn.ModuleList([nn.Linear(configs.seq_len, configs.pred_len) for _ in self.ms_windows])
                self.ms_trend_alphas = nn.ParameterList([
                    nn.Parameter(torch.tensor(residual_init)) for _ in self.ms_windows
                ])
                self.ms_seasonal_alphas = nn.ParameterList([
                    nn.Parameter(torch.tensor(residual_init)) for _ in self.ms_windows
                ])
                self.ms_scale_logits = nn.Parameter(torch.zeros(len(self.ms_windows)))
                for line in self.ms_trend_lines:
                    self._init_average_linear(line, configs.seq_len)
                for line in self.ms_seasonal_lines:
                    self._init_average_linear(line, configs.seq_len)
            elif self.use_seasonal_residual:
                self.ma_window = int(getattr(configs, 'linear_residual_seasonal_ma', 25))
                self.trend_linear = nn.Linear(configs.seq_len, configs.pred_len)
                self.seasonal_linear = nn.Linear(configs.seq_len, configs.pred_len)
                self._init_average_linear(self.trend_linear, configs.seq_len)
                self._init_average_linear(self.seasonal_linear, configs.seq_len)
                self.trend_alpha = nn.Parameter(torch.tensor(residual_init))
                self.seasonal_alpha = nn.Parameter(torch.tensor(residual_init))
            else:
                self.linear_residual = nn.Linear(configs.seq_len, configs.pred_len)
                self._init_average_linear(self.linear_residual, configs.seq_len)
                self.linear_residual_alpha = nn.Parameter(torch.tensor(residual_init))
        self._last_attn_weights = None

        # ============ A2: Adaptive residual gating (data-dependent blend) ============
        # Gate computed from encoder output → controls how much to trust linear residual
        self.use_adaptive_gate = bool(getattr(configs, 'adaptive_residual_gate', 0))
        if self.use_adaptive_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model // 2),
                nn.GELU(),
                nn.Linear(configs.d_model // 2, 1),
            )
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.zeros_(self.gate_mlp[-1].bias)

        # ============ A1: RevIN affine (learnable gamma/beta after instance-norm) ============
        # Default off for backward compatibility; only active when configs.revin_affine == 1
        self.use_revin_affine = bool(getattr(configs, 'revin_affine', 0))
        if self.use_revin_affine:
            n_features = configs.enc_in
            self.revin_gamma = nn.Parameter(torch.ones(n_features))
            self.revin_beta = nn.Parameter(torch.zeros(n_features))

    @staticmethod
    def _init_average_linear(linear: nn.Linear, seq_len: int):
        nn.init.constant_(linear.weight, 1.0 / seq_len)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

    @staticmethod
    def _parse_multiscale_windows(windows):
        if isinstance(windows, str):
            parsed = [int(w.strip()) for w in windows.split(',') if w.strip()]
        elif isinstance(windows, (list, tuple)):
            parsed = [int(w) for w in windows]
        else:
            parsed = [int(windows)]
        parsed = [w for w in parsed if w > 0]
        if not parsed:
            raise ValueError('multi_scale_windows must contain at least one positive integer')
        return parsed

    def _moving_avg(self, x, window):
        """Compute edge-padded moving average along time dim. x: [B, T, C]"""
        if window <= 1:
            return x
        if x.shape[1] < window:
            return x.mean(dim=1, keepdim=True).repeat(1, x.shape[1], 1)
        left_pad = (window - 1) // 2
        right_pad = window - 1 - left_pad
        front = x[:, 0:1, :].repeat(1, left_pad, 1) if left_pad > 0 else x[:, :0, :]
        end = x[:, -1:, :].repeat(1, right_pad, 1) if right_pad > 0 else x[:, :0, :]
        padded = torch.cat([front, x, end], dim=1)
        avg = F.avg_pool1d(padded.permute(0, 2, 1), kernel_size=window, stride=1)
        return avg.permute(0, 2, 1)

    def _compute_seasonal_residual(self, target: torch.Tensor) -> torch.Tensor:
        """DLinear-style trend + seasonal decomposition.
        target: [B, seq_len, 1]
        Returns: [B, pred_len, 1]
        """
        t = target.squeeze(-1)                              # [B, seq_len]
        ma = self._moving_avg(t.unsqueeze(-1), self.ma_window)  # [B, seq_len, 1]
        ma = ma.squeeze(-1)                                  # [B, seq_len]
        trend = ma                                           # [B, seq_len]
        seasonal = t - trend                                # [B, seq_len]
        pred_trend = self.trend_linear(trend)                # [B, pred_len]
        pred_seasonal = self.seasonal_linear(seasonal)        # [B, pred_len]
        return (self.trend_alpha * pred_trend + self.seasonal_alpha * pred_seasonal).unsqueeze(-1)

    def _compute_fft_residual(self, target: torch.Tensor) -> torch.Tensor:
        t = target.squeeze(-1)
        xf = torch.fft.rfft(t, dim=1)
        freq = torch.abs(xf)
        if freq.shape[1] > 0:
            freq[:, 0] = 0
        k = min(max(int(self.fft_top_k), 0), max(freq.shape[1] - 1, 0))
        if k > 0:
            top_idx = torch.topk(freq, k=k, dim=1).indices
            mask = torch.zeros_like(freq, dtype=torch.bool)
            mask.scatter_(1, top_idx, True)
            if mask.shape[1] > 0:
                mask[:, 0] = False
            seasonal = torch.fft.irfft(xf * mask.to(dtype=xf.dtype), n=t.shape[1], dim=1)
        else:
            seasonal = torch.zeros_like(t)
        trend = t - seasonal
        pred_trend = self.fft_trend_linear(trend)
        pred_seasonal = self.fft_seasonal_linear(seasonal)
        return (self.fft_trend_alpha * pred_trend + self.fft_seasonal_alpha * pred_seasonal).unsqueeze(-1)

    def _compute_multiscale_residual(self, target: torch.Tensor) -> torch.Tensor:
        t = target.squeeze(-1)
        weights = F.softmax(self.ms_scale_logits, dim=0)
        residual = None
        for i, window in enumerate(self.ms_windows):
            trend = self._moving_avg(t.unsqueeze(-1), window).squeeze(-1)
            seasonal = t - trend
            pred_trend = self.ms_trend_lines[i](trend)
            pred_seasonal = self.ms_seasonal_lines[i](seasonal)
            scale_residual = (self.ms_trend_alphas[i] * pred_trend + self.ms_seasonal_alphas[i] * pred_seasonal).unsqueeze(-1)
            weighted = weights[i] * scale_residual
            residual = weighted if residual is None else residual + weighted
        return residual

    def _compose_residual(self, dec_out, residual, gate=None):
        if self.two_stage_residual:
            return residual.detach() + dec_out
        if gate is not None:
            gate = gate.view(-1, 1, 1)
            return gate * dec_out + (1 - gate) * residual
        return dec_out + residual

    def _select_top_exogenous(self, x_exo, x_target):
        if self.causal_top_k <= 0 or x_exo.shape[2] <= self.causal_top_k:
            return x_exo, None
        with torch.no_grad():
            exo = x_exo - x_exo.mean(dim=1, keepdim=True)
            target = x_target - x_target.mean(dim=1, keepdim=True)
            corr = (exo * target).mean(dim=1).abs()
            corr = corr.mean(dim=0)
            idx = torch.topk(corr, k=self.causal_top_k, largest=True).indices.sort().values
        return x_exo.index_select(dim=2, index=idx), idx

    def _map_causal_gate_rows(self, idx):
        gate = self.causal_adj.get_causal_gate()
        if idx is None:
            return gate
        mapped = gate.new_ones(idx.shape[0], gate.shape[1])
        selected = gate.index_select(dim=0, index=idx)
        return mapped * selected.detach().mean() + selected

    def _get_causal_gate_for_cross_attention(self, ex_embed: torch.Tensor, x_mark_enc, exo_idx=None):
        """Build a causal gate vector aligned with the cross-attention source length S."""
        S = ex_embed.shape[1]
        if not self._use_timexer_embed:
            gate = self._map_causal_gate_rows(exo_idx).reshape(-1)
            return gate if gate.shape[0] == S else None

        # timexer_embed: source tokens are [C (+ M)]
        gate_exo = self._map_causal_gate_rows(exo_idx).max(dim=1).values  # [C]
        if S == gate_exo.shape[0]:
            return gate_exo
        if S > gate_exo.shape[0]:
            tail = torch.ones(S - gate_exo.shape[0], device=gate_exo.device)
            return torch.cat([gate_exo, tail], dim=0)
        # Unexpected: fewer tokens than vars
        return gate_exo[:S]

    def _maybe_apply_exo_exo_graph(self, ex_embed: torch.Tensor, base_var_count: int):
        if self.exo_exo_mixer is None:
            return ex_embed
        return self.exo_exo_mixer(ex_embed, base_var_count=base_var_count, num_lags=(None if self._use_timexer_embed else self.num_lags))

    def _embed_exogenous(self, x_exo, x_mark_enc):
        """V2: Unified exogenous embedding with x_mark fusion and ablation support.

        Returns:
            ex_embed: [B, S, d_model]
            base_var_count: original number of variates C in x_exo
        """
        base_var_count = x_exo.shape[2]
        if self._use_timexer_embed:
            # timexer_embed ablation: use DataEmbedding_inverted directly
            return self.ex_embedding(x_exo, x_mark_enc), base_var_count
        else:
            # MultiLagVariateEmbedding with x_mark fusion
            return self.ex_embedding(x_exo, x_mark_enc=x_mark_enc), base_var_count

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            if self.use_revin_affine:
                # apply learnable affine on each feature
                x_enc = x_enc * self.revin_gamma.view(1, 1, -1) + self.revin_beta.view(1, 1, -1)
        _, _, N = x_enc.shape
        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        # V2: pass x_mark_enc to exogenous embedding
        x_exo, exo_idx = self._select_top_exogenous(x_enc[:, :, :-1], x_enc[:, :, -1:])
        ex_embed, base_var_count = self._embed_exogenous(x_exo, x_mark_enc)
        ex_embed = self._maybe_apply_exo_exo_graph(ex_embed, base_var_count=base_var_count)
        causal_gate = self._get_causal_gate_for_cross_attention(ex_embed, x_mark_enc, exo_idx=exo_idx)
        enc_out, attn_weights_list = self.encoder(en_embed, ex_embed, causal_gate=causal_gate)
        if self.training and len(attn_weights_list) > 0:
            self._last_attn_weights = attn_weights_list
        if self.use_adaptive_gate:
            # Pool enc_out [B*n_vars, L, d_model] → [B, d_model] for gate
            enc_for_gate = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
            enc_for_gate = enc_for_gate.mean(dim=1).mean(dim=1)  # [B, d_model]
            g = torch.sigmoid(self.gate_mlp(enc_for_gate))  # [B, 1]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_linear_residual:
            if self.use_fft_residual:
                residual = self._compute_fft_residual(x_enc[:, :, -1:])
            elif self.use_multiscale_residual:
                residual = self._compute_multiscale_residual(x_enc[:, :, -1:])
            elif self.use_seasonal_residual:
                residual = self._compute_seasonal_residual(x_enc[:, :, -1:])
            else:
                residual = self.linear_residual_alpha * self.linear_residual(x_enc[:, :, -1].contiguous()).unsqueeze(-1)
            dec_out = self._compose_residual(dec_out, residual, g if self.use_adaptive_gate else None)
        if self.use_norm:
            if self.use_revin_affine:
                # invert affine: subtract beta then divide by gamma (only for last channel = target)
                dec_out = dec_out - self.revin_beta[-1:].view(1, 1, -1)
                dec_out = dec_out / (self.revin_gamma[-1:].view(1, 1, -1) + 1e-8)
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            if self.use_revin_affine:
                x_enc = x_enc * self.revin_gamma.view(1, 1, -1) + self.revin_beta.view(1, 1, -1)
        _, _, N = x_enc.shape
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        # V2: pass x_mark_enc to exogenous embedding
        x_exo, exo_idx = self._select_top_exogenous(x_enc, x_enc.mean(dim=2, keepdim=True))
        ex_embed, base_var_count = self._embed_exogenous(x_exo, x_mark_enc)
        ex_embed = self._maybe_apply_exo_exo_graph(ex_embed, base_var_count=base_var_count)
        causal_gate = self._get_causal_gate_for_cross_attention(ex_embed, x_mark_enc, exo_idx=exo_idx)
        enc_out, attn_weights_list = self.encoder(en_embed, ex_embed, causal_gate=causal_gate)
        if self.training and len(attn_weights_list) > 0:
            self._last_attn_weights = attn_weights_list
        if self.use_adaptive_gate:
            # Pool enc_out [B*n_vars, L, d_model] → [B, d_model] for gate
            enc_for_gate = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
            enc_for_gate = enc_for_gate.mean(dim=1).mean(dim=1)  # [B, d_model]
            g = torch.sigmoid(self.gate_mlp(enc_for_gate))  # [B, 1]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_linear_residual:
            if self.use_fft_residual:
                residual = self._compute_fft_residual(x_enc.mean(dim=2, keepdim=True))
            elif self.use_multiscale_residual:
                residual = self._compute_multiscale_residual(x_enc.mean(dim=2, keepdim=True))
            elif self.use_seasonal_residual:
                residual = self._compute_seasonal_residual(x_enc.mean(dim=2, keepdim=True))
            else:
                residual = self.linear_residual_alpha * self.linear_residual(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
            dec_out = self._compose_residual(dec_out, residual, g if self.use_adaptive_gate else None)
        if self.use_norm:
            if self.use_revin_affine:
                dec_out = dec_out - self.revin_beta.view(1, 1, -1)
                dec_out = dec_out / (self.revin_gamma.view(1, 1, -1) + 1e-8)
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]
        else:
            return None

    def get_causal_losses(self):
        losses = {}
        losses['sparse_loss'] = self._compute_sparse_loss()
        # no_dag ablation skips DAG constraint
        if self.ablation == 'no_dag':
            losses['dag_constraint'] = self.causal_adj.A_raw.new_zeros(())
        else:
            losses['dag_constraint'] = self.causal_adj.compute_dag_constraint()
        if self._last_attn_weights is not None and len(self._last_attn_weights) > 0:
            losses['consist_loss'] = self._compute_consistency_loss(self._last_attn_weights)
        else:
            losses['consist_loss'] = self.causal_adj.A_raw.new_zeros(())
        # expose exo-exo adjacency for saving / analysis
        if self.exo_exo_mixer is not None:
            losses['exo_graph'] = self.exo_exo_mixer.get_adj().detach().cpu()
        else:
            losses['exo_graph'] = None
        return losses

    def _compute_sparse_loss(self):
        gate = self.causal_adj.get_causal_gate()
        if self.causal_top_k > 0 and gate.shape[0] > self.causal_top_k:
            var_strength = gate.max(dim=1).values
            keep = torch.topk(var_strength, k=self.causal_top_k, largest=True).indices
            sparse_loss = gate.index_select(dim=0, index=keep).sum()
        else:
            sparse_loss = gate.sum()
        if self.exo_exo_mixer is not None:
            B = torch.sigmoid(self.exo_exo_mixer.B_raw)
            C = self.n_exo_vars
            B_no_diag = B * (1.0 - torch.eye(C, device=B.device))
            sparse_loss = sparse_loss + 0.5 * B_no_diag.sum()
        return sparse_loss

    def _compute_consistency_loss(self, attn_weights_list):
        """V2: Fixed consistency loss with correct dimension handling.

        attn_weights shape per layer: [B*n_vars, H, L_query, S_source]
        where S_source = C*K (exo vars * lags)
        We average over batch, heads, query dims to get a distribution over S_source.
        """
        if len(attn_weights_list) == 0:
            return torch.tensor(0.0, device=self.causal_adj.A_raw.device)

        # Stack layers and average: [num_layers, B*n_vars, H, L_q, S]
        attn_avg = torch.stack(attn_weights_list, dim=0).mean(dim=0)  # [B*n_vars, H, L_q, S]
        # Average over batch*n_vars (dim 0), heads (dim 1), query positions (dim 2)
        # Keep source dimension S = C*K
        attn_avg = attn_avg.mean(dim=0).mean(dim=0).mean(dim=0)  # [S]

        # Ensure it's a valid probability distribution
        attn_dist = attn_avg / (attn_avg.sum() + 1e-8)
        attn_dist = attn_dist.clamp(min=1e-8)

        # Gate distribution aligned to attention source length
        if self._use_timexer_embed:
            gate_exo = self.causal_adj.get_causal_gate().max(dim=1).values  # [C]
            if attn_dist.shape[0] > gate_exo.shape[0]:
                tail = torch.ones(attn_dist.shape[0] - gate_exo.shape[0], device=gate_exo.device)
                gate = torch.cat([gate_exo, tail], dim=0)
            else:
                gate = gate_exo[:attn_dist.shape[0]]
        else:
            gate = self.causal_adj.get_causal_gate_flat()  # [C*K]
            if attn_dist.shape[0] != gate.shape[0]:
                return torch.tensor(0.0, device=self.causal_adj.A_raw.device)

        gate_dist = gate / (gate.sum() + 1e-8)
        gate_dist = gate_dist.clamp(min=1e-8)

        # KL(attention || gate): attention acts as the teacher signal; this avoids
        # pushing every causal gate toward zero when sparsity is enabled.
        kl_loss = F.kl_div(gate_dist.log(), attn_dist.detach(), reduction='sum')
        return kl_loss
