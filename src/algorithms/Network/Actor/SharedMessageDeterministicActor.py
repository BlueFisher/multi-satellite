import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Conv1D


COMMUNICATION_ABLATION_MODES = (
    "full_communication",
    "sparse_communication",
    "no_communication",
    "shuffle_message",
)


def apply_communication_ablation(
    message: torch.Tensor,
    communication_edges: torch.Tensor,
    mode: str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an evaluation-time intervention to inter-agent communication."""

    if mode not in COMMUNICATION_ABLATION_MODES:
        raise ValueError(
            f"unknown communication ablation {mode!r}; "
            f"expected one of {COMMUNICATION_ABLATION_MODES}"
        )
    if mode == "sparse_communication":
        return message, communication_edges
    if mode == "full_communication":
        return message, torch.ones_like(communication_edges, dtype=torch.bool)
    if mode == "no_communication":
        n_agents = communication_edges.shape[-1]
        identity = torch.eye(
            n_agents, dtype=torch.bool, device=communication_edges.device
        )
        effective_edges = identity.expand_as(communication_edges).clone()
        return message, effective_edges
    n_agents = message.shape[1]
    if n_agents < 2:
        raise ValueError("shuffle_message requires at least two agents")
    offset = int(
        torch.randint(1, n_agents, (1,), generator=generator, device="cpu").item()
    )
    return torch.roll(message, shifts=offset, dims=1), communication_edges


class GraphTransformerLayer(nn.Module):

    def __init__(
        self,
        embed_dim,
        ff_embed_dim,
        num_heads,
        n_agent,
        self_loop_add,
        dropout=0.1,
        weights_dropout=False,
        masked=False,
    ):
        super(GraphTransformerLayer, self).__init__()
        self.self_attn = RelationMultiheadAttention(
            embed_dim,
            num_heads,
            n_agent,
            dropout,
            weights_dropout,
            masked,
            self_loop_add,
        )
        self.fc1 = Conv1D(ff_embed_dim, embed_dim)
        self.fc2 = Conv1D(embed_dim, ff_embed_dim)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(
        self, x, relation, kv=None, attn_mask=None, need_weights=False, dec_agent=False
    ):
        # x: seq_len x bsz x embed_dim
        residual = x
        if kv is None:
            x, self_attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                relation=relation,
                attn_mask=attn_mask,
                need_weights=need_weights,
                dec_agent=dec_agent,
            )
        else:
            x, self_attn = self.self_attn(
                query=x,
                key=kv,
                value=kv,
                relation=relation,
                attn_mask=attn_mask,
                need_weights=need_weights,
                dec_agent=dec_agent,
            )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attn_layer_norm(residual + x)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_layer_norm(residual + x)
        return x, self_attn


class RelationMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        n_agent,
        dropout=0.0,
        weights_dropout=False,
        masked=False,
        self_loop_add=True,
    ):
        super(RelationMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.masked = masked
        self.self_loop_add = self_loop_add

        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        self.relation_in_proj = Conv1D(2 * embed_dim, embed_dim)

        self.out_proj = Conv1D(embed_dim, embed_dim)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

        self.register_buffer("mask", torch.tril(torch.ones(n_agent, n_agent)) == 0)

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.normal_(self.relation_in_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        relation,
        attn_mask=None,
        need_weights=False,
        dec_agent=False,
    ):
        """Input shape: Time x Batch x Channel
        relation:  tgt_len x src_len x bsz x n_emd
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)

        if relation is None:
            # q *= self.scaling
            attn_weights = torch.einsum("ibn,jbn->ijb", [q, k]) * (
                1.0 / math.sqrt(k.size(-1))
            )
        else:
            ra, rb = self.relation_in_proj(relation).chunk(2, dim=-1)
            ra = (
                ra.contiguous()
                .view(tgt_len, src_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
            rb = (
                rb.contiguous()
                .view(tgt_len, src_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

            q = q.unsqueeze(1) + ra
            k = k.unsqueeze(0) + rb
            q *= self.scaling
            # q: tgt_len x src_len x bsz*heads x dim
            # k: tgt_len x src_len x bsz*heads x dim
            # v: src_len x bsz*heads x dim

            attn_weights = torch.einsum("ijbn,ijbn->ijb", [q, k]) * (
                1.0 / math.sqrt(k.size(-1))
            )

        assert list(attn_weights.size()) == [tgt_len, src_len, bsz * self.num_heads]

        if self.masked:
            attn_weights.masked_fill_(self.mask.unsqueeze(-1), float("-inf"))

            # attn_weights = attn_weights.view(tgt_len, src_len, bsz * self.num_heads)

        if dec_agent is True:
            self_loop = (
                torch.eye(tgt_len).unsqueeze(-1).long().to(device=attn_weights.device)
            )
            if self.self_loop_add:
                attn_mask = attn_mask + self_loop
            else:
                attn_mask = attn_mask * (1 - self_loop) + self_loop
            attn_weights.masked_fill_(attn_mask == 0, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=1)
            attn_weights = attn_weights * attn_mask
        else:
            attn_weights = F.softmax(attn_weights, dim=1)

        if self.weights_dropout:
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

        # attn_weights: tgt_len x src_len x bsz*heads
        # v: src_len x bsz*heads x dim
        attn = torch.einsum("ijb,jbn->bin", [attn_weights, v])
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # maximum attention weight over heads
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class SharedMessageDeterministicActor(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, num_agents, hidden_dim=64, args=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.use_relation = args.use_relation
        self.relation_topk: float = args.relation_topk
        self.use_ndq = args.use_ndq
        self.use_message_gate = args.use_message_gate
        self.device = args.device
        self.communication_ablation = getattr(
            args, "communication_ablation", "sparse_communication"
        )
        self.communication_ablation_generator = torch.Generator(device="cpu")
        self.communication_ablation_generator.manual_seed(
            int(getattr(args, "communication_ablation_seed", 20260802))
        )
        self.num_layers = getattr(args, "num_relation_layers", 1)  # 图Transformer层数

        self.fc1 = nn.Linear(obs_dim, hidden_dim)

        if self.use_relation:
            # 可学习的 logits，用于生成稀疏掩码（保持你原有的方式）
            self.relation_logits = nn.Parameter(torch.ones(num_agents, num_agents))

            # 将二值掩码 (0/1) 映射为关系向量的嵌入层
            self.relation_embedding = nn.Embedding(2, hidden_dim)

            # 堆叠多个 GraphTransformerLayer
            self.encoder_layers = nn.ModuleList(
                [
                    GraphTransformerLayer(
                        embed_dim=hidden_dim,
                        ff_embed_dim=hidden_dim * 4,  # 通常扩大4倍
                        num_heads=1,
                        n_agent=num_agents,
                        self_loop_add=True,  # 确保自环不被屏蔽
                        dropout=0.1,
                        weights_dropout=False,
                        masked=False,
                    )
                    for _ in range(self.num_layers)
                ]
            )

            if self.use_ndq:
                self.message_encoder = nn.Sequential(
                    nn.Linear(obs_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                )

                self.message_decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                )

        else:
            # 无关系时使用标准 TransformerEncoderLayer
            self.encoder = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=1,
                dim_feedforward=hidden_dim,
                batch_first=True,
            )

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
        self.args = args

    def relations_top_k(self, logits, top=0.4, temperature=1.0):
        """
        你原有的方法：生成硬掩码（0/1），1 表示屏蔽该连接。
        logits: (batch, n, n)  或 (n, n)
        返回: (batch, n, n) 值为 0 或 1 的浮点张量（可微）
        """
        # 如果 logits 是 2D，扩展 batch 维度
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        batch_size, n, _ = logits.shape
        k_keep = int(np.floor(top * n))
        k_keep = max(0, min(k_keep, n))

        # Gumbel 噪声
        uniform = torch.rand_like(logits)
        uniform = torch.clamp(uniform, min=1e-20, max=1.0 - 1e-20)
        gumbel_noise = -torch.log(-torch.log(uniform))
        perturbed = (logits + gumbel_noise) / temperature

        soft_probs = torch.sigmoid(perturbed)  # (batch, n, n)

        if k_keep == 0:
            hard_mask = torch.ones_like(soft_probs)
        elif k_keep == n:
            hard_mask = torch.zeros_like(soft_probs)
        else:
            topk_vals, topk_idx = torch.topk(soft_probs, k_keep, dim=2)
            hard_mask = torch.ones_like(soft_probs)
            hard_mask.scatter_(2, topk_idx, 0.0)

        # 直通估计
        output = hard_mask + (soft_probs - soft_probs.detach())
        return output

    def forward(self, obs):
        """
        obs: (batch, num_agents, obs_dim)
        """
        batch_size = obs.size(0)
        n = self.num_agents

        embedded = self.fc1(obs)  # (batch, n, hidden)

        extra = {}

        if self.use_relation:
            # 1. 生成二值掩码 (batch, n, n) ，其中 1 表示屏蔽
            binary_mask = self.relations_top_k(
                self.relation_logits.unsqueeze(0).expand(batch_size, -1, -1),
                top=self.relation_topk,
                temperature=1.0,
            )  # (batch, n, n)

            # 2. 将二值掩码转换为关系向量 (batch, n, n, hidden)
            #    注意：binary_mask 是浮点 0/1，需转为 long 索引
            relation_embed = self.relation_embedding(
                binary_mask.long()
            )  # (batch, n, n, hidden)

            # 3. 转换为 GraphTransformerLayer 要求的维度
            #    x: (seq_len, batch, hidden) -> (n, batch, hidden)
            x = embedded.permute(1, 0, 2)  # (n, batch, hidden)

            #    relation: (tgt_len, src_len, batch, hidden) -> (n, n, batch, hidden)
            relation = relation_embed.permute(
                1, 2, 0, 3
            ).contiguous()  # (n, n, batch, hidden)

            #    attn_mask: (tgt_len, src_len, batch), 0 表示屏蔽, 1 表示保留
            #    注意：binary_mask 中 1 表示屏蔽，0 表示保留,
            attn_mask = 1 - binary_mask  # (batch, n, n)
            communication_edges = attn_mask.bool()
            diagonal = torch.arange(n, device=communication_edges.device)
            communication_edges[:, diagonal, diagonal] = True
            extra["communication_edges"] = communication_edges
            attn_mask = attn_mask.permute(1, 2, 0)  # (n, n, batch)
            if self.use_ndq:
                t = self.message_encoder(obs)

                mu = t[:, :, : self.hidden_dim]
                log_std = t[:, :, self.hidden_dim :]
                message_dist = torch.distributions.Normal(mu, torch.exp(log_std))

                message = message_dist.rsample()

                if self.use_message_gate:
                    message = 2 * message * torch.sigmoid(message)

                message, communication_edges = apply_communication_ablation(
                    message,
                    communication_edges,
                    self.communication_ablation,
                    self.communication_ablation_generator,
                )
                binary_mask = (~communication_edges).long()
                relation_embed = self.relation_embedding(binary_mask)
                relation = relation_embed.permute(1, 2, 0, 3).contiguous()
                attn_mask = communication_edges.permute(1, 2, 0)
                extra["communication_edges"] = communication_edges
                decode_action = self.message_decoder(message)
                extra["message"] = message

                kv_message = message.permute(1, 0, 2)
                for layer in self.encoder_layers:
                    x, _ = layer(
                        x,
                        kv=kv_message,
                        relation=relation,
                        attn_mask=attn_mask,
                        dec_agent=True,
                    )

                extra.update(
                    {"mu": mu, "log_std": log_std, "decode_action": decode_action}
                )

            else:
                # 4. 逐层通过 GraphTransformerLayer
                for layer in self.encoder_layers:
                    x, _ = layer(
                        x, relation=relation, attn_mask=attn_mask, dec_agent=True
                    )

            encoded = x.permute(1, 0, 2)  # (batch, n, hidden)
        else:
            # 无关系时直接使用标准 Transformer
            encoded = self.encoder(embedded, src_mask=None)

        hidden = F.relu(self.fc2(encoded))
        actions = self.tanh(self.fc_out(hidden))

        return actions, extra
