import torch
import torch.nn.functional as F
import math
import torch.nn as nn

def scaled_doc_product_attntion(Q, K, V, mask=None):
    """
    缩放点积注意力计算。

    参数:
        Q: 查询矩阵 (batch_size, seq_len_q, embed_size)
        K: 键矩阵 (batch_size, seq_len_k, embed_size)
        V: 值矩阵 (batch_size, seq_len_v, embed_size)
        mask: 掩码矩阵，用于屏蔽不应该关注的位置 (可选)

    返回:
        output: 注意力加权后的输出矩阵
        attention_weights: 注意力权重矩阵
    """
    embed_size = Q.size(-1)
    # 计算点积并进行缩放
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(embed_size)
    # 如果提供了掩码矩阵，则将掩码对应位置的分数设为 -inf
    if mask is not None:
        scores.masked_fill(mask == 0, float("-inf"))
    # 对缩放后的分数应用 Softmax 函数，得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    # 加权求和，计算输出
    context = torch.matmul(attention_weights, V)
    return context, attention_weights

class Attention(nn.Module):
    def __init__(self, embed_size):
        """
        单头注意力机制。

        参数:
            embed_size: 输入序列（Inputs）的嵌入（Input Embedding）维度，也是论文中所提到的d_model。
        """
        super(Attention, self).__init__()
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask=None):
        """
        前向传播函数。

        参数:
            q: 查询矩阵 (batch_size, seq_len_q, embed_size)
            k: 键矩阵 (batch_size, seq_len_k, embed_size)
            v: 值矩阵 (batch_size, seq_len_v, embed_size)
            mask: 掩码矩阵，用于屏蔽不应关注的位置 (batch_size, seq_len_q, seq_len_k)

        返回:
            out: 注意力加权后的输出
            attention_weights: 注意力权重矩阵
        """
        # 将输入序列通过线性变换生成 Q, K, V
        Q = self.w_q(q) # (batch_size, seq_len_q, embed_size)
        K = self.w_k(k) # (batch_size, seq_len_q, embed_size)
        V = self.w_v(v) # (batch_size, seq_len_q, embed_size)
        # 使用缩放点积注意力函数计算输出和权重
        out, attn = scaled_doc_product_attntion(Q, K, V, mask)
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, n_heads):
        """
        多头注意力机制。（暂时使用更复杂的变量名来减少理解难度，在最后将统一映射到论文的表达）
        参数:
            embed_size: 输入序列的嵌入维度。
            num_heads: 注意力头的数量，对应于数学公式中的 h。
        """
        super(MultiHeadAttention, self).__init__()
        self.enbed_size = embed_size
        self.n_heads = n_heads
        self.head_dim = embed_size // n_heads
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)
        self.linear = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, attn_mask):
        batch_size = q.size(0)
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        # 分头
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # 这里用-1主要是qkv的seq_len可能不一样的，和q.size(1), k.size(1), v.size(1)等价
        q_s = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)  # v_s: [batch_size x n_heads x len_v x d_v]
        # 将掩码矩阵attn_mask从(batch_size, seq_len_q, seq_len_k)扩展到(batch_size, n_heads, seq_len_q, seq_len_k)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = scaled_doc_product_attntion(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.enbed_size) # context: [batch_size x len_q x n_heads * d_v]
        context = self.linear(context)
        return context, attn # output: [batch_size x len_q x d_model]

if __name__ == "__main__":
    batch_size = 2
    seq_len_q = 4
    seq_len_k = 6
    embed_size = 8
    n_heads = 2

    q = torch.rand(batch_size, seq_len_q, embed_size)
    k = torch.rand(batch_size, seq_len_k, embed_size)
    v = torch.rand(batch_size, seq_len_k, embed_size)
    attn_mask = torch.ones(batch_size, seq_len_q, seq_len_k)

    multi_head_attn = MultiHeadAttention(embed_size, n_heads)
    out, attn_weights = multi_head_attn(q, k, v, attn_mask)

    print("Output shape:", out.shape)  # Expected: (batch_size, seq_len_q, embed_size)
    print("Attention weights shape:", attn_weights.shape)  # Expected: (batch_size, n_heads, seq_len_q, seq_len_k)
