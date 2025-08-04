# Transformer位置编码

## 1 正弦余弦位置编码（Sinusoidal Position Encoding）

这是 Transformer 原论文（Vaswani et al., 2017）中使用的方法，通过**固定的数学公式**生成位置嵌入，不参与模型训练。

### 1.1 原理

对于序列中位置为`pos`的 token，其第`i`维的位置嵌入计算如下：

- 偶数位置：$PE(pos,2i)​=sin(pos/10000^{2i/d_{model}}​​)$
- 奇数位置：$PE(pos,2i+1)​=cos(pos/10000^{2i/d_{model}​}​)$

其中，`d_model`是嵌入维度，`pos`是 token 在序列中的位置（从 0 开始），`i`是嵌入向量的维度索引。

#### 1.1.1 相对关系

三角函数（正弦、余弦）的三个关键性质，直接支撑了位置编码对相对位置的建模能力：

**加法定理（核心）** 正弦和余弦满足线性组合的加法定理： $\sin(a + b) = \sin a \cdot \cos b + \cos a \cdot \sin b$ 

$ \cos(a + b) = \cos a \cdot \cos b - \sin a \cdot \sin b$ 

这意味着：**“两个位置的和（或差）的三角函数值” 可以用 “两个位置各自的三角函数值” 通过线性组合表示**

#### 1.1.2 高低频特征

正弦余弦编码通过设计不同维度的周期$T_i = 10000^{2i/d_{\text{model}}}$，实现了对 “短距离” 和 “长距离” 相对位置的差异化捕捉：

低维度（`i` 小）：周期 \(T_i\) 小（如 $T_0 = 10000^{0} = 1$，周期接近 $2\pi$)，属于 “高频” 编码，对**短距离变化**（如 `k=1,2`）敏感；

高维度（`i` 大）：周期 \(T_i\) 大（如 $i = d_{\text{model}}/2$ 时，$T_i = 10000$），属于 “低频” 编码，对**长距离变化**（如 `k=100, 1000`）敏感

#### 1.1.3 无长度限制

正弦余弦编码是通过**公式动态生成**的（而非预定义的参数矩阵），因此：

- 无需预设最大序列长度（可生成任意长度 `pos` 的编码）；
- 对于训练时未见过的超长序列（如长于训练数据的文档），其位置编码仍能保持相对位置的数学规律，模型泛化性更好

可学习位置编码在面对变长序列时，通常需要额外处理（如截断、padding 或插值），但存在明显缺陷：

- 若序列更长：超出预训练最大长度的位置没有对应的可学习参数，需插值（如将预训练的编码按比例拉伸），但会破坏原始位置关系；
- 若序列更短：需截断或 padding，可能引入噪声

### 1.2 特点

- **固定不可训练**：位置嵌入由公式直接生成，不参与反向传播。
- **相对位置信息**：通过三角函数的性质（sin(a+b)和cos(a+b)可由sina,cosa,sinb,cosb表示），模型能间接学习到相对位置关系。
- **长序列泛化好**：理论上可生成任意长度的位置嵌入（即使长于训练时的序列长度）。

### 1.3 代码实现

```python
def sinusoidal_pos_encoding(seq_len, d_model, normalize=False):
    """
    生成正弦余弦位置编码

    参数:
        seq_len: 序列长度（位置数量）
        d_model: 嵌入维度
        normalize: 是否对位置编码进行归一化（可选）

    返回:
        pos_encoding: 形状为 [seq_len, d_model] 的位置编码矩阵
    """
    # 初始化位置编码矩阵
    pos_encoding = np.zeros((seq_len, d_model))

    # 生成位置索引（0到seq_len-1）
    positions = np.arange(seq_len)[:, np.newaxis]  # 形状: [seq_len, 1]

    # 计算每个维度的频率参数：10000^(2i/d_model)
    # 对于i in 0,1,...,d_model//2 - 1
    i = np.arange(d_model // 2)  # 形状: [d_model//2]
    div_term = np.power(10000, 2 * i / d_model)  # 形状: [d_model//2]

    # 偶数维度用sin，奇数维度用cos
    pos_encoding[:, 0::2] = np.sin(positions / div_term)  # 0,2,4...维度
    pos_encoding[:, 1::2] = np.cos(positions / div_term)  # 1,3,5...维度

    # 可选：归一化（部分场景中可稳定训练）
    if normalize:
        pos_encoding = pos_encoding / np.sqrt(d_model)

    return pos_encoding
```

## 2 可学习位置嵌入（Learned Position Embedding）

这种方法将位置嵌入视为**可训练的参数**，与词嵌入（Token Embedding）类似，通过模型训练优化。

### 2.1 原理

- 预先定义一个位置嵌入矩阵$\text{PE} \in \mathbb{R}^{L \times d_{\text{model}}}$，其中`L`是预设的最大序列长度，`d_model`是嵌入维度。
- 对于位置`pos`（$0 \leq pos < L$），直接取用矩阵中的第`pos`行作为其位置嵌入。
- 训练时，该矩阵会与模型其他参数一起通过反向传播更新。

### 2.2 特点

- **灵活性高**：模型可根据任务自适应学习位置信息，在很多任务（如 BERT 的预训练）中表现优于正弦余弦编码。
- **依赖预设长度**：最大序列长度`L`需预先定义，若输入序列长于`L`，需截断或额外处理（如插值）。
- **泛化性受限**：对超出训练长度的序列，泛化能力可能不如正弦余弦编码。

### 2.3 代码实现

```python
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, maximum_position_encoding, dropout):
        ...
        self.pos_encoding = nn.Parameter(torch.zeros(1, maximum_position_encoding, d_model))
        ...
    def forward(self, x, mask):
        ...
        seq_length = x.size(1)  
        x += self.pos_encoding[:, :seq_length, :]
        ...
```

## 3 旋转位置编码（Rotary Position Embedding, RoPE）

RoPE 是近年来在大语言模型（如 LLaMA、GPT-NeoX）中广泛使用的位置编码方法，核心是通过**旋转矩阵**将位置信息融入 Query 和 Key 的计算中。

### 3.1 原理

- 对于位置`pos`的 token，其 Query（Q）和 Key（K）向量会被一个**位置相关的旋转矩阵**进行旋转： $\hat{Q}_m = R_m(\theta) \cdot Q_m, \quad \hat{K}_n = R_n(\theta) \cdot K_n$ 其中$R_m(\theta)$是位置`m`的旋转矩阵，$\theta$是与维度相关的参数。
- 旋转后，Q 和 K 的内积$\hat{Q}_m \cdot \hat{K}_n$自然包含了位置差`m-n`的信息。

#### 3.1.1 旋转矩阵的设计

RoPE 的旋转矩阵针对**2 维向量**设计，高维向量通过 “两两分组” 映射到 2 维空间后应用旋转。

对于位置`pos`和维度对`(2i, 2i+1)`（将`d_model`维向量按顺序分成`d_model/2`个 2 维组），旋转矩阵定义为：

$R(pos, i) = \begin{bmatrix} \cos\theta_i(pos) & -\sin\theta_i(pos) \\sin\theta_i(pos) & \cos\theta_i(pos) \end{bmatrix}$

其中`θ_i(pos)`是位置相关的旋转角度，定义为：

$\theta_i(pos) = pos \cdot \theta_i, \quad \theta_i = \frac{1}{10000^{2i/d_{\text{model}}}}$

这个角度设计与正弦余弦位置编码类似，通过指数衰减的`θ_i`，让不同维度组对应不同周期的旋转（高频维度对近邻位置敏感，低频维度对远距位置敏感）

#### 3.1.2 高维向量的旋转

对于`d_model`维的向量`x`（如Q或K），RoPE的旋转过程为：

- 将`x`按维度分成`(x_0, x_1), (x_2, x_3), ..., (x_{d-2}, x_{d-1})`共`d_model/2`个2维子向量；
- 每个子向量`(x_{2i}, x_{2i+1})`用旋转矩阵`R(pos, i)`旋转： $\begin{bmatrix} \hat{x}_{2i} \\ \hat{x}_{2i+1} \end{bmatrix} = R(pos, i) \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}$
- 合并所有旋转后的子向量，得到完整的旋转后向量$\hat{x}$

### 3.2 特点

- **相对位置严格守恒**：直接建模相对位置（而非绝对位置），对长序列的泛化能力强。
- **无额外参数**：无需额外的位置嵌入矩阵，通过矩阵旋转融入位置信息。
- **与注意力机制深度结合**：位置信息仅影响 Q 和 K 的交互，不直接作为嵌入叠加，更符合注意力的计算逻辑。

### 3.3 代码实现

```python
def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    预计算旋转角度的复数表示（cosθ + i·sinθ）

    参数:
        dim: 模型维度（需为偶数）
        max_seq_len: 最大序列长度
        theta: 周期基数（同正弦余弦编码）

    返回:
        freqs_cis: 形状为 [max_seq_len, dim//2] 的复数张量，存储每个位置和维度的旋转角度
    """
    # 计算每个维度组的θ_i = 1 / theta^(2i/dim)
    freqs = 1.0 / (theta **(torch.arange(0, dim, 2)[: (dim // 2)] / dim))  # 形状: [dim//2]
    # 生成位置索引：0 ~ max_seq_len-1
    t = torch.arange(max_seq_len, device=freqs.device)  # 形状: [max_seq_len]
    # 计算每个位置和维度的角度：pos * θ_i（广播为 [max_seq_len, dim//2]）
    freqs = torch.outer(t, freqs)  # 形状: [max_seq_len, dim//2]
    # 转换为复数形式：cosθ + i·sinθ
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数张量，形状同上
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    对输入向量（Q或K）应用RoPE旋转

    参数:
        x: 输入向量，形状为 [batch_size, seq_len, n_heads, head_dim]
        freqs_cis: 预计算的旋转角度，形状为 [max_seq_len, head_dim//2]

    返回:
        x_rot: 旋转后的向量，形状与x相同
    """
    batch_size, seq_len, n_heads, head_dim = x.shape
    # 确保head_dim为偶数（RoPE要求维度成对）
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # 将x重塑为复数形式：[batch_size, seq_len, n_heads, head_dim//2]
    # 每个元素为 (x[2i] + x[2i+1]·i)
    x_complex = torch.view_as_complex(x.float().reshape(batch_size, seq_len, n_heads, -1, 2))
    # 提取当前序列长度对应的旋转角度
    freqs_cis = freqs_cis[:seq_len].unsqueeze(0).unsqueeze(2)  # 形状: [1, seq_len, 1, head_dim//2]
    # 复数乘法实现旋转：x_complex * freqs_cis（广播适用所有batch和head）
    x_rotated = x_complex * freqs_cis
    # 将复数转回实数张量：[batch_size, seq_len, n_heads, head_dim]
    x_out = torch.view_as_real(x_rotated).reshape(batch_size, seq_len, n_heads, head_dim)
    return x_out.type_as(x)  # 保持与输入相同的数据类型

# 在Attention模块forward中
class AttentionWithRoPE(nn.Module):
    def foward(self, x):
        ...
        # 对Q和K应用RoPE旋转（V不需要旋转）
        q_rot = apply_rotary_emb(q, self.freqs_cis)  # [batch_size, seq_len, n_heads, head_dim]
        k_rot = apply_rotary_emb(k, self.freqs_cis)  # 同上
        ...
```

## 4 ALiBi（Attention with Linear Biases）

ALiBi 不使用显式的 “位置嵌入”，而是通过在**注意力分数中加入偏置**来引入位置信息。

### 4.1 原理

- 对于 Query 位置`i`和 Key 位置`j`，在计算注意力分数时，额外添加一个与距离相关的偏置： $\text{score}(i,j) = \frac{Q_i K_j^T}{\sqrt{d_k}} + m \cdot |i-j|$ 其中`m`是一个预先定义的负参数（距离越远，偏置越负），`|i-j|`是两个位置的距离。
- 偏置会使模型更倾向于关注距离当前位置更近的 token。

### 4.2 特点

- **无额外嵌入参数**：无需学习或存储位置嵌入矩阵，简化模型结构。
- **动态适应序列长度**：天然支持任意长度的序列（无需预设最大长度）。
- **在长序列任务中表现优异**：尤其在需要捕捉长距离依赖的场景（如文档理解）中效果较好。

### 4.3 代码实现

```python
def generate_alibi_bias(seq_len: int, num_heads: int, m: float = -0.1) -> torch.Tensor:
    """
    生成ALiBi的偏置矩阵

    参数:
        seq_len: 序列长度
        num_heads: 注意力头数量
        m: 基础偏置系数（负数，控制偏置强度）

    返回:
        alibi_bias: 形状为 [num_heads, seq_len, seq_len] 的偏置矩阵
    """
    # 为每个注意力头生成不同的偏置系数：m * 1, m * 2, ..., m * num_heads
    # 头索引越大，偏置系数的绝对值越大（抑制作用越强）
    head_biases = torch.arange(1, num_heads + 1, device=torch.device('cpu')) * m  # 形状: [num_heads]

    # 生成相对距离矩阵：对于位置i和j，距离为|i - j|
    # 构造形状为 [seq_len, seq_len] 的矩阵，其中distance[i][j] = |i - j|
    range_vec = torch.arange(seq_len, device=torch.device('cpu'))
    distance = torch.abs(range_vec[:, None] - range_vec[None, :])  # [seq_len, seq_len]

    # 为每个头扩展偏置：[num_heads, 1, 1] * [1, seq_len, seq_len] → [num_heads, seq_len, seq_len]
    alibi_bias = head_biases[:, None, None] * distance[None, :, :]

    return alibi_bias


class AlibiAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, m: float = -0.1):
        ...
        self.m = m
        ...
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        ...
        batch_size, seq_len, _ = x.shape
        # 生成并添加ALiBi偏置
        alibi_bias = generate_alibi_bias(seq_len, self.num_heads, self.m).to(x.device)  # [heads, seq_len, seq_len]
        attn_scores = attn_scores + alibi_bias[None, :, :, :]  # 扩展batch维度后相加
        ...

"""
alibi_bias: 
tensor([[-0.0000, -0.1000, -0.2000,  ..., -9.7000, -9.8000, -9.9000],
        [-0.1000, -0.0000, -0.1000,  ..., -9.6000, -9.7000, -9.8000],
        [-0.2000, -0.1000, -0.0000,  ..., -9.5000, -9.6000, -9.7000],
        ...,
        [-9.7000, -9.6000, -9.5000,  ..., -0.0000, -0.1000, -0.2000],
        [-9.8000, -9.7000, -9.6000,  ..., -0.1000, -0.0000, -0.1000],
        [-9.9000, -9.8000, -9.7000,  ..., -0.2000, -0.1000, -0.0000]])
"""


```


