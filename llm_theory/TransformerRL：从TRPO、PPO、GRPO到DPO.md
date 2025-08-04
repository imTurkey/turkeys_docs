# TransformerRL：从TRPO、PPO、GRPO到DPO

## 1 TRPO：信任域策略优化

### 1.1 核心原理

TRPO（Trust Region Policy Optimization）于 2015 年由 OpenAI 提出，其核心是通过**KL 散度约束策略更新范围**，确保策略迭代过程中性能单调提升。具体而言，TRPO 在优化目标函数时引入信任域（Trust Region），要求新旧策略的 KL 散度不超过预设阈值： 

$\max_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \sum_{t=0}^T \gamma^t A_{\theta_{\text{old}}}(s_t, a_t) \right] \quad \text{s.t.} \quad D_{\text{KL}}(\pi_{\theta_{\text{old}}} || \pi_{\theta}) \leq \delta$ 

其中，$A_{\theta_{\text{old}}}$为优势函数，$\delta$为信任域半径。通过这种约束，TRPO 避免了传统策略梯度算法因步长过大导致的性能崩溃。

### 1.2 技术特点

- **自然梯度优化**：采用共轭梯度法求解二阶优化问题，提升收敛速度。
- **离线策略更新**：通过重要性采样复用历史数据，减少环境交互次数。
- **理论保证**：严格的数学证明确保策略性能单调递增。

### 1.3 局限性

- **计算复杂度高**：需计算 Hessian 矩阵与向量的乘积，对大规模模型不友好。
- **实现难度大**：涉及复杂的线性搜索和二阶优化步骤，工程落地门槛高。
- **早期 RLHF 应用**：在 InstructGPT 早期版本中曾被尝试，但因资源消耗问题被 PPO 取代。

## 2 PPO：近端策略优化

### 2.1 核心改进

PPO（Proximal Policy Optimization）于 2017 年提出，通过**剪辑目标函数**替代显式信任域约束，大幅简化实现流程：

 $\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$ 

其中，$r_t(\theta)$为新旧策略的动作概率比，$\epsilon$为剪辑系数（通常取 0.2）。PPO 通过限制概率比的变化范围，间接控制策略更新幅度。

### 2.2 技术优势

- **单阶段优化**：无需计算自然梯度，直接通过随机梯度下降（SGD）更新参数。
- **样本效率提升**：引入广义优势估计（GAE）减少方差，支持更少的环境交互次数。
- **灵活变体**：包括 PPO-Clip（剪辑目标）和 PPO-KL（KL 散度惩罚）两种实现，适配不同场景。

### 2.3 工业级应用

PPO 成为 RLHF 的主流选择，典型案例包括：

- **ChatGPT**：通过 PPO 优化模型生成符合人类偏好的对话内容。
- **Anthropic 的 Claude**：利用 PPO 实现 “无害性” 对齐，降低模型生成有害内容的概率。
- **数学推理**：在 GSM8K 等基准测试中，PPO 优化的模型推理准确率较监督微调提升 15% 以上。

## 3 GRPO：群组相对优势优化

### 3.1 核心思想

GRPO（Group Relative Policy Optimization）由 DeepSeek 团队于 2024 年提出，通过**群组相对优势估计**替代传统价值网络，核心公式为：

 $A_i = \frac{R_{\phi}(r_i) - \text{mean}(\mathcal{G})}{\text{std}(\mathcal{G})}$ 

其中，$\mathcal{G}$为同一问题生成的多个回答集合，$R_{\phi}$为奖励模型评分。GRPO 通过群组内标准化奖励，直接计算每个回答的相对优势，无需独立训练价值网络。

### 3.2 关键创新

- **无价值网络设计**：避免 PPO 中价值函数与策略网络的同步优化难题，显存占用减少 50%。
- **群组采样机制**：每个问题生成多个回答（如 DeepSeek-R1 中每问题生成 64 个样本），利用多样性探索提升训练稳定性。
- **分层优势估计**：在全局和局部层面分别计算优势，缓解稀疏奖励问题。

### 3.3 应用案例

- **数学推理**：DeepSeekMath 通过 GRPO 优化，在 MATH 基准测试中准确率超越 70B 开源模型，达到闭源模型水平。
- **代码生成**：在 Rust 代码生成任务中，GRPO 将编译通过率从 61% 提升至 80%，单元测试通过率从 22% 提升至 37%，训练成本低于 100 美元。
- **多模态任务**：南洋理工联合字节提出的 Share-GRPO，在 MathVista 数据集上实现 75.4% 的准确率，较传统 GRPO 提升 3.1%。

#### 4. 工程实践

- **开源支持**：Hugging Face 的 TRL 框架已集成 GRPO，支持与 LoRA 等轻量化技术结合，在消费级 GPU（如 16GB 显存）上完成训练。
- **动态采样策略**：根据问题难度自动调整采样数量，在计算效率与模型性能间取得平衡。

## 4 DPO：直接偏好优化

### 4.1 核心思想

DPO（Direct Preference Optimization）2023 年提出，直接利用人类偏好数据优化策略，无需显式训练奖励模型。通过最大化偏好响应的对数概率差，结合动态重要性权重规避模型退化： 

$\mathcal{L}_{\text{DPO}}(\theta) = \mathbb{E} \left[ \log \frac{\pi*{\theta}(y_w \mid x)}{\pi_{\theta}(y_l \mid x)} - \beta \left( \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\text{SFT}}(y_w \mid x)} - \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\text{SFT}}(y_l \mid x)} \right) \right]$ 

其中，$\pi_{\text{SFT}}$为监督微调模型，$\beta$控制策略偏离程度。

- **技术优势**：
  
  - **计算高效**：省去奖励模型训练阶段，显存占用减少 30% 以上。
  - **稳定性强**：类似监督学习的优化过程，避免 RL 训练中的波动性。
  - **多语言适配**：在 XGLM-56B 等多语言模型中，DPO 使跨语言响应一致性提升 18%。

- **应用场景**：
  
  - **内容安全**：Anthropic 的 Claude 通过 DPO 优化 “无害性” 对齐，将有害内容生成率降低至 0.3% 以下。
  - **代码生成**：在 HumanEval 数据集上，DPO 优化的 CodeGen 模型通过率较 PPO 提升 9%。

## 5 对比

| **维度**   | **TRPO**   | **PPO**                   | **GRPO**    | DPO              |
| -------- | ---------- | ------------------------- | ----------- | ---------------- |
| **核心约束** | KL 散度显式约束  | 概率比剪辑 / KL 惩罚             | 群组相对优势标准化   | 直接偏好优化           |
| **内存占用** | 高（二阶优化）    | 中（需价值网络）                  | 低（无价值网络）    | 低（无 RM）          |
| **训练效率** | 低（离线策略）    | 中（在线策略）                   | 高（群组采样复用）   | 高                |
| **典型场景** | 早期 RLHF 研究 | 通用对话 / 安全对齐               | 数学推理 / 代码生成 | 内容安全、代码生成        |
| **开源支持** | 较少（需自行实现）  | 成熟（TRL、Stable Baselines3） | 新兴（TRL 已支持） | Hugging Face TRL |
