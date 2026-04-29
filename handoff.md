# Motorail Loading Problem: VNS + BPC 联合求解器研究与重构日志 (Handoff Context)

## 1. 研究背景与核心问题
*   **研究问题**：**汽车列车装载问题 (Motorail Loading Problem)**。主要探讨如何将多种品牌、不同尺寸（长、高）的汽车（Vehicle）装载到双层汽车运输专列（Carriage）中，以最大化装载效率（在此转化为最小化未充分利用或负的总装载长度）。
*   **车辆需求分类**：汽车被划分为**强制装载需求 (Mandatory)** 和**可选装载需求 (Optional)**。
*   **物理难点与几何约束**：
    *   车厢分为**上层 (Upper Deck)** 和**下层 (Lower Deck)**。
    *   **可调甲板 (Adjustable Decks)**：车厢内部的甲板可以上下调节，使得上下层可以动态共享高度空间。我们将甲板状态抽象为 4 种组合模式：`h-h` (水平), `m-m` (中间/倾斜), `h-m`, `m-h`。
    *   **异形区域 (Irregular Geometry)**：由于车轮、车厢底盘和车顶轮廓，每层的空间不是简单的长方体，而是被划分为多个限高不同的连续区域（如上层的 D-E-D，下层的 A-B-C-B-A）。
    *   **连续放置 (Continuous Placement)**：与经典的离散装箱问题不同，汽车在车厢内是首尾连续排列的（带有安全间距 `spacing` = 400mm），这意味着一辆车可能会跨越两个限高不同的物理区域。

## 2. 算法演进与重大突破：分块方案与结构性解耦 (Structural Decoupling)
为了高效求解这个极度复杂的装箱问题，我们的算法经历了从传统的**整车厢生成**到**单层解耦生成**的重大架构演进。

### 痛点：原本的 "Merge" 组合爆炸
原本在列生成（Column Generation, CG）的 Pricing 阶段，我们会分别计算上层和下层的可行排列（Labeling），然后通过笛卡尔积（Cartesian Product）将上下层合并（Merge）成一整节车厢的方案（Column）。
由于每层可能有成百上千种排法，合并阶段会引发 $O(N^2)$ 的组合爆炸，导致每次迭代耗时极长。

### 突破：LayerMaster 架构解耦
*   **放弃全车厢列，改为“单层半车厢列”**。彻底删除了 Pricing 问题中灾难性的上下层 `Merge` 逻辑。
*   **主问题独立匹配**：主问题通过 `wagon_map` 约束 ($\sum \vartheta_{\text{up}}^p = \sum \vartheta_{\text{low}}^p$) 隐式地将上下层拼成合法的车厢。因为只要甲板模式 $p$ 相同，任何合法的上层都能和任何合法的下层无缝组合！
*   **定价问题对偶反馈**：Pricing 子问题对这四个甲板模式 $p$ 生成对应的对偶变量 $\gamma_p$，上层的 Reduced Cost 减去 $\gamma_p$，下层的加回 $\gamma_p$。

## 3. VNS 启发式暖启动 (C++ `VNS_cpp/`)
为了给 BPC 提供高质量的初始列，我们设计了变邻域搜索 (VNS)：
*   **连续/离散校验统一 (`Feasibility.h`)**：重写了连续物理空间滑动放置校验逻辑，并在内部强行注入了与 Gurobi 模型完全一致的**离散容量拦截**。彻底解决了 C++ 认为合法但 BPC 判定为非法的问题。
*   **震荡与重建 (`Perturb.h`)**：现在的扰动机制包含了甲板位置的随机翻转（`Reposition`），以及基于物理规则退避的贪心车辆抹除和 `BestInsert` 重新装载。
*   **算子优化**：废弃了低效的全局 Best Improvement 算子，引入并混排了 `RelocateRandom`, `SwapRandom`, `OptRandom` 等 **First Improvement (首次改进)** 算子。
*   **时间熔断机制**：全局超时 (300秒) 和 无提升超时 (15~120秒)。保证代码在大规模下不会假死。

## 4. BPC 精确求解器优化 (Python `src/model/BPC_LayerMaster/`)
*   **安全距离免检策略 (`labeling.py`)**：引入了极速剪枝机制。只要单层车辆数 $\le 2$ 或者“所有车辆总长 + 缓冲间距”塞得进最核心的无分隔大区（上层 14.9m，下层 12.4m），就**直接跳过 DP 和 Gurobi 检查**。该策略使 Feasibility Check 耗时**暴降 150 倍**。
*   **MIP Gap 与分支树追踪 (`BBtree.py`)**：
    *   加入全局 Lower Bound (LP 放宽后的理论下界) 追踪。
    *   当找到的全整数解 (Incumbent) 与全局理论下界的间隙（MIP Gap） $\le 0.01\%$ (`1e-4`) 时，自动宣告证明全局最优并退出搜索。
    *   由于列变成了单层列，全车厢变量 $a_i$ 失去物理定义，目前分支定界树仅对 $q_i$ （装载数量）进行分支。

## 5. 里程碑测试数据与结论
目前的联合求解器（Python 调度 C++ VNS 并接收 JSON 列文件作为 Warmstart，随后进入 LayerMaster BPC 求解）在极限性能上取得了震撼的结果：
*   **m7c7** 算例（使用 Warmstart）：在 0.3 秒内，0 节点收敛。
*   **m10c10** 算例（不使用 Warmstart）：LP 根节点搜索秒级完成，总耗时 <20秒。
*   **m11c11** 算例（使用 Warmstart）：VNS 闪电给解，BPC 仅耗时 172秒，在探索 21 个节点后证明全局绝对最优界（-496100.00）。