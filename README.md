
# FISTA图像修复算法与BM3D、ISTA对比研究

## 项目概述

本项目实现了**FISTA（Fast Iterative Shrinkage-Thresholding Algorithm）** 算法及其在图像修复任务中的应用，并与**ISTA（Iterative Shrinkage-Thresholding Algorithm）** 和**BM3D（Block-Matching and 3D Filtering）** 算法进行对比研究。

### 主要特性

- 实现FISTA-L1（小波域L1正则化）算法
- 实现FISTA-TV（全变差正则化）算法  
- 实现ISTA基准算法
- 集成BM3D图像去噪算法
- 完整的实验框架：数据加载、图像损坏、算法评估
- 量化指标计算：PSNR、SSIM、运行时间
- 可视化结果生成与收敛曲线分析
- LaTeX格式学术报告生成

## 研究目标

1. **理论验证**：验证FISTA相比ISTA的收敛速度优势（O(1/k²) vs O(1/k)）
2. **算法对比**：比较不同正则化方法（L1 vs TV）的效果
3. **性能评估**：在Set14数据集上进行系统评估
4. **应用指导**：为不同场景提供算法选择建议

## 实验结果摘要

在Set14数据集（特别是ppt3.png）上，60%像素随机缺失的实验结果：

| 算法         | PSNR (dB) | SSIM       | 时间 (秒) | 迭代次数 |
| ------------ | --------- | ---------- | --------- | -------- |
| 损坏图像     | 9.40      | --         | --        | --       |
| **FISTA-L1** | **20.85** | **0.5486** | 0.98      | 100      |
| FISTA-TV     | 12.26     | 0.1620     | **0.33**  | 100      |
| ISTA         | 20.82     | 0.5476     | 0.88      | 100      |
| BM3D         | 9.46      | 0.1218     | 6.82      | --       |

### 关键发现

1. **最佳质量**：FISTA-L1获得最高PSNR（20.85 dB）
2. **最快速度**：FISTA-TV计算最快（0.33秒）
3. **收敛优势**：FISTA比ISTA收敛更快
4. **适用场景**：BM3D不适合大范围缺失修复

## 环境安装

### 系统要求

- Python 3.7+
- 8GB+ RAM
- Windows/Linux/MacOS

### 安装步骤

1. **创建Conda环境**

```bash
conda create -n fista_denoising python=3.7
conda activate fista_denoising
```

2. **安装依赖包**

```bash
# 基础包
pip install numpy scipy matplotlib opencv-python

# 专业包
pip install PyWavelets scikit-image jupyter ipython tqdm h5py

# BM3D（Windows可能需要下载whl文件）
pip install bm3d
# 或从 https://www.lfd.uci.edu/~gohlke/pythonlibs/#bm3d 下载对应版本
```

3. **验证安装**

```bash
python -c "import numpy, scipy, cv2, pywt, skimage, bm3d; print('所有包安装成功！')"
```

## 项目结构

```
FISTA-Image-Inpainting/
├── data/                    # 数据集目录
│   └── Set14/              # Set14数据集
│       ├── baboon.png
│       ├── ppt3.png
│       └── ...
├── results/               # 实验结果
│   ├── ppt3_inpainting_results.png  # 主可视化结果
│   ├── convergence_curves.png       # 收敛曲线
│   ├── experiment_report.txt        # 实验报告
│   └── final_project_report.txt     # 最终报告
├── notebooks/             # Jupyter笔记本
│   └── FISTA_Demo.ipynb  # 主实验笔记本
├── paper/                 # 学术论文
│   ├── fista_paper.tex   # LaTeX论文
│   └── FISTA_Project.pdf   # 生成的PDF
├── README.md             # 本文件
```

## 快速开始

### 运行Jupyter Notebook（推荐）

```bash
# 启动Jupyter
jupyter notebook notebooks/FISTA_Demo.ipynb

# 在浏览器中打开，按顺序运行所有单元格
```

## 算法详解

### 1. FISTA-L1（小波域L1正则化）

```python
# 核心迭代
y_k = x_k + ((t_{k-1} - 1)/t_k) * (x_k - x_{k-1})
x_{k+1} = prox_{λg}(y_k - ∇f(y_k))
t_{k+1} = (1 + sqrt(1 + 4*t_k²))/2

# 小波变换 + 软阈值
coeffs = wavelet_transform(x)
coeffs = soft_threshold(coeffs, λ)
x = inverse_wavelet_transform(coeffs)
```

### 2. FISTA-TV（全变差正则化）

```python
# TV正则化项
g(x) = λ * ||∇x||_1

# 使用Chambolle对偶方法
dx, dy = compute_gradient(x)
div = compute_divergence(p_x, p_y)
x = x + τ * div
```

### 3. 参数调优建议

| 参数         | FISTA-L1  | FISTA-TV   | ISTA      |
| ------------ | --------- | ---------- | --------- |
| 正则化系数λ  | 0.01-0.1  | 0.001-0.05 | 0.01-0.1  |
| 最大迭代次数 | 50-200    | 50-200     | 100-300   |
| 小波基       | db4, haar | -          | db4, haar |
| 步长         | 自适应    | τ=0.125    | 0.1       |

## 实验设置

### 数据集

- **Set14数据集**：14张标准测试图像
- **重点图像**：ppt3.png (480×500)
- **损坏模式**：随机像素缺失（60%）

### 评估指标

1. **PSNR（峰值信噪比）**：客观质量指标

   ```
   PSNR = 10 * log10(MAX² / MSE)
   ```

2. **SSIM（结构相似性）**：感知质量指标

   - 亮度对比
   - 结构对比

3. **运行时间**：算法效率

4. **收敛曲线**：损失函数随迭代变化

## 性能分析

### 1. 算法对比

| 维度         | FISTA-L1 | FISTA-TV | ISTA  | BM3D  |
| ------------ | -------- | -------- | ----- | ----- |
| **修复质量** | ★★★★★    | ★★★☆☆    | ★★★★☆ | ★★☆☆☆ |
| **计算速度** | ★★★☆☆    | ★★★★★    | ★★★☆☆ | ★☆☆☆☆ |
| **纹理保持** | ★★★★★    | ★★☆☆☆    | ★★★★★ | ★★★☆☆ |
| **边缘保持** | ★★★☆☆    | ★★★★★    | ★★★☆☆ | ★★★★☆ |
| **收敛速度** | ★★★★★    | ★★★★★    | ★★★☆☆ | -     |

### 2. 应用场景推荐

- **实时应用**：推荐FISTA-TV（速度最快）
- **高质量修复**：推荐FISTA-L1（PSNR最高）
- **纹理丰富图像**：使用FISTA-L1或ISTA
- **边缘明显图像**：使用FISTA-TV
- **小噪声去除**：使用BM3D
- **大范围缺失修复**：使用迭代优化算法

### 3. 收敛行为分析

```python
# 收敛速度对比
FISTA: O(1/k²)  # 快速收敛
ISTA: O(1/k)    # 较慢收敛

# 实验观察
- FISTA: 50次迭代接近最优
- ISTA: 70次迭代达到相同质量
- 前10次迭代损失减少: FISTA > ISTA
```

### 论文结构

1. **引言**：图像修复背景与问题定义
2. **方法**：算法理论与实现细节
3. **实验**：数据集、评估指标、结果
4. **分析**：算法对比、时间分析、收敛分析
5. **结论**：主要发现与应用建议

