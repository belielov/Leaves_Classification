# Leaves-Classification
[数据集下载地址](https://www.kaggle.com/c/classify-leaves/data)
## Jupyter 内核
***
### 关联内核(kernel)与虚拟环境
1. 激活你的虚拟环境 `conda activate your_env_name`
2. 在虚拟环境中安装 ipykernel `pip install ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn`
3. 将虚拟环境注册到 Jupyter `python -m ipykernel install --user --name=your_env_name --display-name="显示名称"`
4. 验证安装 `jupyter kernelspec list`
***
### 在 Jupyter Notebook 中使用虚拟环境内核
1. （可在虚拟环境或基础环境中启动）启动 Jupyter Notebook： `jupyter notebook`
2. 创建或打开 Notebook 后：
   - 菜单栏选择 Kernel > Change Kernel
   - 选择你设置的显示名称（如 "显示名称"）
3. 验证当前内核：
```python
import sys
print(sys.version)  # 应显示 3.9.x
print(sys.executable)  # 应指向虚拟环境路径
```
***
### 管理多个内核
- 查看所有已注册内核： `jupyter kernelspec list` 
- 删除不需要的内核： `jupyter kernelspec uninstall your_env_name`
***
### 查看可用GPU
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA devices: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```
***
### 删除虚拟环境
```commandline
conda env list

conda deactivate

conda env remove --name Leaves_Classification

conda env list
```
## 代码解释
```python
test = pd.read_csv("../dataset/test.csv")  
testset = Leaf_Dataset(test, transform=transforms_test, test=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, drop_last=False)
```
这三行代码的关联如下：
1. `test = pd.read_csv("../dataset/test.csv")`
   - 从文件系统读取测试集 CSV 文件
   - 创建一个包含测试集图像路径的 DataFrame
   - 这个DataFrame被存储在变量 test 中
2. `testset = Leaf_Dataset(test, transform=transforms_test, test=True)`
   - 使用上一步创建的 test DataFrame 初始化自定义数据集
   - 应用测试集特定的预处理变换(transforms_test)
   - 设置 test=True 表示这是测试集（无标签）
3. `test_loader = torch.utils.data.DataLoader(...)`
   - 将测试集数据集包装为 DataLoader
   - batch_size=64：每次处理64个样本
   - shuffle=False：不随机打乱顺序（保持原始顺序）
   - drop_last=False：保留最后一批（即使不足64个样本）
***
### `pandas.factorize()`
pandas.factorize() 是 Pandas 中用于**分类变量编码**的核心方法，它将离散的类别值（如字符串标签）转换为整数编码。
#### 核心功能
```python
labels, uniques = pd.factorize(values)
```
- 输入：任意序列（列表、Series）
- 输出：
  - `labels`：整数编码数组（从0开始）
  - `uniques`：唯一值数组（原始类别）
#### 性能优势
- **向量化操作**：比 Python 循环快10-100倍
- **内存优化**：直接操作数组，避免中间列表
- **无缝集成**：原生支持 Pandas 数据结构
***
### 优化器、损失函数和调度器
优化器 (Optimizer)：更新模型参数以最小化损失函数。它根据损失函数的梯度调整权重。

损失函数 (Loss Function)：量化模型预测值与真实标签的差距，为优化器提供目标方向。

学习率调度器 (Scheduler)：动态调整学习率，以提升训练效果（如避免震荡、加速收敛）。