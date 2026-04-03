# 遮挡数据集下载指南

## 概述
本指南提供三个遮挡数据集的下载方法：3DPW（基础）、3DOH50K、3DPW-Occ

## 1. 3DPW 基础数据集

### 状态检查
```bash
ls -lh /home/yangz/4D-Humans/data/3DPW/
```

当前已有：
- ✅ imageFiles (部分)
- ✅ sequenceFiles (train, test, validation)
- ✅ paddingMaskFiles
- ✅ joints.mat

### 自动下载（如需补全）
```bash
cd /home/yangz/NViT-master/nvit
chmod +x download_3dpw.sh
./download_3dpw.sh
```

### 手动下载链接
- **官方网站**: https://virtualhumans.mpi-inf.mpg.de/3DPW/
- **直接下载**:
  - imageFiles.zip: https://virtualhumans.mpi-inf.mpg.de/3DPW/imageFiles.zip
  - sequenceFiles.zip: https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip
  - readme_and_demo.zip: https://virtualhumans.mpi-inf.mpg.de/3DPW/readme_and_demo.zip

## 2. 3DOH50K 数据集

### 官方来源
- **项目主页**: https://www.yangangwang.com/
- **论文**: Zhang et al., "Object-occluded human shape and pose estimation from a single color image", CVPR 2020

### 下载步骤
1. 访问 https://www.yangangwang.com/
2. 查找 "Object-Occluded Human" 或 "3DOH50K" 项目
3. 下载数据集（可能需要填写申请表或同意许可协议）

### 数据集信息
- **图像数量**: 51,600 张
- **标注内容**:
  - 2D 和 3D 姿态
  - SMPL 参数
  - 二值分割掩码
- **场景**: 真实世界遮挡场景

### 许可要求
- 仅限科研用途
- 如在论文中使用，需引用原论文
- 商业用途需获得书面许可

## 3. 3DPW-Occ（基于 3DOH50K）

### 说明
3DPW-Occ 是 Zhang et al. 论文中使用的评估子集，通常包含在 3DOH50K 数据集中。

### 获取方式
下载 3DOH50K 后，查找：
- 标注文件中的 3DPW-Occ 子集
- 或按照论文中的筛选标准从 3DPW 中提取遮挡样本

## 4. 3DPW-AdvOcc（对抗性遮挡）

### 生成方法
使用 3DNBF 代码库生成：

```bash
# 克隆 3DNBF
git clone https://github.com/edz-o/3DNBF
cd 3DNBF

# 按照 README 配置环境和数据
# 使用 OccludedHumanImageDataset 生成对抗性遮挡
```

### 配置示例
```python
test=dict(
    type='OccludedHumanImageDataset',
    orig_cfg=original_dataset,
    occ_size=80,
    occ_stride=40,
)
```

## 当前状态

### 已完成 ✅
- 3DPW 基础数据集（部分可用）
- 本地 3DPW-OCC 子集（22 样本，62.9% 遮挡率）

### 待下载 📥
- [ ] 3DPW 完整 imageFiles（如需）
- [ ] 3DOH50K 数据集
- [ ] 3DPW-Occ 官方版本（如果与 3DOH50K 分开）

## 快速开始

### 仅需评估基准
```bash
# 1. 确保 3DPW 基础数据集完整
./download_3dpw.sh

# 2. 访问 https://www.yangangwang.com/ 下载 3DOH50K
# （需要手动操作）

# 3. 配置数据路径
# 编辑配置文件指向下载的数据集
```

### 生成 3DPW-AdvOcc
```bash
# 1. 完成上述步骤
# 2. 克隆并配置 3DNBF
git clone https://github.com/edz-o/3DNBF
cd 3DNBF
# 按照 README 操作
```

## 存储需求

- 3DPW: ~60GB (imageFiles) + ~500MB (sequenceFiles)
- 3DOH50K: ~20GB（估计）
- 总计: ~80GB

## 故障排除

### 下载速度慢
```bash
# 使用代理或镜像
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
```

### 解压失败
```bash
# 检查磁盘空间
df -h /home/yangz/4D-Humans/data/

# 手动解压
unzip -o imageFiles.zip -d /home/yangz/4D-Humans/data/3DPW/
```

## 联系方式

- 3DPW: https://virtualhumans.mpi-inf.mpg.de/contact.html
- 3DOH50K: Yangang Wang (https://www.yangangwang.com/)
