# VGGT-Omega 模型位置说明

## 是什么

VGGT-Ω（VGGT-Omega）是 Meta AI Research 与牛津大学 VGG 组联合发布的视觉几何基础模型（CVPR 2026），用于从图像恢复相机位姿、深度、点图等三维几何信息，是 GS 导航模组调试定位/建类算法的基础模型之一。

- 项目主页：https://vggt-omega.github.io/
- 代码仓库：https://github.com/facebookresearch/vggt-omega
- 许可证：**CC-BY-NC-4.0（仅可非商业使用）**，对外产品化前必须确认授权

## 在 gpu-server 上的位置

模型权重已下载到服务器**数据盘共享目录**，全组共用一份（约 9.7GB，不要重复下载）：

```
/data/shared/huggingface/VGGT-Omega-bucket/
├── README.md                      # 官方模型卡
├── vggt_omega_1b_256_text.pt      # 5.4 GB，1B 模型，256 分辨率，带 text 分支
└── vggt_omega_1b_512.pt           # 4.6 GB，1B 模型，512 分辨率
```

来源：Hugging Face 仓库 `holdonyb/VGGT-Omega-bucket`，2026-05-29 同步到服务器。

## 个人快捷路径

每个账号的个人目录下都有现成的软链接，指向上面这份共享权重（不占额外空间）：

```
/data/users/<用户名>/models/vggt-omega  ->  /data/shared/huggingface/VGGT-Omega-bucket
```

在宿主机上直接用这个路径即可：

```python
import torch
ckpt = torch.load("/data/users/<用户名>/models/vggt-omega/vggt_omega_1b_512.pt", map_location="cpu")
```

## 在 Docker 容器里用

把共享目录只读挂载进容器：

```bash
docker run --rm -it --gpus all \
  -v /data/shared/huggingface/VGGT-Omega-bucket:/models/vggt-omega:ro \
  -v /data/users/<用户名>:/workspace \
  <镜像名> bash
```

容器内权重路径即 `/models/vggt-omega/vggt_omega_1b_512.pt`。

## 注意事项

- 权重文件**只读使用**：不要修改、移动、删除共享目录里的任何文件；其他人也在用同一份。
- 需要不同格式/量化版本时，转换产物放自己的 `/data/users/<用户名>/` 目录。
- 该模型是非商业许可（CC-BY-NC-4.0），内部研发调试没问题，商用需另行确认。

---
最后更新：2026-09-12
