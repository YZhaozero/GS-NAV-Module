# gpu-server 账号说明（赵岩）

| 项目 | 内容 |
|------|------|
| 姓名 | 赵岩 |
| 用户名 | `zhaoyan` |
| 服务器 | `34.59.41.27`，SSH 端口 `22` |
| 登录方式 | 仅 SSH 密钥（服务器已关闭密码登录） |
| 私钥文件 | `zhaoyan_gpu_server_ed25519`（本压缩包内，请保密） |
| 个人目录 | `/data/users/zhaoyan/`（私有，已建好） |
| VGGT-Omega | `/data/users/zhaoyan/models/vggt-omega`（软链接，详见 02 文档） |
| 权限 | docker 组（可跑 GPU 容器）；无 sudo |

## 首次登录验证

```bash
ssh -i zhaoyan_gpu_server_ed25519 zhaoyan@34.59.41.27
# 登录后执行：
whoami                                  # 应输出 zhaoyan
ls /data/users/zhaoyan/models/vggt-omega/    # 应看到两个 .pt 权重文件
docker run --rm --gpus all tron-sim-nav:latest nvidia-smi -L   # 应看到 A100
```

Windows 下如果提示私钥权限过宽，按 `01-GPU服务器使用说明.md` 里的 icacls 步骤处理。

详细用法见另外两份文档：`01-GPU服务器使用说明.md`、`02-VGGT-Omega位置说明.md`。

账号开通日期：2026-09-12　管理员：holdo
