# GPU 服务器使用说明（GS导航模组团队）

## 服务器概况

| 项目 | 内容 |
|------|------|
| 地址 | `34.59.41.27`，SSH 端口 `22` |
| 系统 | Debian 12 (bookworm) |
| GPU | NVIDIA A100-SXM4-40GB × 1（全组共享） |
| 磁盘 | 系统盘 252G（/，约剩 116G）；数据盘 2T（/data，约剩 358G） |
| Python | 系统自带 python3.11 + pip3；算法环境建议用 Docker 容器 |

## 登录方式（仅支持 SSH 密钥，密码登录已关闭）

你的账号包里有一个私钥文件 `<用户名>_gpu_server_ed25519`，请妥善保管、不要外发。

**Windows PowerShell / Git Bash：**

```bash
ssh -i <私钥文件路径> <用户名>@34.59.41.27
```

如果 Windows 报错 `permissions are too open`，在 PowerShell 里执行（把路径换成你的）：

```powershell
icacls "C:\path\to\dingym_gpu_server_ed25519" /inheritance:r
icacls "C:\path\to\dingym_gpu_server_ed25519" /grant:r "$env:USERNAME:R"
```

**VS Code / Cursor 远程开发：** 安装 `Remote-SSH` 插件，在 `~/.ssh/config` 加：

```
Host gpu-server
    HostName 34.59.41.27
    User <用户名>
    IdentityFile <私钥文件路径>
    IdentitiesOnly yes
```

之后即可一键远程连接、编辑、调试服务器上的代码。

## 目录约定

| 路径 | 用途 | 权限 |
|------|------|------|
| `/data/users/<用户名>/` | 你的个人工作目录，代码、数据、模型输出都放这里 | 私有（别人不可读） |
| `/data/users/<用户名>/models/vggt-omega` | VGGT-Omega 模型权重快捷链接 | 只读 |
| `/data/shared/huggingface/` | 共享模型权重目录 | 只读，**不要往里写东西** |
| `/home/<用户名>/` | 系统盘 home | 空间小，只放配置文件 |

**大文件一律放 `/data/users/<用户名>/`，不要放 home（系统盘只有 116G 剩余）。** 数据盘全组共用，剩余约 358G，请及时清理不用的数据和镜像。

## Docker / GPU 使用

账号已加入 `docker` 组，可直接运行 docker 命令（无需 sudo）。

```bash
# 查看 GPU
docker run --rm --gpus all <镜像名> nvidia-smi

# 典型开发方式：挂个人目录进容器
docker run --rm -it --gpus all \
  -v /data/users/<用户名>:/workspace \
  <镜像名> bash
```

注意：

- 容器里要用 GPU 必须加 `--gpus all`。
- 服务器只有一张 A100，四人共享。长时间训练前在组里打个招呼，用完及时释放（`docker ps` / `docker stop`）。
- 不用的镜像及时 `docker rmi`，数据盘是共享资源。

## 权限边界

- 无 `sudo`（不能改系统配置、不能装系统软件）。需要装系统级软件请联系管理员。
- 可以在自己目录内自由安装 Python 包（`pip3 install --user ...`）或构建任意 Docker 镜像。
- 不要修改 `/data/shared/` 下的任何共享文件。
- 不要查看他人 `/data/users/` 下的私有目录（权限已隔离）。

## 管理员

服务器管理员：holdo。账号、权限、镜像、磁盘问题找他处理。

---
最后更新：2026-09-12
