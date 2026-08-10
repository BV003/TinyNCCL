# 开发同步 — Mac 代码 覆盖 到 Vast 云机

> 场景：本地在 **Mac** 上开发、改代码，Vast 云机只负责**拉取/运行**（不在云机上 git 提交）。
> 策略：以 **Mac 为代码源头**，改完后用 **scp 整目录覆盖** 到云机。
> 前置：已生成并配置好 `~/.ssh/vastai_id` 私钥（公钥已加入 Vast.ai 账号）。

---

## 0. 关键信息（按你的实例填）

| 项 | 值 |
|---|---|
| 云机用户 | `root` |
| 云机 IP | `167.179.138.57` |
| SSH 端口 | `40022`（以 Vast.ai 页面给出的直连命令端口为准） |
| 私钥 | `~/.ssh/vastai_id` |
| 本地项目路径 | `/Users/michael/Documents/codehome/TinyNCCL` |
| 云机目标路径 | `/root/tiny-nccl` |

---

## 1. 完整覆盖流程（推荐）

每次从 Mac 同步到云机，执行两步：

### 第 1 步：先清掉云机旧目录（避免残留旧文件 / 旧结构冲突）
```bash
ssh -p 40022 -i ~/.ssh/vastai_id root@167.179.138.57 'rm -rf /root/tiny-nccl'
```

### 第 2 步：整目录 scp 覆盖（连同 .git、src、scripts、docs 等）
```bash
scp -P 40022 -i ~/.ssh/vastai_id -r \
    /Users/michael/Documents/codehome/TinyNCCL \
    root@167.179.138.57:/root/tiny-nccl
```

> scp 会把整个项目目录复制到 `/root/tiny-nccl`。
> 云机上自动生成的编译产物（`*.so`、`*.o`、`__pycache__`）下次传入前会被第 1 步清掉，正常。

### 第 3 步：云机上编译 + 运行验证
```bash
# 在云机（交互终端或 ssh 命令）
cd /root/tiny-nccl
python scripts/build.py        # 编译 CUDA 扩展（只在 .cu 变更时需要）
python scripts/topology.py     # 验证环境（可选）
```

---

## 2. 只传「改动过的目录」的轻量版（小改动场景）

不改 `.cu`、只改 Python 时，不必整包覆盖，只传需要更新的目录即可。

```bash
# 只传 src 和 scripts（覆盖同名文件）
scp -P 40022 -i ~/.ssh/vastai_id -r \
    /Users/michael/Documents/codehome/TinyNCCL/src \
    /Users/michael/Documents/codehome/TinyNCCL/scripts \
    root@167.179.138.57:/root/tiny-nccl/
```

> ⚠️ 轻量版**不会删除**云机上「本地已删掉的文件」。若本地删除了某些文件要同步，请回落到第 1 步的完整覆盖。

---

## 3. 什么时候要覆盖 / 重编译

| 我改了 | 要不要覆盖云机 | 要不要重跑 build |
|---|---|---|
| `src/*.cu`（CUDA 内核） | 要 | **要**（`python scripts/build.py`） |
| `src/ring_allreduce.py` 等 Python 调度 | 要 | 不需要（Python 解释执行） |
| `scripts/*.py`（build / topology） | 要 | 可能要（若 build 逻辑变了） |
| `tests/*.py` | 要 | 不需要 |

---

## 4. 注意事项

- **不要在云机上留「唯一一份」的代码**：云机是执行环境，改代码以 Mac 为准；云机上的改动请先同步回 Mac（`scp` 反向）。
- **端口变化**：每次重开实例，Vast.ai 可能给新端口，覆盖命令里的 `-p` 要改成对应的新端口。
- **ssh 命令失败时**：确认 `-p` 端口正确、`-i ~/.ssh/vastai_id` 已指定、且该实例已带上账号里的公钥（重开后的实例才有；老实例需重开）。
- **生成命令的小技巧**：可以先 `scp` 前用 `ssh ... 'df -h /root'` 确认空间够。

---

## 5. 快速脚本（可选）

把下面存成 `sync.sh` 放在项目根目录，以后一条命令完成「清 + 传」：

```bash
#!/usr/bin/env bash
# sync.sh — 将 Mac 项目整目录同步到 Vast 云机
set -e
PORT="${PORT:-40022}"
IP="${IP:-167.179.138.57}"
LOCAL="${LOCAL:-/Users/michael/Documents/codehome/TinyNCCL}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/tiny-nccl}"
KEY="${KEY:-$HOME/.ssh/vastai_id}"

echo "[1/3] 清空云机旧目录 ..."
ssh -p "$PORT" -i "$KEY" "root@$IP" "rm -rf '$REMOTE_ROOT'"

echo "[2/3] scp 整目录覆盖 ..."
scp -P "$PORT" -i "$KEY" -r "$LOCAL" "root@$IP:$REMOTE_ROOT"

echo "[3/3] 同步完成。云机上运行:"
echo "    cd $REMOTE_ROOT && python scripts/build.py"
```

使用：
```bash
chmod +x sync.sh
./sync.sh                 # 用默认值
PORT=40022 ./sync.sh      # 指定端口
```
