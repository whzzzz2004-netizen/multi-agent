# Multi-Agent

这是你当前正在维护的 GitHub 仓库：

- 仓库地址：`https://github.com/whzzzz2004-netizen/multi-agent`
- 当前项目入口包：`rdagent`
- 当前命令行入口：`rdagent`

这个仓库最初来自 `RD-Agent`，但这里的说明文档已经改成以你当前这份仓库的实际使用方式为准，而不是原开发者的主页、演示站和宣传信息。

## 当前环境要求

- Python `3.10` 或 `3.11`
- Linux 环境
- 建议安装 `conda` 或使用独立虚拟环境
- 部分场景依赖 Docker

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/whzzzz2004-netizen/multi-agent.git
cd multi-agent
```

### 2. 创建环境

使用 conda：

```bash
conda create -n multi-agent python=3.10 -y
conda activate multi-agent
```

或者使用 venv：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

基础开发安装：

```bash
make dev
```

如果你只想做最基础的本地安装，也可以：

```bash
pip install -e .
```

## 配置说明

CLI 会在启动时自动加载仓库根目录下的 `.env` 文件：

```python
load_dotenv(".env")
```

也就是说，你的运行配置应该优先放在项目根目录 `.env` 中。

常见做法：

```bash
cp .env.example .env
```

如果当前仓库里还没有 `.env.example`，就直接新建 `.env`。

## 常用命令

### 健康检查

```bash
rdagent health_check --no-check-env
```

### 启动日志 UI

```bash
rdagent ui
```

### 启动实时日志服务

```bash
rdagent server_ui
```

### Data Science 场景

```bash
rdagent data_science --help
```

### LLM Finetune 场景

```bash
rdagent llm_finetune --help
```

### Quant / Research 相关命令

```bash
rdagent fin_factor --help
rdagent fin_model --help
rdagent fin_quant --help
rdagent fin_research --help
```

## 当前 CLI 入口

目前这个仓库暴露的 CLI 命令来自 [rdagent/app/cli.py](/home/dministrator/RD-Agent/rdagent/app/cli.py)，主要包括：

- `rdagent fin_factor`
- `rdagent fin_model`
- `rdagent fin_quant`
- `rdagent fin_research`
- `rdagent fin_mine_factors`
- `rdagent fin_model_from_pool`
- `rdagent fin_lgbm_from_pool`
- `rdagent fin_backtest_model_library`
- `rdagent fin_import_models_from_report`
- `rdagent fin_list_model_library`
- `rdagent fin_factor_report`
- `rdagent general_model`
- `rdagent data_science`
- `rdagent llm_finetune`
- `rdagent grade_summary`
- `rdagent ui`
- `rdagent server_ui`
- `rdagent ds_user_interact`
- `rdagent health_check`
- `rdagent collect_info`

## 开发常用命令

这些命令来自 [Makefile](/home/dministrator/RD-Agent/Makefile)：

```bash
make dev
make lint
make test
make clean
make deepclean
```

说明：

- `make clean` 会清理常见缓存，例如 `__pycache__`、`.pytest_cache`、`.mypy_cache`
- `make deepclean` 会在 `clean` 的基础上进一步移除虚拟环境相关内容

## Git 协作建议

如果你要把这个仓库用于多人协作，建议保持下面的习惯：

- 不要把 `log/`、`logs/`、`__pycache__/`、临时脚本、测试压缩包提交进仓库
- 新功能尽量走独立分支，例如 `feature/xxx`
- 合并前通过 Pull Request 检查改动
- 改完文档后同步更新本 README，避免配置说明和实际代码不一致

## 当前仓库里特别要注意的内容

- 项目里已经存在一些本地运行产物和缓存目录，但大多数已被 `.gitignore` 忽略
- 当前命令入口仍然叫 `rdagent`，所以如果后续你想彻底改名，需要同时改包名、入口脚本和文档
- 当前 `pyproject.toml` 仍然决定了打包名、脚本入口和依赖来源

## 后续建议

如果你准备继续把这个仓库从原项目分叉成你自己的版本，下一步最值得做的是：

1. 增加一个可提交的 `.env.example`
2. 清理顶层临时文件，比如 `test.py`、`test.ipynb`、日志 zip
3. 把文档里剩余指向原仓库的链接逐步替换掉
4. 如果确定长期维护，再决定是否把包名 `rdagent` 改成你自己的名字
