# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述 / Project Overview

精品咖啡多因子归因系统，基于 [autoresearch](https://github.com/karpathy/autoresearch) 模式。模型公式：`Coffee Score = f(G, P, R, B, U)`，其中 G=产地/遗传, P=处理法, R=烘焙, B=冲煮, U=用户偏好。

Specialty coffee multi-factor attribution system built on the autoresearch pattern.

## 常用命令 / Common Commands

```bash
# 安装依赖 / Install dependencies
cd research && pip install -e .

# 数据准备与基线模型 / Data prep & baseline
python3 prepare.py --generate     # 生成合成数据 / Generate synthetic data
python3 train.py                  # 运行基线模型 / Run baseline model
python3 export_results.py         # 导出结果到 site/data/ / Export to site

# API 服务 / API server (需额外安装 fastapi uvicorn)
uvicorn api.server:app --reload --port 8000

# 本地预览网站 / Preview site
python3 -m http.server --directory site 8080

# Jupyter (Docker)
cd jupyter && docker-compose up
```

无测试框架、无 linter 配置。No test framework or linter configured.

## 架构与数据流 / Architecture & Data Flow

三个核心部分 / Three core parts:

- **research/** — Python 研究引擎。AI agent 迭代优化 `train.py` 以最小化 val_mae。数据集：966 条真实 CQI 阿拉比卡咖啡豆，52 特征，目标 `scores.overall` (63–91)。
- **api/** — FastAPI 后端。端点：`/api/predict`, `/api/recommend`, `/api/explore`, `/api/model-info`, `/api/beans/summary`, `/api/experiments`。加载 `prepare_v2.py`（33 特征，仅 G+P）和 `model.pkl`。
- **site/** — 静态网站（GitHub Pages）。原生 JS + Chart.js，无构建步骤。推送 `main` 分支的 `site/**` 路径自动部署。

```
train.py → results.tsv → export_results.py → site/data/*.json → git push → GitHub Pages
```

## 自主研究协议 / Autoresearch Protocol

运行自主研究时 / When running autonomous research:

- **唯一可修改文件 / Only modifiable file**: `research/train.py`
- **只读文件 / Read-only**: `prepare.py`, `prepare_v2.py`, `ingest.py`, 评估函数
- **指标 / Metric**: val_mae（越低越好 / lower is better），基线 1.90
- **时间预算 / Time budget**: 每次实验 5 分钟
- **可用包 / Available packages**: numpy, scipy, scikit-learn, pandas, matplotlib（不可安装新包）
- **数据分割 / Data split**: 80/20 train/val, seed=42
- 详细说明见 `research/program.md` / See `research/program.md` for full instructions

实验流程 / Experiment loop:
1. 修改 `train.py` → commit → 运行 → 提取 val_mae
2. 优于历史最佳 → KEPT；否则 → DISCARDED + `git reset --hard HEAD~1`
3. 结果追加到 `results.tsv`（格式：`commit\tval_mae\tnum_params\tstatus\tdescription`）

## 关键领域知识 / Key Domain Knowledge

- 海拔是最强预测因子（importance 0.27），其次纬度（0.18）、昼夜温差（0.09）
- R（烘焙）和 B（冲煮）因子在 CQI 数据中为常量，不携带信息——可考虑移除以减少噪声
- 评分分布紧密（std=2.9），微小的 MAE 改善即有意义
- `prepare_v2.py` 已实现仅 G+P 的 33 特征编码，供 `api/` 和 `train_v2.py` 使用

## train.py 配置参数 / Config Knobs

```python
MODEL_TYPE = "gbr"    # gbr, ridge, lasso, elastic, rf, svr, linear_sgd
GBR_PARAMS = {n_estimators, max_depth, learning_rate, subsample, min_samples_leaf, max_features}
USE_SCALER = True     # StandardScaler 预处理
POLY_DEGREE = 0       # 0=关闭, 2=交互特征
```

## 部署 / Deployment

GitHub Actions (`.github/workflows/pages.yml`) 在 `main` 分支 `site/**` 路径变更时自动部署到 GitHub Pages。
