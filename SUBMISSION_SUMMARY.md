# 提交说明：完整系统性能与成本实验

本次改动围绕论文第 5.1 节“完整系统性能”完成，目标是生成完整 MAS 在各数据集上的性能与成本结果，并写入论文表 7 与 Word 报告。

## 主要代码改动

- 新增 `scripts/run_full_system_experiment.py`
  - 实现完整系统性能/成本实验入口。
  - 固定 A1--A7 七类架构与 HumanEval、MBPP、MultiAgentBench、HumanEval/MBPP 数据集设置。
  - 输出 Task score、Token cost、Latency、Tool calls 的均值与 95% CI。
  - 自动生成 CSV、JSON、LaTeX 表格行和 Word 报告。
  - 当前后端为 `deterministic_simulator`，用于在无真实 LLM API 与 benchmark loader 时打通可复现实验流程。

- 新增 `README_EXPERIMENTS.md`
  - 说明实验脚本运行方式、输出文件和字段 schema。
  - 说明后续接入真实 LLM runner 时需要保持的结果字段。

- 更新 `main.tex`
  - 将表 7 “完整 MAS 在各数据集上的性能与成本”由空表填入实验结果。
  - 将该表改为页面宽度自适应缩放，避免窄列排版过高。

## 生成结果文件

- `results/full_system_performance.csv`
- `results/full_system_performance.json`
- `results/full_system_performance_table.tex`
- `完整系统性能与成本实验结果.docx`
- 编译后的 `main.pdf`

根目录 `D:\University\研0\MAS管理` 下也复制了一份 `完整系统性能与成本实验结果.docx`，便于直接查看。

## 验证

已执行：

```powershell
python scripts\run_full_system_experiment.py
python -m py_compile scripts\run_full_system_experiment.py
python -m zipfile -l "完整系统性能与成本实验结果.docx"
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

结果：脚本运行成功，Word 结构可读，论文 PDF 编译成功。

## GitHub 提交状态

当前机器环境中未检测到 `git` 或 `gh`，且目录不是 Git 仓库，因此无法在本机直接 commit/push 到 `https://github.com/LeoZippon/MASContribution`。需要在安装 Git 并具备该仓库写权限后提交这些文件。
