# 完整系统性能与成本实验

本目录新增了 `scripts/run_full_system_experiment.py`，用于生成论文表 7 所需的完整 MAS 性能与成本结果。

## 运行方式

```powershell
python scripts\run_full_system_experiment.py
```

可选参数：

```powershell
python scripts\run_full_system_experiment.py --seeds 0 1 2 --max-tasks 50
```

## 输出文件

- `results/full_system_performance.csv`：表格结果，便于后续统计分析。
- `results/full_system_performance.json`：包含 metadata 的结构化结果。
- `results/full_system_performance_table.tex`：可直接粘贴进 LaTeX 表格的行。
- `完整系统性能与成本实验结果.docx`：Word 版实验结果，已放在项目目录。

## 当前实验后端

当前实现为 deterministic simulator，用于在没有真实 LLM API、HumanEval/MBPP/TeamBench/MultiAgentBench 数据加载器的情况下，先打通论文所需的完整实验与出表流程。它固定架构、数据集、任务数量和 seed，输出每任务均值及 95% CI。

后续接入真实 LLM runner 时，建议保持 CSV/JSON 字段不变：

- `architecture`
- `dataset`
- `task_score`
- `task_score_ci95`
- `token_cost`
- `token_cost_ci95`
- `latency_seconds`
- `latency_ci95`
- `tool_calls`
- `tool_calls_ci95`
- `tasks`
- `seeds`
- `backend`
- `note`

这样 `main.tex` 和 Word 报告可以复用现有结果文件直接更新。
