# 中文 LaTeX 论文稿：MAS Agent Contribution Attribution

本目录包含修改后的中文论文工程，题目为：

> 什么使一个智能体重要？面向大语言模型多智能体系统的结构、角色与权限归因

## 文件说明

- `main.tex`：中文论文主体，已从“实验方案/提纲”改写为正式论文叙述，包含 Abstract、Introduction、问题定义、评估指标、影响因素、研究假设、实验设置、假设检验与结果表、Related Work 和总结。
- `related_work_section_zh.tex`：扩展后的中文相关工作章节，由 `main.tex` 引入。
- `references.bib`：已合并新增相关工作条目的 BibLaTeX 参考文献库。
- `references.tex`：当前环境使用的内联参考文献文件，避免依赖外部 `biber`。
- `Makefile`：Tectonic 编译命令。
- `configs/base_experiment.yaml`：实验配置模板。

## 编译方式

当前环境推荐使用 Tectonic：

```bash
make
```

或手动执行：

```bash
/home/lzp/.conda/envs/latex/bin/tectonic --synctex --keep-logs --keep-intermediates main.tex
```

## 结果表

第 5 节保留了结果表结构。尚未完成实验的结果单元格为空，可后续在完成实证结果后填入。
