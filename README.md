# MASContributionBench

MASContributionBench is a benchmark framework for studying agent-level contribution attribution in LLM-based multi-agent systems. It evaluates which agents matter, why they matter, and how contribution changes under task, topology, role, permission, and communication interventions.

## Core Idea

1. **Tasks**: HumanEval, MBPP, TeamBench, MultiAgentBench, MARBLE, and SWE-bench Lite.
2. **Architectures**: PEV, Chain, DAG, MetaGPT-lite, Star, Debate, and Graph coordination.
3. **Agents**: planner, coder, verifier, critic, supervisor, finalizer, and researcher.
4. **Attribution**: LOO, Shapley, Banzhaf, Myerson, and Owen-style contribution estimates.

## Directory Contract

- `data/raw/`: downloaded source datasets, kept as close to original as possible.
- `data/processed/`: unified JSONL task format and metadata.
- `data/runs/`: raw experiment records for each run.
- `data/traces/`: message-level and tool-level MAS traces.
- `configs/`: benchmark, agent, permission, prompt, and experiment definitions.
- `src/`: reusable benchmark implementation.
- `scripts/`: command-line entry points.
- `outputs/`: paper-facing results, tables, figures, and reports.
