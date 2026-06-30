# Experiment Plan

## Stage 1: Data Construction
Build unified task files for HumanEval and MBPP first, then add TeamBench, MultiAgentBench, MARBLE, and SWE-bench Lite as external validation.

## Stage 2: Full-System Performance
Run A1-A7 architectures and record task score, cost, latency, and trace-level features.

## Stage 3: Contribution Attribution
Compute LOO for all tasks and sampled Shapley/Banzhaf for representative subsets.

## Stage 4: Controlled Interventions
Change topology, role placement, and permissions while holding task and model fixed.

## Stage 5: Analysis
Compare contribution rankings, architecture sensitivity, and task-difficulty interactions.
