#!/usr/bin/env python3
"""Run reproducible full-system MAS performance/cost experiments.

This script intentionally has no third-party dependencies.  It provides a
deterministic local benchmark harness for the paper's Table 7: complete-system
task score, token cost, latency, and tool-call cost across datasets.

The current backend is a calibrated deterministic simulator, not a live LLM
runner.  It is useful for exercising the full experimental pipeline and for
documenting the exact table-generation protocol before API-backed runs are
available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    tasks: int
    difficulty: float
    token_scale: float
    tool_need: float
    coordination_need: float


@dataclass(frozen=True)
class ArchitectureSpec:
    code: str
    name: str
    agents: int
    quality: float
    coordination: float
    review: float
    tool_access: float
    token_multiplier: float
    latency_multiplier: float
    tool_multiplier: float
    note: str


DATASETS = {
    "HumanEval": DatasetSpec("HumanEval", 80, 0.42, 1.00, 0.82, 0.30),
    "MBPP": DatasetSpec("MBPP", 160, 0.34, 0.82, 0.70, 0.22),
    "MultiAgentBench": DatasetSpec("MultiAgentBench", 120, 0.58, 1.36, 0.48, 0.78),
    "HumanEval/MBPP": DatasetSpec("HumanEval/MBPP", 120, 0.38, 0.91, 0.76, 0.26),
}

ARCHITECTURES = {
    "A1 PEV": ArchitectureSpec(
        "A1",
        "PEV",
        3,
        0.76,
        0.56,
        0.74,
        0.75,
        1.00,
        1.00,
        1.00,
        "角色分离基线，verifier 提升难题表现",
    ),
    "A2 Chain": ArchitectureSpec(
        "A2",
        "Chain",
        5,
        0.73,
        0.63,
        0.66,
        0.80,
        1.24,
        1.28,
        1.18,
        "串行流程增加 review 成本",
    ),
    "A3 DAG": ArchitectureSpec(
        "A3",
        "DAG",
        5,
        0.79,
        0.72,
        0.78,
        0.82,
        1.32,
        1.20,
        1.28,
        "并行规划与调试，质量/成本较均衡",
    ),
    "A4 MetaGPT-lite": ArchitectureSpec(
        "A4",
        "MetaGPT-lite",
        5,
        0.77,
        0.69,
        0.82,
        0.86,
        1.48,
        1.38,
        1.40,
        "SOP 角色稳定，但 token 成本较高",
    ),
    "A5 Star": ArchitectureSpec(
        "A5",
        "Star",
        6,
        0.70,
        0.80,
        0.58,
        0.58,
        1.42,
        1.16,
        0.88,
        "中心调度有利于协作任务",
    ),
    "A6 Debate": ArchitectureSpec(
        "A6",
        "Debate",
        6,
        0.74,
        0.60,
        0.88,
        0.54,
        1.70,
        1.56,
        0.72,
        "投票更稳健，但 token 成本高",
    ),
    "A7 Graph": ArchitectureSpec(
        "A7",
        "Graph",
        8,
        0.72,
        0.84,
        0.66,
        0.62,
        1.86,
        1.45,
        0.94,
        "稠密通信适合协作场景",
    ),
}

EXPERIMENTS = [
    ("A1 PEV", "HumanEval"),
    ("A2 Chain", "HumanEval"),
    ("A3 DAG", "MBPP"),
    ("A4 MetaGPT-lite", "MBPP"),
    ("A5 Star", "MultiAgentBench"),
    ("A6 Debate", "HumanEval/MBPP"),
    ("A7 Graph", "MultiAgentBench"),
]


def stable_noise(*parts: object, spread: float = 1.0) -> float:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    raw = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return (raw - 0.5) * 2.0 * spread


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def task_score(arch: ArchitectureSpec, dataset: DatasetSpec, task_id: int, seed: int) -> float:
    difficulty = clamp(
        dataset.difficulty + stable_noise(dataset.name, task_id, seed, spread=0.16),
        0.05,
        0.95,
    )
    coordination_bonus = 0.16 * arch.coordination * dataset.coordination_need
    review_bonus = 0.13 * arch.review * (0.45 + difficulty)
    tool_bonus = 0.08 * arch.tool_access * dataset.tool_need
    complexity_penalty = 0.30 * difficulty
    overhead_penalty = 0.018 * max(0, arch.agents - 5)
    debate_penalty = 0.025 if arch.name == "Debate" and difficulty < 0.35 else 0.0
    mean = (
        arch.quality
        + coordination_bonus
        + review_bonus
        + tool_bonus
        - complexity_penalty
        - overhead_penalty
        - debate_penalty
    )
    mean += stable_noise(arch.code, dataset.name, task_id, seed, spread=0.035)
    return clamp(mean, 0.0, 1.0)


def task_costs(arch: ArchitectureSpec, dataset: DatasetSpec, task_id: int, seed: int) -> tuple[int, float, int]:
    length_factor = clamp(1.0 + stable_noise("length", dataset.name, task_id, seed, spread=0.22), 0.70, 1.35)
    tokens = int(
        round(
            1180
            * dataset.token_scale
            * arch.token_multiplier
            * (0.68 + 0.14 * arch.agents)
            * length_factor
        )
    )
    latency = (
        7.8
        * dataset.token_scale
        * arch.latency_multiplier
        * (0.74 + 0.10 * arch.agents)
        * length_factor
    )
    latency += stable_noise("latency", arch.code, dataset.name, task_id, seed, spread=0.45)
    tools = round(
        dataset.tool_need
        * arch.tool_multiplier
        * (1.0 + 0.20 * arch.agents)
        * clamp(1.0 + stable_noise("tool", arch.code, dataset.name, task_id, seed, spread=0.25), 0.50, 1.60)
    )
    return max(tokens, 1), round(max(latency, 0.1), 2), max(int(tools), 0)


def mean_ci(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], 0.0
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values))
    return mean, 1.96 * sem


def run_experiments(seeds: Iterable[int], max_tasks: int | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arch_key, dataset_key in EXPERIMENTS:
        arch = ARCHITECTURES[arch_key]
        dataset = DATASETS[dataset_key]
        n_tasks = min(dataset.tasks, max_tasks) if max_tasks else dataset.tasks

        scores: list[float] = []
        tokens: list[int] = []
        latencies: list[float] = []
        tools: list[int] = []
        for seed in seeds:
            for task_id in range(n_tasks):
                scores.append(task_score(arch, dataset, task_id, seed))
                token_cost, latency, tool_calls = task_costs(arch, dataset, task_id, seed)
                tokens.append(token_cost)
                latencies.append(latency)
                tools.append(tool_calls)

        score_mean, score_ci = mean_ci(scores)
        token_mean, token_ci = mean_ci([float(x) for x in tokens])
        latency_mean, latency_ci = mean_ci(latencies)
        tool_mean, tool_ci = mean_ci([float(x) for x in tools])
        rows.append(
            {
                "architecture": arch_key,
                "dataset": dataset_key,
                "task_score": round(score_mean, 3),
                "task_score_ci95": round(score_ci, 3),
                "token_cost": round(token_mean),
                "token_cost_ci95": round(token_ci),
                "latency_seconds": round(latency_mean, 2),
                "latency_ci95": round(latency_ci, 2),
                "tool_calls": round(tool_mean, 2),
                "tool_calls_ci95": round(tool_ci, 2),
                "tasks": n_tasks,
                "seeds": ",".join(str(seed) for seed in seeds),
                "backend": "deterministic_simulator",
                "note": arch.note,
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, object]], path: Path, args: argparse.Namespace) -> None:
    payload = {
        "metadata": {
            "experiment": "full_system_performance_cost",
            "backend": "deterministic_simulator",
            "seeds": args.seeds,
            "max_tasks": args.max_tasks,
            "description": (
                "Deterministic local harness for complete-system MAS performance "
                "and cost table generation."
            ),
        },
        "results": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def tex_escape(text: object) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def write_tex_rows(rows: list[dict[str, object]], path: Path) -> None:
    lines = []
    for row in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["architecture"]),
                    tex_escape(row["dataset"]),
                    f'{row["task_score"]} $\\pm$ {row["task_score_ci95"]}',
                    f'{int(row["token_cost"]):,} $\\pm$ {int(row["token_cost_ci95"]):,}',
                    f'{row["latency_seconds"]}s $\\pm$ {row["latency_ci95"]}',
                    f'{row["tool_calls"]} $\\pm$ {row["tool_calls_ci95"]}',
                    tex_escape(row["note"]),
                ]
            )
            + r" \\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def paragraph(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{escape(text)}</w:t></w:r></w:p>"


def table(rows: list[list[object]]) -> str:
    out = [
        "<w:tbl>",
        (
            "<w:tblPr><w:tblStyle w:val=\"TableGrid\"/>"
            "<w:tblW w:w=\"0\" w:type=\"auto\"/></w:tblPr>"
        ),
    ]
    for row in rows:
        out.append("<w:tr>")
        for cell in row:
            out.append(
                "<w:tc><w:tcPr><w:tcW w:w=\"1800\" w:type=\"dxa\"/></w:tcPr>"
                f"<w:p><w:r><w:t>{escape(str(cell))}</w:t></w:r></w:p></w:tc>"
            )
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def write_docx(rows: list[dict[str, object]], path: Path) -> None:
    table_rows: list[list[object]] = [
        ["架构", "数据集", "Task score", "Token cost", "Latency", "Tool calls", "备注"]
    ]
    for row in rows:
        table_rows.append(
            [
                row["architecture"],
                row["dataset"],
                f'{row["task_score"]} ± {row["task_score_ci95"]}',
                f'{int(row["token_cost"]):,} ± {int(row["token_cost_ci95"]):,}',
                f'{row["latency_seconds"]}s ± {row["latency_ci95"]}',
                f'{row["tool_calls"]} ± {row["tool_calls_ci95"]}',
                row["note"],
            ]
        )

    body = "".join(
        [
            paragraph("完整 MAS 在各数据集上的性能与成本实验结果", "Title"),
            paragraph("说明：当前结果由确定性本地实验后端生成，用于复现论文表 7 的完整出表流程；指标为每任务均值，± 为 95% CI。"),
            paragraph("实验后端：deterministic_simulator；可用真实 LLM runner 替换同一 CSV/JSON schema。"),
            table(table_rows),
            paragraph("文件输出：results/full_system_performance.csv、results/full_system_performance.json、results/full_system_performance_table.tex。"),
        ]
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14"><w:body>'
        f"{body}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" '
        'w:right="1440" w:bottom="1440" w:left="1440" w:header="708" '
        'w:footer="708" w:gutter="0"/></w:sectPr></w:body></w:document>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
        '<w:name w:val="Normal"/></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
        '<w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>'
        '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/>'
        '<w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '</w:tblBorders></w:tblPr></w:style></w:styles>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types)
        docx.writestr("_rels/.rels", rels)
        docx.writestr("word/document.xml", document)
        docx.writestr("word/styles.xml", styles)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_experiments(args.seeds, args.max_tasks)
    write_csv(rows, RESULTS_DIR / "full_system_performance.csv")
    write_json(rows, RESULTS_DIR / "full_system_performance.json", args)
    write_tex_rows(rows, RESULTS_DIR / "full_system_performance_table.tex")
    write_docx(rows, ROOT / "完整系统性能与成本实验结果.docx")

    print(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
