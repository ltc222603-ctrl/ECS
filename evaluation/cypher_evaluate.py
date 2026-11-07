"""统计 Cypher 的 EXPLAIN 验证通过率，不实际执行查询。

环境变量：NEO4J_URI、NEO4J_PASSWORD，
以及可选的 NEO4J_USERNAME、NEO4J_DATABASE（默认均为 neo4j）。
运行：python evaluation/cypher_evaluate.py --input <查询记录.jsonl>

外部 JSONL 每行包含 cypher（原始查询），可选 corrected_cypher
（已有的修复后查询）、parameters 和 corrected_parameters。
本脚本只验证已有查询，不生成或修复 Cypher，也不调用评价 LLM。
通过 EXPLAIN 不代表查询结果正确，或满足用户的业务意图。
"""

import argparse
import csv
import json
import os
from pathlib import Path

from neo4j import GraphDatabase, READ_ACCESS
from neo4j.exceptions import ClientError


def validate(session, cypher: str, parameters: dict) -> tuple[bool, str]:
    if not cypher.strip():
        return False, "empty_cypher"
    try:
        session.run("EXPLAIN\n" + cypher, parameters=parameters).consume()
        return True, ""
    except ClientError as error:
        # 查询本身的错误计为失败；认证、权限和服务异常继续抛出。
        if error.code.startswith(("Neo.ClientError.Statement.", "Neo.ClientError.Procedure.")):
            return False, error.code
        raise


def evaluate(input_path: Path, output_dir: Path) -> None:
    samples = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    results = []

    with GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    ) as driver:
        driver.verify_connectivity()
        with driver.session(
            database=os.getenv("NEO4J_DATABASE", "neo4j"), default_access_mode=READ_ACCESS,
        ) as session:
            for index, sample in enumerate(samples, start=1):
                parameters = sample.get("parameters", {})
                initial_pass, initial_error = validate(session, sample["cypher"], parameters)
                final_pass, final_error = initial_pass, initial_error
                corrected = sample.get("corrected_cypher")
                if corrected is not None:
                    final_pass, final_error = validate(
                        session, corrected, sample.get("corrected_parameters", parameters),
                    )
                repair_attempted = not initial_pass and corrected is not None
                results.append({
                    "sample_id": sample.get("id", index),
                    "initial_pass": initial_pass,
                    "final_pass": final_pass,
                    "repair_attempted": repair_attempted,
                    "repair_success": repair_attempted and final_pass,
                    "initial_error": initial_error,
                    "final_error": final_error,
                })
                print(f"Validated {index}/{len(samples)}")

    count = len(results)
    initial_passes = sum(row["initial_pass"] for row in results)
    final_passes = sum(row["final_pass"] for row in results)
    repair_attempts = sum(row["repair_attempted"] for row in results)
    repair_successes = sum(row["repair_success"] for row in results)
    summary = {
        "validation_method": "Neo4j EXPLAIN; no query execution or LLM judgment",
        "sample_count": count,
        "initial_pass_count": initial_passes,
        "initial_pass_rate": initial_passes / count,
        "final_pass_count": final_passes,
        "final_pass_rate": final_passes / count,
        "repair_attempt_count": repair_attempts,
        "repair_success_count": repair_successes,
        "repair_success_rate": repair_successes / repair_attempts if repair_attempts else None,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "validation.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="外部 JSONL 查询记录")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/cypher"))
    args = parser.parse_args()
    evaluate(args.input, args.output_dir)
