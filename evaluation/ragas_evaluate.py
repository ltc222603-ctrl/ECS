import argparse
import asyncio
import csv
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)


async def evaluate(input_path: Path, output_dir: Path, model: str) -> None:
    samples = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    async with AsyncOpenAI(
        api_key=os.environ["RAGAS_EVAL_API_KEY"],
        base_url=os.getenv("RAGAS_EVAL_BASE_URL"),
    ) as client:
        judge = llm_factory(model, client=client)
        metrics = {
            "context_precision": ContextPrecision(llm=judge),
            "context_recall": ContextRecall(llm=judge),
            "faithfulness": Faithfulness(llm=judge),
            "factual_correctness": FactualCorrectness(llm=judge, mode="f1"),
        }
        results = []

        for index, sample in enumerate(samples, start=1):
            retrieval = {
                "user_input": sample["user_input"],
                "retrieved_contexts": sample["retrieved_contexts"],
                "reference": sample["reference"],
            }
            scores = {
                "context_precision": await metrics["context_precision"].ascore(**retrieval),
                "context_recall": await metrics["context_recall"].ascore(**retrieval),
                "faithfulness": await metrics["faithfulness"].ascore(
                    user_input=sample["user_input"],
                    retrieved_contexts=sample["retrieved_contexts"],
                    response=sample["response"],
                ),
                "factual_correctness": await metrics["factual_correctness"].ascore(
                    response=sample["response"], reference=sample["reference"],
                ),
            }
            results.append({
                "sample_id": sample.get("id", index),
                **{name: float(score.value) for name, score in scores.items()},
            })
            print(f"Evaluated {index}/{len(samples)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scores.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_id", *metrics])
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "model": model,
        "sample_count": len(results),
        "mean_scores": {
            name: sum(row[name] for row in results) / len(results)
            for name in metrics
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="外部 JSONL 测试记录")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--model", default=os.getenv("RAGAS_EVAL_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()
    asyncio.run(evaluate(args.input, args.output_dir, args.model))
