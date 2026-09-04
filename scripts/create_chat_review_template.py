"""固定会話評価の生成結果から、人手レビュー用JSONを作成する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from _common import repo_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_review_template(
    evaluation_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    evaluation_file = repo_path(evaluation_path).resolve()
    output_file = repo_path(output_path).resolve()
    if not evaluation_file.is_file():
        raise FileNotFoundError(f"評価JSONが見つかりません: {evaluation_file}")
    evaluation = json.loads(evaluation_file.read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict) or not isinstance(
        evaluation.get("results"), list
    ):
        raise TypeError("評価JSONはresults配列を含むobjectで指定してください")

    reviews: list[dict[str, Any]] = []
    for index, result in enumerate(evaluation["results"], start=1):
        if not isinstance(result, dict):
            raise TypeError(f"評価結果#{index}がobjectではありません")
        required = {
            "conversation_id",
            "target_index",
            "reference",
            "completion",
        }
        missing = sorted(required - result.keys())
        if missing:
            raise ValueError(f"評価結果#{index}に必須項目がありません: {missing}")
        reviews.append(
            {
                "review_id": f"chat-test-v1-{index:03d}",
                "conversation_id": result["conversation_id"],
                "record_index": result.get("record_index"),
                "target_index": result["target_index"],
                "target_speaker": result.get("target_speaker"),
                "source": result.get("source"),
                "stratum": result.get("stratum"),
                "history_truncated": result.get("history_truncated"),
                "train_text_overlap": result.get("train_text_overlap"),
                "prompt": result.get("rendered_prompt"),
                "reference": result["reference"],
                "completion": result["completion"],
                "generated_token_count": result.get("generated_token_count"),
                "eos_reached": result.get("eos_reached"),
                "token_overlap_f1": result.get("token_overlap_f1"),
                "context_fit": None,
                "role_fit": None,
                "not_collapsed": None,
                "review_notes": "",
                "reviewer": "",
            }
        )

    template: dict[str, Any] = {
        "format": "chat-human-review-template-v1",
        "evaluation": str(evaluation_file),
        "evaluation_sha256": _sha256_file(evaluation_file),
        "instructions": {
            "context_fit": "履歴の話題と直前発話に自然につながるならtrue、明らかに外れるならfalse",
            "role_fit": "求められた応答役割（相づち、質問への回答など）を果たすならtrue、果たさないならfalse",
            "not_collapsed": "文字列崩壊・無関係な反復・不自然なメンション列が主でなければtrue、明らかに崩れていればfalse",
            "review_notes": "判断根拠、曖昧さ、context超過やtrain重複の影響を自由記述",
        },
        "review_status": "pending_human_review",
        "reviewed_count": 0,
        "reviews": reviews,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    template = create_review_template(args.evaluation, args.output)
    print(f"reviews: {len(template['reviews'])}")
    print(f"output: {repo_path(args.output).resolve()}")


if __name__ == "__main__":
    main()
