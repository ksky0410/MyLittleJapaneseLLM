"""Medilinkの医師国家試験SQLiteを学習用JSONL/TXTへ変換する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

from _common import repo_path

DEFAULT_VALIDATION_VERSIONS = (119,)
DEFAULT_TEST_VERSIONS = (120,)
DEFAULT_CHALLENGE_VERSIONS = (700,)
BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "div",
    "dl",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
IGNORED_TAGS = {"script", "style"}
WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
NEWLINE_RE = re.compile(r"\n[ \t\f\v]*\n+")
LINE_BREAK_RE = re.compile(r"[\r\n\u0085\u2028\u2029]")


class _HTMLToText(HTMLParser):
    """HTMLを本文へ変換し、img要素の個数を数える。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.image_count = 0
        self._ignored_depth = 0

    def _append(self, value: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in IGNORED_TAGS:
            self._ignored_depth += 1
        elif tag == "img":
            self.image_count += 1
            self._append("[図表あり]")
        elif tag == "br" or tag in BLOCK_TAGS:
            self._append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() in IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        elif tag in BLOCK_TAGS:
            self._append("\n")

    def handle_data(self, data: str) -> None:
        self._append(data)


def html_to_text(value: object) -> tuple[str, int]:
    """HTML文字列を本文へ変換し、画像プレースホルダ数を返す。"""

    if value is None:
        return "", 0
    parser = _HTMLToText()
    parser.feed(str(value))
    parser.close()
    text = "".join(parser.parts).replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    text = NEWLINE_RE.sub("\n", text)
    return text.strip(), parser.image_count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sqlite_uri(path: Path) -> str:
    return f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"


@contextmanager
def read_only_connection(path: str | Path) -> Iterator[sqlite3.Connection]:
    """SQLiteをmode=roとquery_onlyで開く。"""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SQLiteデータベースが見つかりません: {source}")
    connection = sqlite3.connect(_sqlite_uri(source), uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        yield connection
    finally:
        connection.close()


def _parse_json_object(value: object) -> dict[str, object]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_list(value: object) -> list[object]:
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _string_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _join_html_values(values: Iterable[object]) -> tuple[str, int]:
    texts: list[str] = []
    image_count = 0
    for value in values:
        text, images = html_to_text(value)
        if text:
            texts.append(text)
        image_count += images
    return "\n".join(texts), image_count


def _normalise_versions(values: Iterable[int], name: str) -> tuple[int, ...]:
    versions = tuple(values)
    if any(version < 0 for version in versions):
        raise ValueError(f"{name}は0以上の整数で指定してください")
    if len(set(versions)) != len(versions):
        raise ValueError(f"{name}に重複した値があります")
    return versions


def _split_for_version(
    exam_version: int,
    validation_versions: tuple[int, ...],
    test_versions: tuple[int, ...],
    challenge_versions: tuple[int, ...],
) -> str:
    if exam_version in validation_versions:
        return "validation"
    if exam_version in test_versions:
        return "test"
    if exam_version in challenge_versions:
        return "challenge"
    return "train"


def _one_line(value: object) -> str:
    text = _string_value(value)
    text = LINE_BREAK_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _format_text_record(record: dict[str, object]) -> str:
    options = record["options"]
    explanations = record["explanations"]
    option_text = "；".join(
        f"{key}：{_one_line(value)}" for key, value in options.items()
    )
    explanation_text = "；".join(
        f"{key}：{_one_line(value)}" for key, value in explanations.items()
    )
    correct = "、".join(str(value) for value in record["correct"])
    fields = (
        f"問題番号：{record['number']}",
        f"試験回：{record['exam_version']}",
        f"問題：{_one_line(record['question'])}",
        f"選択肢：{option_text}",
        f"正解：{correct}",
        f"ポイント：{_one_line(record['point'])}",
        f"選択肢解説：{explanation_text}",
    )
    return "　".join(fields)


def import_medical_qb(
    input_path: str | Path,
    output_dir: str | Path = "artifacts/corpus/medical-qb-v1",
    *,
    validation_versions: Iterable[int] = DEFAULT_VALIDATION_VERSIONS,
    test_versions: Iterable[int] = DEFAULT_TEST_VERSIONS,
    challenge_versions: Iterable[int] = DEFAULT_CHALLENGE_VERSIONS,
) -> dict[str, object]:
    """questions/descriptionsを分割して出力し、manifestを返す。"""

    input_file = Path(input_path)
    output_root = Path(output_dir)
    validation = _normalise_versions(validation_versions, "validation_versions")
    test = _normalise_versions(test_versions, "test_versions")
    challenge = _normalise_versions(challenge_versions, "challenge_versions")
    split_version_sets = {
        "validation_versions": set(validation),
        "test_versions": set(test),
        "challenge_versions": set(challenge),
    }
    for first_name, first_versions in split_version_sets.items():
        for second_name, second_versions in split_version_sets.items():
            if first_name >= second_name:
                continue
            if first_versions & second_versions:
                raise ValueError(f"{first_name}と{second_name}は重複できません")

    split_records: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
        "challenge": [],
    }
    descriptions: dict[str, tuple[dict[str, object] | None, bool]] = {}

    with read_only_connection(input_file) as connection:
        description_rows = connection.execute(
            "SELECT number, json FROM descriptions"
        ).fetchall()
        for number, raw_json in description_rows:
            try:
                parsed = json.loads(raw_json)
            except (TypeError, json.JSONDecodeError):
                descriptions[str(number)] = (None, True)
                continue
            descriptions[str(number)] = (
                parsed if isinstance(parsed, dict) else None,
                not isinstance(parsed, dict),
            )

        question_rows = connection.execute(
            """
            SELECT number, exam_version, pre_body, body, options_json,
                   correct_answers_json, has_images
            FROM questions
            ORDER BY exam_version, number
            """
        ).fetchall()

    questions_count = len(question_rows)
    adopted_count = 0
    skipped_count = 0
    missing_description_count = 0
    malformed_description_count = 0
    missing_explanation_count = 0
    image_question_count = 0
    image_occurrence_count = 0
    exam_version_counts: dict[str, int] = {}

    for row in question_rows:
        number, exam_version, pre_body, body, raw_options, raw_correct, has_images = row
        if not number or exam_version is None:
            skipped_count += 1
            continue

        question, question_images = _join_html_values((pre_body, body))
        if not question:
            skipped_count += 1
            continue
        options_raw = _parse_json_object(raw_options)
        options: dict[str, str] = {}
        option_images = 0
        for key in sorted(options_raw):
            text, images = html_to_text(options_raw[key])
            options[str(key)] = text
            option_images += images
        correct = [str(value) for value in _parse_json_list(raw_correct)]

        description_entry = descriptions.get(str(number))
        description: dict[str, object] | None
        if description_entry is None:
            description = None
            missing_description_count += 1
        else:
            description, malformed = description_entry
            if malformed:
                missing_description_count += 1
                malformed_description_count += 1
        if description is None:
            point = ""
            explanations_raw: dict[str, object] = {}
        else:
            point = html_to_text(description.get("point", ""))[0]
            explanations_value = description.get("explanations", {})
            if not isinstance(explanations_value, dict):
                explanations_raw = {}
                missing_explanation_count += 1
            else:
                explanations_raw = explanations_value
                if not explanations_raw:
                    missing_explanation_count += 1
        explanations: dict[str, str] = {}
        explanation_images = 0
        for key in sorted(explanations_raw):
            text, images = html_to_text(explanations_raw[key])
            explanations[str(key)] = text
            explanation_images += images

        image_occurrences = question_images + option_images + explanation_images
        if has_images and image_occurrences == 0:
            question = f"{question}\n[図表あり]"
            image_occurrences = 1
        has_any_images = bool(image_occurrences)
        if has_any_images:
            image_question_count += 1
            image_occurrence_count += image_occurrences

        record: dict[str, object] = {
            "number": str(number),
            "exam_version": int(exam_version),
            "question": question,
            "options": options,
            "correct": correct,
            "point": point,
            "explanations": explanations,
            "has_images": has_any_images,
        }
        split = _split_for_version(int(exam_version), validation, test, challenge)
        split_records[split].append(record)
        adopted_count += 1
        version_key = str(int(exam_version))
        exam_version_counts[version_key] = exam_version_counts.get(version_key, 0) + 1

    output_root.mkdir(parents=True, exist_ok=True)
    split_manifest: dict[str, dict[str, object]] = {}
    for split, records in split_records.items():
        jsonl_text = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        )
        text_text = "".join(_format_text_record(record) + "\n" for record in records)
        jsonl_path = output_root / f"{split}.jsonl"
        text_path = output_root / f"{split}.txt"
        jsonl_path.write_text(jsonl_text, encoding="utf-8")
        text_path.write_text(text_text, encoding="utf-8")
        split_manifest[split] = {
            "count": len(records),
            "characters": len(text_text),
            "jsonl_characters": len(jsonl_text),
            "jsonl_sha256": hashlib.sha256(jsonl_text.encode("utf-8")).hexdigest(),
            "txt_sha256": hashlib.sha256(text_text.encode("utf-8")).hexdigest(),
            "output_sha256": hashlib.sha256(text_text.encode("utf-8")).hexdigest(),
            "jsonl_path": str(jsonl_path.resolve()),
            "txt_path": str(text_path.resolve()),
        }

    manifest: dict[str, object] = {
        "format": "medical-qb-v1",
        "source": str(input_file.resolve()),
        "input_sha256": _sha256_file(input_file),
        "read_only": True,
        "questions_count": questions_count,
        "adopted_count": adopted_count,
        "skipped_count": skipped_count,
        "missing_count": missing_description_count,
        "missing_description_count": missing_description_count,
        "malformed_description_count": malformed_description_count,
        "missing_explanation_count": missing_explanation_count,
        "image_count": image_question_count,
        "image_question_count": image_question_count,
        "image_occurrence_count": image_occurrence_count,
        "exam_version_counts": exam_version_counts,
        "validation_versions": list(validation),
        "test_versions": list(test),
        "challenge_versions": list(challenge),
        "splits": split_manifest,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("0以上の整数を指定してください")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="読み取り専用で開くqb.sqlite")
    parser.add_argument(
        "--output-dir",
        default="artifacts/corpus/medical-qb-v1",
        help="small_llm側の出力先",
    )
    parser.add_argument(
        "--validation-version",
        type=_nonnegative_int,
        action="append",
        help="validationへ分けるexam_version。複数指定可。既定119",
    )
    parser.add_argument(
        "--test-version",
        type=_nonnegative_int,
        action="append",
        help="testへ分けるexam_version。複数指定可。既定120",
    )
    parser.add_argument(
        "--challenge-version",
        type=_nonnegative_int,
        action="append",
        help="challengeへ分けるexam_version。複数指定可。既定700",
    )
    args = parser.parse_args()
    manifest = import_medical_qb(
        repo_path(args.input),
        repo_path(args.output_dir),
        validation_versions=(
            args.validation_version
            if args.validation_version is not None
            else DEFAULT_VALIDATION_VERSIONS
        ),
        test_versions=(
            args.test_version
            if args.test_version is not None
            else DEFAULT_TEST_VERSIONS
        ),
        challenge_versions=(
            args.challenge_version
            if args.challenge_version is not None
            else DEFAULT_CHALLENGE_VERSIONS
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
