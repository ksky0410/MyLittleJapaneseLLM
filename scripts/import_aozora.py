"""青空文庫のShift_JISテキストを学習用のUTF-8本文へ変換する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from _common import repo_path

SEPARATOR_RE = re.compile(r"^\s*[-−―—]{5,}\s*$")
ANNOTATION_RE = re.compile(r"［＃.*?］")
FULL_RUBY_RE = re.compile(r"｜(?P<base>[^《\n]*)《[^》\n]*》")
ATTACHED_RUBY_RE = re.compile(
    r"(?P<base>[^\s、。！？「」『』（）()［］【】《》])《[^》\n]*》"
)
RUBY_RE = re.compile(r"《[^》\n]*》")

HEADER_MARKERS = ("【テキスト中に現れる記号について】", "【著者名】")
FOOTER_MARKERS = (
    "底本：",
    "底本:",
    "入力：",
    "入力:",
    "校正：",
    "校正:",
    "青空文庫作成ファイル：",
    "青空文庫作成ファイル:",
)
FOOTER_TAIL_WINDOW = 256
FOOTER_CLUSTER_GAP = 16


def sha256_bytes(data: bytes) -> str:
    """bytesのSHA-256を返す。"""

    return hashlib.sha256(data).hexdigest()


def _read_source(path: Path, encoding: str) -> tuple[bytes, str | bytes, str | None]:
    """txtまたはzip内の最初のtxtを読み、デコード前のbytesも返す。"""

    input_bytes = path.read_bytes()
    if not zipfile.is_zipfile(path):
        return input_bytes, input_bytes.decode(encoding), None

    with zipfile.ZipFile(path) as archive:
        members = sorted(
            info
            for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".txt")
        )
        if not members:
            raise ValueError(f"zip内にtxtファイルがありません: {path}")
        member = members[0]
        return input_bytes, archive.read(member), member.filename


def _locate_body(lines: list[str]) -> tuple[int, int, int, int]:
    """Aozoraのヘッダーとフッターを除く本文範囲を返す。"""

    separators = [
        index for index, line in enumerate(lines) if SEPARATOR_RE.fullmatch(line)
    ]
    has_header_marker = any(
        marker in line for line in lines for marker in HEADER_MARKERS
    )

    if len(separators) >= 2 or (separators and has_header_marker):
        # Aozoraの標準形式では、2本目の区切り線の直後から本文が始まる。
        start = separators[1] + 1 if len(separators) >= 2 else separators[0] + 1
    else:
        start = 0

    end = len(lines)
    for separator in separators:
        if separator < start:
            continue
        following = lines[separator + 1 : separator + 16]
        if any(_is_footer_marker_line(line) for line in following):
            end = separator
            break

    # 実作品には、本文直後に区切り線を置かず、底本情報から直接始まる
    # フッターもある。候補を本文末尾側へ絞り、後方のまとまりの先頭を
    # 終端にすることで、本文中の引用を不用意に切らないようにする。
    footer_candidates = [
        index
        for index, line in enumerate(lines[start:], start)
        if _is_footer_marker_line(line)
    ]
    tail_start = max(start, len(lines) - FOOTER_TAIL_WINDOW)
    tail_candidates = [index for index in footer_candidates if index >= tail_start]
    if tail_candidates:
        clusters: list[list[int]] = [[tail_candidates[0]]]
        for index in tail_candidates[1:]:
            if index - clusters[-1][-1] <= FOOTER_CLUSTER_GAP:
                clusters[-1].append(index)
            else:
                clusters.append([index])

        # 入力・校正・青空文庫作成ファイルは、底本よりフッターである
        # 可能性が高い。これらを含む後方クラスタを優先する。
        strong_clusters = [
            cluster
            for cluster in clusters
            if any(_footer_marker_name(lines[index]) != "底本" for index in cluster)
        ]
        footer_start = (strong_clusters or clusters)[-1][0]
        end = min(end, footer_start)

    return start, end, start, len(lines) - end


def _footer_marker_name(line: str) -> str | None:
    """行頭にあるフッターマーカーの種類を返す。"""

    stripped = line.strip()
    for marker in FOOTER_MARKERS:
        if stripped.startswith(marker):
            return marker.rstrip("：:")
    return None


def _is_footer_marker_line(line: str) -> bool:
    """行全体ではなく、行頭のAozoraフッターマーカーだけを検出する。"""

    return _footer_marker_name(line) is not None


def _convert_body_line(line: str) -> tuple[str, int, int, int]:
    """1行の注記とルビ表記を本文へ変換する。"""

    line = line.replace("\ufeff", "")
    line, annotation_count = ANNOTATION_RE.subn("", line)

    line, full_ruby_count = FULL_RUBY_RE.subn(lambda match: match.group("base"), line)
    line, attached_ruby_count = ATTACHED_RUBY_RE.subn(
        lambda match: match.group("base"), line
    )
    line, ruby_count = RUBY_RE.subn("", line)
    line, vertical_marker_count = re.subn("｜", "", line)

    return (
        line.strip(),
        annotation_count,
        full_ruby_count + attached_ruby_count + ruby_count,
        vertical_marker_count,
    )


def import_aozora(
    input_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    source: str | None = None,
    encoding: str = "shift_jis",
    max_chars: int = 4000,
) -> dict[str, object]:
    """Aozoraのtxt/zipを変換し、出力manifestを返す。"""

    input_file = Path(input_path)
    output_file = Path(output_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")
    if max_chars < 1:
        raise ValueError("max_charsは1以上で指定してください")

    input_bytes, decoded_or_bytes, member_name = _read_source(input_file, encoding)
    if isinstance(decoded_or_bytes, bytes):
        decoded_text = decoded_or_bytes.decode(encoding)
    else:
        decoded_text = decoded_or_bytes
    lines = decoded_text.splitlines()
    if not lines:
        raise ValueError("入力テキストが空です")

    start, end, header_lines, footer_lines = _locate_body(lines)
    output_lines: list[str] = []
    removed_lines = header_lines + footer_lines
    removed_blank_lines = 0
    removed_annotation_lines = 0
    annotation_count = sum(len(ANNOTATION_RE.findall(line)) for line in lines)
    ruby_count = 0
    vertical_marker_count = 0
    split_paragraphs = 0
    split_segments = 0

    for original_line in lines[start:end]:
        line, line_annotations, line_rubies, line_vertical_markers = _convert_body_line(
            original_line
        )
        ruby_count += line_rubies
        vertical_marker_count += line_vertical_markers
        if not line:
            removed_lines += 1
            if line_annotations:
                removed_annotation_lines += 1
            else:
                removed_blank_lines += 1
            continue

        chunks = [
            line[index : index + max_chars] for index in range(0, len(line), max_chars)
        ]
        if len(chunks) > 1:
            split_paragraphs += 1
            split_segments += len(chunks) - 1
        output_lines.extend(chunks)

    if not output_lines:
        raise ValueError("ヘッダー・フッターと注記を除いた本文が空です")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_text = "\n".join(output_lines) + "\n"
    output_bytes = output_text.encode("utf-8")
    output_file.write_bytes(output_bytes)

    if manifest_path is None:
        manifest_file = output_file.with_suffix(".manifest.json")
    else:
        manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "aozora-import-v1",
        "source": source or str(input_file.resolve()),
        "input_path": str(input_file.resolve()),
        "input_member": member_name,
        "encoding": encoding,
        "input_sha256": sha256_bytes(input_bytes),
        "output_sha256": sha256_bytes(output_bytes),
        "input_characters": len(decoded_text),
        "output_characters": len(output_text),
        "input_lines": len(lines),
        "output_lines": len(output_lines),
        "removed_lines": removed_lines,
        "removed_blank_lines": removed_blank_lines,
        "removed_annotation_lines": removed_annotation_lines,
        "annotation_count": annotation_count,
        "ruby_count": ruby_count,
        "vertical_marker_count": vertical_marker_count,
        "split_count": split_paragraphs,
        "split_paragraphs": split_paragraphs,
        "split_segments": split_segments,
        "max_paragraph_characters": max_chars,
    }
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Shift_JISのtxtまたはzip")
    parser.add_argument("--output", required=True, help="UTF-8本文の出力先")
    parser.add_argument(
        "--manifest", help="manifest JSONの出力先。省略時はoutputの拡張子を置換"
    )
    parser.add_argument("--source", help="出所URLなど。省略時は入力ファイルの絶対パス")
    parser.add_argument("--encoding", default="shift_jis", help="入力文字コード")
    parser.add_argument(
        "--max-chars", type=int, default=4000, help="1段落あたりの最大文字数"
    )
    args = parser.parse_args()

    manifest = import_aozora(
        repo_path(args.input),
        repo_path(args.output),
        repo_path(args.manifest) if args.manifest else None,
        source=args.source,
        encoding=args.encoding,
        max_chars=args.max_chars,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
