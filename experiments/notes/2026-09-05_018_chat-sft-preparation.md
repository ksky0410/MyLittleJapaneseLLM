# 実験018：会話SFT用データの整形

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。Issue #1の会話コーパスを通常のnext-token pretrainingと切り分け、応答側loss maskingを使うSFTデータへ変換します。今回の仮説は、会話全体のTokenを一様に学習するより、履歴と話者境界を文脈として与え、現在の返答本文とEOSだけにlossをかける方が、短い返答を学習する目的に適しているというものです。

入力はsmall_llm側へ取り込んだ`artifacts/corpus/conversation-v1/train.jsonl`と`validation.jsonl`です。元のRealPersonaChat・MRMPのリポジトリや、医師国家試験SQLiteは変更しません。Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、context lengthは256、seedは42です。会話単位の各2発話目以降を一つの例とし、sequence length 257へ切り、長い例は左側を切り、右側をpadします。応答本文のTokenと直後のEOSだけがmask 1、それ以外はmask 0です。

実験前の実行コードは`60d8bc4`（`fix: separate chat response eos count`）です。最初の整形後レビューで、`response_body_token_count`の集計がEOSを含んでいたことが分かりました。これは学習配列のmaskには影響しない記録上の不整合でしたが、body token数とEOS込みresponse token数を分離して修正しました。修正前に作成したNPZとmanifestは最終成果物として扱わず、修正後に同じコマンドを再実行します。使用コマンドは次のとおりです。

```bash
.venv/bin/python scripts/prepare_chat_sft.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input artifacts/corpus/conversation-v1 \
  --output artifacts/sft/chat-v1-context256 \
  --manifest artifacts/sft/chat-v1-context256.manifest.json \
  --context-length 256 \
  --seed 42
```

NPZは学習に必要なローカル成果物ですが、サイズが大きいためGitには追加しません。manifestには入力JSONL、Tokenizer、NPZのSHA-256、例数、応答Token数、切り詰め例数を残します。成功判定はtrain/validation両方のNPZとmanifestが作成され、応答maskが0件にならず、同じコマンドを再実行したときに決定的な集計と配列ハッシュが得られることです。

## 実験中の記録

修正前の初回実行では、train 396,966例、validation 49,045例を作成しましたが、body token数の記録にEOSが混ざっていたため採用しませんでした。修正後に同じseed・入力・Tokenizer・context lengthで再実行し、split別の会話数、例数、応答Token数、切り詰め数、ファイルSHA-256を確定します。

## 結果と解釈

未実施です。

## 次に試すこと

整形結果を使い、実験017のToken予算pretraining step 500 checkpointから、学習率5e-5・weight decay 0.01の500 step SFTを行います。SFT後は会話SFT validation lossとIssue #1固定promptを確認し、pretrainingのみのモデルと比較します。
