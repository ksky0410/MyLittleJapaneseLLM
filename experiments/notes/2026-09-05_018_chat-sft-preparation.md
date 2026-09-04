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

修正前の初回実行では、train 396,966例、validation 49,045例を作成しましたが、body token数の記録にEOSが混ざっていたため採用しませんでした。修正版を同じseed・入力・Tokenizer・context lengthで再実行し、次の結果を確定しました。処理時間は約122.09秒で、エラーはありませんでした。

- train：11,635会話、396,966例、応答本文5,506,080 token、EOS込み応答5,903,046 token、切り詰め276,668例
- validation：1,454会話、49,045例、応答本文689,615 token、EOS込み応答738,660 token、切り詰め34,090例
- train NPZ SHA-256：`400b8ffbc5b3752eaa16e003dab168c75e0a77046ac61c39630ef2409a73e609`
- validation NPZ SHA-256：`5f52b3f4269e914184834d6e13d800604827abfd96f2b4c1ff5f665cd3f8f7b4`

各NPZは`[例数, 256]`のinput、target、loss maskを持ち、maskが0件ではないことを確認しました。NPZ本体はGitへ追加せず、次のmanifestに入力・Tokenizer・NPZのSHA-256と集計を保存します。

## 結果と解釈

全会話の2発話目以降をSFT例へ展開できました。trainとvalidationの会話数は元のsplitと一致し、短すぎて除外された会話はありませんでした。context length 256に収まらない履歴は左側から切り詰めていますが、応答部分を可能な限り残す設計です。trainの約69.7%、validationの約69.6%が切り詰め対象であり、現在の会話例では長い履歴を常に保持できません。この点はSFT結果の解釈に残る制約です。

EOSは応答終了を学習するためmask対象ですが、body token数とは分けて記録できました。これにより、学習する応答本文量と応答終了記号の量を区別して後から確認できます。

- [SFT整形manifest](../../artifacts/sft/chat-v1-context256.manifest.json)

## 次に試すこと

整形結果を使い、実験017のToken予算pretraining step 500 checkpointから、学習率5e-5・weight decay 0.01の500 step SFTを行います。SFT後は会話SFT validation lossとIssue #1固定promptを確認し、pretrainingのみのモデルと比較します。
