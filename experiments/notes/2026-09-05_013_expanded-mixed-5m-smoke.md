# 実験ノート：一般作品を増やした混合日本語5M smoke

## 基本情報

- 実験番号：013
- 記録日：2026-09-05
- 担当者：ユーザーとCodex
- 実験開始時のGitコミット：`06256bc`
- ブランチ：`main`
- 状態：Tokenizer作成前。学習は未実施

## 仮説と比較

実験009の混合コーパスに、坊っちゃん・こころ・それからを追加した一般sourceを使う。一般の作品数とunique単位数を増やすことで、同じ5Mモデル・同じ500 stepでも、一般validation lossと生成文の反復が改善するかを確認する。会話・医療の採用単位数は前回と同じ比率にし、データ量と作品多様性の差を主な変更点とする。

比較対象は実験009のabsolute 5M smokeである。実験009の一般validation lossは5.7469、会話3.5532、医療4.9087だった。ただし、Tokenizerも変わるため、結果はデータ多様性だけの完全な因果比較ではない。Tokenizer後には、混合全体のtoken数に加えて、可能な範囲でsource別のtoken寄与も記録する。

## データと設定

- train本文：`artifacts/corpus/mixed-ja-80-10-10-v2.txt`
- 混合manifest：`artifacts/corpus/mixed-ja-80-10-10-v2.manifest.json`
- 混合条件：seed 42、target 7,000単位、一般・会話・医療の重み8.0・1.0・1.0
- 実採用単位：一般5,600、会話700、医療700
- Tokenizer：SentencePiece Unigram、語彙数4,096、train本文だけで学習、最大文長20,000 UTF-8 bytes
- 一般validation：`artifacts/corpus/aozora-neko-formal-v2/val.txt`
- 会話validation：`artifacts/corpus/conversation-v1/validation.txt`
- 医療validation：`artifacts/corpus/medical-qb-v2/validation.txt`
- 学習設定：`configs/expanded-mixed-ja-5m-smoke.toml`
- モデル：absolute position embedding、dim240、6層、6 heads、context256、MLP ratio4、推定5,197,920 parameters
- optimizer：AdamW。learning rate 3e-4から3e-5、warmup300、weight decay0.1
- batch size：8。最大500 step。seed42
- 1 stepあたり2,048 token、処理量は約1.024M token

## 実行前コマンド

```bash
.venv/bin/python scripts/train_tokenizer.py \
  --input artifacts/corpus/mixed-ja-80-10-10-v2.txt \
  --model-prefix artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram \
  --vocab-size 4096 \
  --model-type unigram \
  --max-sentence-length 20000

.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input artifacts/corpus/mixed-ja-80-10-10-v2.txt \
  --output artifacts/tokens/mixed-ja-80-10-10-v2-train.bin

.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input artifacts/corpus/aozora-neko-formal-v2/val.txt \
  --output artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin

.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input artifacts/corpus/conversation-v1/validation.txt \
  --output artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin

.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input artifacts/corpus/medical-qb-v2/validation.txt \
  --output artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin
```

学習は、Tokenizerと4種類のToken列の作成・ハッシュ確認後に開始する。

```bash
.venv/bin/python scripts/train.py \
  --config configs/expanded-mixed-ja-5m-smoke.toml
```

## 成功条件

Tokenizerが混合train全行を読み、Token列の件数とSHA-256をmetadataへ残すこと。500 stepをNaN、OOM、途中停止なく完走し、checkpointをreloadできること。general・conversation・medicalのdomain別評価JSONと、stepごとの生成文が保存されること。悪い生成や反復があっても削除しない。

## 結果

Tokenizer・encode・学習・domain別評価の各結果を、実行直後に追記する。最良checkpoint、loss、perplexity、所要時間、生成文へのリンク、実験009との差、source別token寄与を記録する。軽量artifactとノートはGitHubへpushし、巨大なToken・Tokenizer・checkpoint本体はGit管理対象外とする。
