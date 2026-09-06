# 実験100：Runpod 50M best checkpointから未使用20M tokensを継続事前学習

## 目的

実験098で作成したRunpod A40上の50M日本語モデルを、既存の学習列と重ならない日本語コーパスで継続事前学習する。実験098のcheckpointは約2,000万tokensの混合コーパスを40,000 step、総提示約8,192万tokensで学習しているが、自然な日本語と質問への応答はまだ弱い。今回の実験では、モデル構造とTokenizerを変えず、データの新規性と継続学習量だけを増やす。

## 事前仮説

未使用のFineWeb2とWikipediaを約2,000万tokens追加し、同じ50Mモデルへさらに40,000 step提示すれば、一般日本語のvalidation lossと固定生成の文法的な安定性が改善する可能性がある。特に、実験098で学習した文章の単純な再提示ではなく、新しい文書を中心にすることで、語彙・固有名詞・説明文の範囲が広がると予想する。

ただし、これは会話SFTではないため、質問に直接答える能力の改善は保証しない。事前学習の改善と会話応答の改善を混同しないよう、今回のcheckpointを後続のSFTの基盤として評価する。

## 実験条件

- 実施日：2026年9月7日
- 担当：Codex
- 事前学習基盤：`artifacts/checkpoints/issue1-both-50m-pretrain-20m-40k-runpod-cuda/best.pt`
- 基盤checkpointのstep：36,000
- 基盤checkpointのSHA-256：`83e8be941b645823efd1ae0a358d2c4521faa49b58de7696229298973bd25ac7`
- モデル：dim 576、12層、9 heads、RoPE、LayerNorm、SwiGLU、context 256、約50.2M parameters
- Tokenizer：`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`
- TokenizerのSHA-256：`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- 学習：batch size 8、context 256、40,000 step、1 step 2,048 tokens、約81.92M提示tokens
- optimizer：AdamW。optimizer stateはcheckpointから引き継がず、重みだけを初期値として読み込む
- 学習率：`1e-4`から`1e-5`までcosine decay、warmup 1,000 step、weight decay 0.1
- seed：100
- 評価：FineWeb2 Japanese test、500 stepごと、20 evaluation batches、生成文も500 stepごとに保存
- 使用GPU：Runpod A40を第一候補とする。前回実測は約14.5 step/秒、料金は約$0.49/時

## 新規データ設計

実験098で使用した範囲を避け、同じローカルparquetの後続行から抽出する。

- FineWeb2 Edu Japanese：`train-00000-of-00283.parquet`の`skip_rows=40000`以降から最大10M tokens。実験098の追加FineWebは約39,820行までを読んでいるため、その後から開始する
- Wikipedia shard 0：`train-00000-of-00015.parquet`の`skip_rows=739`以降から最大5M tokens。既存Wikipedia v1は先頭約5M tokens、739行までを使用している
- Wikipedia shard 1：`train-00001-of-00015.parquet`の`skip_rows=2902`以降から最大5M tokens。実験098の追加Wikipediaは1,000行目から2,901行目までを使用している

3つの抽出結果を、FineWeb 2、Wikipedia shard 0 1、Wikipedia shard 1 1の重みで混ぜ、Tokenizer上限20,000,000 tokensの新規学習列を作る。抽出範囲は既存manifestで固定し、混合manifestには入力・出力SHA-256と実際の採用tokensを記録する。既存の`medilink_analysis`原データと医師国家試験原本は読み取り専用で保持し、今回の事前学習列にはまだ混ぜない。

## 開始前の予想と成功基準

FineWeb test lossが実験098のbest評価値2.973267を下回る、または固定promptの生成で空応答・極端な反復・記号崩壊が明確に減ることを有望な結果とする。ただし、lossの改善だけで自然な会話の成功とは判定しない。後続で同じ48例のchat-test、新しい未使用会話test、人手レビューを実施する。

40,000 stepまでNaN、OOM、shape errorなく完走し、500 stepごとのmetrics、生成サンプル、checkpoint metadataを保存することを実行上の成功基準とする。途中で停止した場合も、最後のmetricsと停止理由をこのノートへ追記する。

## 実行コマンド予定

データ抽出と混合：

```bash
PYTHONPATH=scripts uv run python scripts/import_fineweb2_edu_japanese.py \
  --input data/downloads/fineweb-2-edu-japanese-180ca00/train-00000-of-00283.parquet \
  --output artifacts/corpus/fineweb2-edu-japanese-v3/continuation-train.txt \
  --manifest artifacts/corpus/fineweb2-edu-japanese-v3/continuation-train.manifest.json \
  --skip-rows 40000 --max-tokens 10000000 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model

PYTHONPATH=scripts uv run python scripts/import_wikimedia_wikipedia_ja.py \
  --input data/downloads/wikimedia-wikipedia-20231101-ja/train-00000-of-00015.parquet \
  --output artifacts/corpus/wikimedia-wikipedia-ja-v3/shard0-continuation.txt \
  --manifest artifacts/corpus/wikimedia-wikipedia-ja-v3/shard0-continuation.manifest.json \
  --skip-rows 739 --max-tokens 5000000 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model

PYTHONPATH=scripts uv run python scripts/import_wikimedia_wikipedia_ja.py \
  --input data/downloads/wikimedia-wikipedia-20231101-ja/train-00001-of-00015.parquet \
  --output artifacts/corpus/wikimedia-wikipedia-ja-v3/shard1-continuation.txt \
  --manifest artifacts/corpus/wikimedia-wikipedia-ja-v3/shard1-continuation.manifest.json \
  --skip-rows 2902 --max-tokens 5000000 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model

PYTHONPATH=scripts uv run python scripts/mix_corpora.py \
  --source fineweb=artifacts/corpus/fineweb2-edu-japanese-v3/continuation-train.txt \
  --source wikipedia0=artifacts/corpus/wikimedia-wikipedia-ja-v3/shard0-continuation.txt \
  --source wikipedia1=artifacts/corpus/wikimedia-wikipedia-ja-v3/shard1-continuation.txt \
  --weight fineweb=2 --weight wikipedia0=1 --weight wikipedia1=1 \
  --target-tokens 20000000 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --seed 10001 \
  --output artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-continuation-20m-v1.txt \
  --manifest artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-continuation-20m-v1.manifest.json

PYTHONPATH=scripts uv run python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-continuation-20m-v1.txt \
  --output artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-continuation-20m-v1-train.bin
```

Runpodでの学習：

```bash
PYTHONPATH=scripts uv run python scripts/train_torch.py \
  --config configs/issue1-both-50m-pretrain-continuation-20m-40k-runpod-cuda.toml \
  --initial-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-20m-40k-runpod-cuda/best.pt \
  --device cuda
```

## データ準備結果

開始前の計画どおり、既存データの範囲を避けて3つの抽出を実行した。FineWeb2の入力SHA-256は`a38a4b50e7aee2e9c2ca1eeb96858d794751c1a4c2d2f6f1119964fc8d4d6838`で、`skip_rows=40000`から19,125文書、9,999,538 tokensを抽出した。出力SHA-256は`9143be2a0dcb6403c41867772c26e8b40e7b6e299b892d925a2b929d4dc28fa0`である。

Wikipedia shard 0は入力SHA-256`4751c14478e712fd637bd83c2cf3537b0e299ea5115e9a78ddededf42f34c29d`、`skip_rows=739`から654文書、4,995,173 tokensを抽出した。出力SHA-256は`d1528c80f01e3ccc53bb18c86fa275178e5eadda3e02b01c590d5b9aa0b5c236`である。Wikipedia shard 1は入力SHA-256`ccb1b97328fee31da6c62eb2d0de5cfaff6c9c27b95c2a8706978aecd9028a7`、`skip_rows=2902`から1,866文書、4,998,623 tokensを抽出した。出力SHA-256は`abe96c97527a651feef8aa3abc669187ca2118013ff11e8daaf536e02321cb1e`である。

3つをFineWeb 2、Wikipedia shard 0 1、Wikipedia shard 1 1の重みで混合した結果、重複除去後の学習本文は21,645単位、19,993,334 tokensとなった。source別token比率はFineWeb 50.0144%、Wikipedia shard 0 24.9842%、shard 1 25.0014%である。混合本文のSHA-256は`01be70ac22c857fb6988afe978aa20d19886c30772cd875a799a55ae48c085fe`で、manifestは`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-continuation-20m-v1.manifest.json`に保存した。

Tokenizerでのencodeも完了し、binaryは19,993,334 tokensとなった。学習binaryとmetadataは、それぞれ`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-continuation-20m-v1-train.bin`と同名`.json`に保存した。ここまでデータ準備では元のparquet、既存corpus、`medilink_analysis`を変更していない。

## 実験中の追記

ここには、データ抽出結果、入力hash、bundle hash、Pod ID、GPU、料金、速度、各500 stepのlossと生成文、警告、途中停止を発生時点で追記する。

### 開始時の追記

データ準備後のテストは`PYTHONPATH=scripts uv run pytest -q`で`108 passed`となった。設定と開始前ノートはcommit`896a1c3`としてGitHubの`main`へpush済みである。Runpodへ送ったbundleは205MB、SHA-256は`beb3beb7339d2e81946d1fca61f1234ef3cb3bb1e97db5a2a750c8a22c3efac4`である。

2026年9月7日、Runpod A40 Secure Pod `j9c46julmtbcb4`（CA-MTL-1、約$0.49/時）で入力を展開した。Pod上でA40 46,068MiB、PyTorch 2.9.1+cu128、CUDA 12.8、AMPを確認し、不足していたNumPy 2.5.3とSentencePiece 0.2.2をPodのPython環境へ導入した。bundleのhashはローカルと一致し、best checkpointのSHA-256は`83e8be941b645823efd1ae0a358d2c4521faa49b58de7696229298973bd25ac7`、新規train binaryのSHA-256は`f19878618870a487ce5b0aab6970d6d72b2ef71ab76ee79520e7c3fe3341dec1`となった。

本番前の20 step smokeを本番とは別の`artifacts/checkpoints/exp100-smoke`へ実行した。50,207,616 parameters、CUDA AMP、NaN・OOM・shape errorなしで完走し、step 1のFineWeb validation lossは2.97326596、step 20は2.97170031だった。smokeの学習率はwarmup中であり、本番性能の判定には使わない。checkpoint読み込み、新規token列、評価、生成、保存処理が接続されていることを確認した。

smoke完了後、同じPod上で本番40,000 stepを`configs/issue1-both-50m-pretrain-continuation-20m-40k-runpod-cuda.toml`、初期重み`artifacts/checkpoints/issue1-both-50m-pretrain-20m-40k-runpod-cuda/best.pt`、`--device cuda`で開始した。初期optimizer stateはなく、Runpod上の`/workspace/exp100/train.log`へ標準出力を保存している。Podを削除せず、500 stepごとにloss・生成文・経過時間を確認する。

## 実験終了後の追記

ここには、最終step、best step、FineWeb testと各domain loss、固定chat-test、新規会話test、人手レビュー、学習時間、最大メモリ、Runpod料金、次に会話SFTへ渡すcheckpointのパスを追記する。
