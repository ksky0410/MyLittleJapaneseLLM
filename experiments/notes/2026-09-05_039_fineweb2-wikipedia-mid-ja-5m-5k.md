# 実験039：Wikipedia中間比率7.5M Token列の5,000 step学習

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験037ではWikipediaを含む約10M Token列を5,000 step学習し、general validation loss 5.216578、Wikipedia test loss 4.275204、fixed chat F1 0.063502となりました。実験038ではWikipediaなし約5M Token列を同じ5,000 step学習し、general loss 4.986886、Wikipedia test loss 4.674947、fixed chat F1 0.063110となりました。

今回の仮説は、037のWikipedia比率が現在の5M級モデルには高く、Wikipediaを半分程度に下げれば、038に近いgeneral・conversation・medical・FineWeb性能を保ちながらWikipedia testも改善するというものです。Wikipediaをゼロから少量へ増やすことで、domainの広がりと既存分布への反復のバランスを確認します。fixed chat F1は037・038と同程度か、038を上回ることを期待しますが、自動Token overlapだけで会話能力を断定しません。

混合器には、実験037と同じ5 sourceを用い、weightを`aozora=8`、`fineweb=8`、`wikipedia=4`、`conversation=1`、`medical=1`と指定します。targetは7,500,000 Tokenです。weightは論理単位の選択優先度であり、実際のToken比率はsourceごとの単位長と上限で変わるため、作成後のmanifestにある実測Token比率を採用します。これは「Wikipediaを正確に半分のToken比率にする」実験ではなく、037と038の中間を狙う探索条件です。

モデル、Tokenizer、batch size、seed、optimizer、learning-rate schedule、最大stepは037・038と固定します。037・038と同じMacBook MLX backendで学習し、Colab PyTorch backendとの差を混ぜません。

## 使用するデータ、Tokenizer、モデル

sourceは次の5つです。元の医師国家試験データは`/Users/koseki/projects/medilink_analysis`側を変更せず、リポジトリ内の既存加工済み`artifacts/corpus/medical-qb-v2/train.txt`を読み取り専用で使います。

- 青空文庫一般本文：`artifacts/corpus/aozora-general-v1.txt`
- FineWeb2 Edu Japanese：`artifacts/corpus/fineweb2-edu-japanese-v1/train.txt`
- Wikimedia Wikipedia日本語本文：`artifacts/corpus/wikimedia-wikipedia-ja-v1.txt`
- 公開会話コーパス：`artifacts/corpus/conversation-v1/train.txt`
- 医師国家試験データの加工済み本文：`artifacts/corpus/medical-qb-v2/train.txt`

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。学習Token列、混合manifest、各sourceのhashはデータ作成後に追記します。

モデルはdim 240、6層、6 heads、context length 256、MLP倍率4、absolute position embedding、概算5,197,920 parametersです。batch size 8、最大5,000 step、evaluation/sample interval 100、evaluation batches 20、AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42です。設定は`configs/fineweb2-wikipedia-mid-ja-5m-5k.toml`です。

## 実行前の再現情報

実験038を完了したcommitは`e84db48`です。config SHA-256は`e0673e6e44f8ac652e78d0b60ffec5cbb7e4e996a72c8be424b58e2425bd25c4`です。データ混合とToken化を先に実行し、manifestとToken列のhashを確認してから学習を開始します。

予定コマンドは次のとおりです。

```bash
.venv/bin/python scripts/mix_corpora.py \
  --source aozora=artifacts/corpus/aozora-general-v1.txt \
  --source fineweb=artifacts/corpus/fineweb2-edu-japanese-v1/train.txt \
  --source wikipedia=artifacts/corpus/wikimedia-wikipedia-ja-v1.txt \
  --source conversation=artifacts/corpus/conversation-v1/train.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --weight aozora=8 \
  --weight fineweb=8 \
  --weight wikipedia=4 \
  --weight conversation=1 \
  --weight medical=1 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --target-tokens 7500000 \
  --seed 42 \
  --output artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.txt \
  --manifest artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.manifest.json

.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.txt \
  --output artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin

.venv/bin/python scripts/train.py --config configs/fineweb2-wikipedia-mid-ja-5m-5k.toml
```

成功基準は、混合とToken化が決定的に完了し、5,000 step学習をNaN、shape error、データ長エラー、メモリ不足なく完走することです。学習中は1,000 stepを超えない間隔でmetricsと生成TXTを記録し、完了後は037・038と同じdomain評価とfixed chat評価を実行します。

## 実験中の記録

未実施です。データ作成、Token化、学習の各段階で実際の条件と失敗を追記します。

## 結果と解釈

未実施です。

## 次に試すこと

未実施です。037・038・039を比較した後、最もよいsource比率を固定して20M級モデルへ移行します。構造変更はデータ条件が決まってからRoPE、RMSNorm、SwiGLU、GQAの順に一つずつ試します。20M以上の長時間学習や大規模Token列はColab GPUを優先します。
