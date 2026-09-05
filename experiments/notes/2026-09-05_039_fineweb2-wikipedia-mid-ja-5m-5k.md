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

2026-09-05、weight `8:8:4:1:1`、seed 42、target 7,500,000 Tokenで混合を完了しました。混合本文は`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.txt`、実際のToken数は7,499,997、本文SHA-256は`6a0ea02b8d0e5baf83c05e1d487c53fce6e9b8b1056d06c440caa546948aca31`です。manifestは`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.manifest.json`、SHA-256は`1206a38179282b5e9f271b73100bfc028c712de2cee67d7577d941cfab49a69f`です。実測Token比率は、青空文庫6.777%、FineWeb53.282%、Wikipedia26.632%、会話6.651%、医療6.657%でした。Wikipediaは037の42.189%と038の0%の中間に近い比率になりました。

Token化も完了しました。Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin`、7,499,997 Token、SHA-256は`3bad9f5f9546d98fc598d602a053648679d6e7817161f0add7a219b020c7440a`です。Token metadataは`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin.json`、SHA-256は`0d187e0964b403d38d95d70fb82128f495e32e82beb23f0048a1dbf41ed20fd3`です。vocab size 4,096、EOS ID 3、Tokenizer SHA-256は計画どおりです。学習開始条件を満たしました。

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

データ混合とToken化は正常に完了しました。混合器のエラー、Token数不足、Tokenizer mismatchは発生していません。学習開始前の計画、実測source比率、本文とToken列のhashをこのノートへ記録済みです。2026-09-05に5,000 stepのMLX学習を開始しました。step 1ではtrain loss 8.781590、general validation loss 8.800259、perplexity 6635.964でした。warmup中のstep 300ではtrain loss 5.823933、validation loss 6.379888、perplexity 589.862、learning rate `3.000000e-4`、経過時間31.31秒でした。step 1,000ではtrain loss 4.936625、validation loss 5.706295、perplexity 300.755、learning rate `2.855307e-4`、経過時間123.05秒、step 1,300ではtrain loss 4.772246、validation loss 5.586837、perplexity 266.890、step 1,600ではtrain loss 4.473612、validation loss 5.485989、perplexity 241.287、step 1,700ではtrain loss 4.479951、validation loss 5.478712、perplexity 239.538、step 1,800ではtrain loss 4.845807、validation loss 5.441229、perplexity 230.726、step 1,900ではtrain loss 4.241998、validation loss 5.422994、perplexity 226.556、step 2,000ではtrain loss 4.191769、validation loss 5.415304、perplexity 224.821、step 2,100ではtrain loss 4.514877、validation loss 5.371833、perplexity 215.257、step 2,400ではvalidation loss 5.336113、step 2,500では5.321655、step 2,600では5.331264、step 2,700では5.292813、step 2,800では5.274889、step 2,900では5.248127、step 3,000では5.254659、step 3,100では5.223692、step 3,200では5.233642、step 3,300では5.219729、step 3,400では5.216045、step 3,500では5.191690、step 3,600では5.194812、step 3,700では5.174340、step 3,800では5.167134、step 3,900では5.158949、step 4,100では5.140823まで下がりました。step 2,500と3,000、3,200、3,600、4,000で小さな反発はありましたが、その後に回復しています。step 4,100時点でNaN、shape error、データ長エラー、メモリ不足は発生しておらず、学習は継続中です。step 300から4,100までの生成結果は`artifacts/samples/fineweb2-wikipedia-mid-ja-5m-5k/`へ保存されています。

step 4,200ではvalidation loss 5.138404、step 4,300では5.129627、step 4,400では5.124960まで下がりました。step 4,400時点でもNaN、shape error、データ長エラー、メモリ不足は発生しておらず、学習は継続中です。

## 結果と解釈

未実施です。

## 次に試すこと

未実施です。037・038・039を比較した後、最もよいsource比率を固定して20M級モデルへ移行します。構造変更はデータ条件が決まってからRoPE、RMSNorm、SwiGLU、GQAの順に一つずつ試します。20M以上の長時間学習や大規模Token列はColab GPUを優先します。
