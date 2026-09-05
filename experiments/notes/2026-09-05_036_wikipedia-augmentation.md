# 実験036：日本語Wikipedia追加によるデータ量比較

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験029では約5M TokenのFineWeb混合コーパスを2,500 stepまで学習し、実験030では同じデータと学習条件のまま5M級から約20M級へ拡張しました。20M化によってvalidation lossは改善しましたが、固定promptの文法と固定chat-testの意味対応はまだ弱く、約5M Tokenが容量に対して少なすぎる可能性があります。

今回の仮説は、FineWebに加えて日本語Wikipedia本文を約5M Token投入し、学習候補を約10M Tokenへ増やせば、5Mモデルでも一般validationとWikipedia由来validationのlossが改善するというものです。一方でWikipediaは百科事典文体であり会話ではないため、会話validationや固定chat-testが改善するとは仮定しません。会話lossが悪化する場合は、source比率の問題として次のsource ablationで調べます。

実験029・030との比較可能性を優先し、モデル、Tokenizer、batch size、seed、optimizer、learning rate schedule、context length、最大stepを固定します。変更するのは学習Token列のsource構成だけです。Wikipediaの取得・抽出処理は実験031に分離し、元parquetを再取得せず、その出力本文を使用します。

## データとTokenizer

新しい学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1-train.bin`です。作成時には次の5 sourceを使い、weightを`aozora=8`、`fineweb=8`、`wikipedia=8`、`conversation=1`、`medical=1`とします。

- 青空文庫一般本文：`artifacts/corpus/aozora-general-v1.txt`
- FineWeb2 Edu Japanese抽出本文：`artifacts/corpus/fineweb2-edu-japanese-v1/train.txt`
- Wikimedia Wikipedia日本語抽出本文：`artifacts/corpus/wikimedia-wikipedia-ja-v1.txt`
- 公開会話コーパス：`artifacts/corpus/conversation-v1/train.txt`
- 医師国家試験データ：`artifacts/corpus/medical-qb-v2/train.txt`

Token予算は10,000,000とし、論理単位を分割・複製せず、既存Tokenizerで測ったEOS込みToken数が上限を超えないように選びます。source間の同一本文は混合器のsource順で最初の一つだけを採用します。混合条件、実際のsource別Token数、入力hash、出力hashは`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.manifest.json`へ保存します。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`を固定します。一般・会話・医療validationとFineWeb testは実験030と同じToken列を使用し、Wikipedia追加の影響は後続でWikipedia専用validationを作成して確認します。

## モデルと学習条件

モデルは実験029と同じ5,197,920 parameter概算のdim 240、6層、6 heads、context length 256、MLP倍率4、absolute position embedding、重み共有を使用します。batch size 8、最大2,500 step、evaluation interval 100、sample interval 100、evaluation batches 20、AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42です。設定は`configs/fineweb2-wikipedia-augmented-ja-5m-2p5k.toml`へ固定します。

本学習の前に`configs/fineweb2-wikipedia-augmented-ja-5m-smoke.toml`で100 stepのsmokeを行い、データ長、モデルshape、Metal、NaNの有無を確認します。smoke出力と本学習出力は別ディレクトリに保存し、既存の実験029のcheckpointとsampleを上書きしません。

## 実行コマンド

```bash
.venv/bin/python scripts/mix_corpora.py \
  --source aozora=artifacts/corpus/aozora-general-v1.txt \
  --source fineweb=artifacts/corpus/fineweb2-edu-japanese-v1/train.txt \
  --source wikipedia=artifacts/corpus/wikimedia-wikipedia-ja-v1.txt \
  --source conversation=artifacts/corpus/conversation-v1/train.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --weight aozora=8 \
  --weight fineweb=8 \
  --weight wikipedia=8 \
  --weight conversation=1 \
  --weight medical=1 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --target-tokens 10000000 \
  --seed 42 \
  --output artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.txt \
  --manifest artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.manifest.json

.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.txt \
  --output artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1-train.bin

.venv/bin/python scripts/inspect_model.py --config configs/fineweb2-wikipedia-augmented-ja-5m-2p5k.toml
.venv/bin/python scripts/train.py --config configs/fineweb2-wikipedia-augmented-ja-5m-smoke.toml
.venv/bin/python scripts/train.py --config configs/fineweb2-wikipedia-augmented-ja-5m-2p5k.toml
```

学習開始前の公開commitは`79cfda1`です。smoke configのSHA-256は`37868d1dded50b3d3c2310a2acf720fb005b22e45b71fa53c632aed147625ae3`、本学習configは`667a6752cbf9565aa50d0737a383cc1fc17cb7c09547431403a8871c4166a6d0`、`scripts/train.py`は`e8f600df408f53772b3f0729c1d8047a656e1f63b0e8907e04d6502eae612ee0`です。実行環境はPython 3.13.1、既存の`.venv`へ導入済みのMLXです。smokeを先に完走させ、成功後に本学習を開始します。

## 実験中の記録

2026-09-05、実験031で抽出したWikipedia本文と既存4 sourceを用いた混合Token列を準備しました。混合manifestは`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.manifest.json`、出力本文のSHA-256は`4ddbc8da19ab87663a3d94e44db2d5a881993679f38c19c42df41c813fd8b305`、実際のToken数は9,999,973です。Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1-train.bin`、SHA-256は`d043d06180d2c6deb0e0c14038fd1b3f736f86f062cf61260bd19282f8ce48e4`です。Token metadataではvocab size 4,096、EOS ID 3を確認しました。

混合後の実Token比率は、青空文庫5.08%、FineWeb42.18%、Wikipedia42.19%、会話5.27%、医療5.27%でした。指定weightは単位採用の希望比率であり、sourceごとの文書長が違うためToken比率とは一致しません。Wikipediaは738記事しかないため646記事でToken予算に到達しており、追加sourceとしての影響は確認できますが、日本語Wikipedia全体の代表性を示すものではありません。

Wikipedia専用validationも作成しました。本文manifestは`artifacts/corpus/wikimedia-wikipedia-ja-validation-v1.manifest.json`で、Token化後のToken数は998,845、Token列`artifacts/tokens/wikimedia-wikipedia-ja-validation-v1.bin`のSHA-256は`2898e8ab7385dc7beb26e4ba956639eaa791b059a1a7e763ae9d4b958e09d269`です。学習はまだ開始していません。学習実行前に混合manifest、Token列hash、config hash、Python・MLXの実行環境を確認し、smokeと本学習の各節目を追記します。

## 結果と解釈

データ混合とWikipedia専用validationのToken化は成功しました。モデル学習と評価は未実施です。学習前処理だけであり、Wikipedia追加によるlossや生成品質の効果はまだ判断できません。

## 次に試すこと

Wikipedia追加でvalidation lossが改善した場合は、同じ10M Tokenで20Mモデルへ拡張し、容量とデータ量の交互作用を調べます。会話testが悪化した場合は、Wikipediaを一般sourceとして別の比率に下げ、会話と医療の比率を固定したsource ablationを行います。現代的な構造変更は、データ比較を終えてからRoPE、RMSNorm、SwiGLUの順に一要素ずつ導入します。
