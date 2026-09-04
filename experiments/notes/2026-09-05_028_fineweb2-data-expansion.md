# 実験028：一般日本語データの追加による5Mモデルの変化

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験027で、短文samplingは出力を短く止める効果を示した一方、長い応答の内容保持や文脈適合は改善しませんでした。現在の5M級モデルは、拡張した青空文庫・会話・医療を混ぜた約1M Tokenの学習データを使っており、一般日本語の多様性がボトルネックになっている可能性があります。今回はモデル構造やTokenizerを変えず、一般日本語データだけを追加して同じstep数で比較します。

今回の仮説は、教育的品質でフィルタされた日本語Web文書を追加すると、同じ5M級モデル・同じ500 stepでも、一般日本語validation lossが下がり、固定promptや会話評価で極端に短い・崩れた生成が少し減るというものです。反対に、学習stepを増やさずデータだけを増やすと1つの例に触れる回数が減るため、短期のvalidation lossや生成が悪化する可能性もあります。結果が改善しても、データ量・文書多様性・出典の差をまとめた効果として扱い、FineWeb2 Edu全体の優位性とは主張しません。

## データと出典

追加データは、Hugging Face上の`hotchpotch/fineweb-2-edu-japanese`にある`small_tokens_cleaned`のtrain shard一つです。取得対象のdataset commitは`180ca004c6a89b590daaad86cb062a07a5353c69`（2025-05-09）に固定します。dataset cardによれば、これはFineWeb2から教育的スコア2.5以上の日本語文書を抽出したデータで、`small_tokens_cleaned`は512 Token以下の文書からWeb特有のノイズを除去したsubsetです。ライセンスはODC-By 1.0で、元データ由来のCommon Crawl Terms of Useも適用されます。

dataset cardには、`small_tokens`と`small_tokens_cleaned`の先頭範囲に重複があるため、最初の20,000件を飛ばすよう注意書きがあります。したがってtrain shardの先頭20,000行を採用せず、その後から現在のTokenizerで約5M Token分を決定的に抽出します。test parquetは学習へ混ぜず、同じTokenizerで別のvalidation Token列を作成して、追加データ側のlossを確認します。元のparquetは`data/downloads/`へ保存し、Gitへ追加しません。元データや`medilink_analysis`内のSQLiteは変更しません。

取得URLは次のとおりです。

```text
https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/resolve/180ca004c6a89b590daaad86cb062a07a5353c69/small_tokens_cleaned/train-00000-of-00283.parquet?download=true
https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/resolve/180ca004c6a89b590daaad86cb062a07a5353c69/small_tokens_cleaned/test-00000-of-00001.parquet?download=true
```

追加sourceは既存の`aozora-general-v1.txt`、`conversation-v1/train.txt`、`medical-qb-v2/train.txt`と、FineWeb抽出本文を`mix_corpora.py`へ渡します。既存sourceとFineWeb sourceの希望weightはそれぞれ8、会話と医療は1とし、合計5,000,000 Token以下を選びます。sourceの実際のToken比率は、FineWeb shardの有限サイズと論理単位長によってweightからずれるため、mix manifestの値を採用します。

## 固定する条件

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`をそのまま使います。モデルはvocab sizeをTokenizerから読み、dim 240、6層、6 heads、context length 256、MLP倍率4、absolute position embeddingの約5M parameterです。batch size 8、最大500 step、evaluation interval 100、sample interval 100、evaluation batches 20、AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42を維持します。

対照条件は`artifacts/tokens/mixed-ja-token-budget-1m-train.bin`を使った実験017のstep 500です。処置条件は同じモデル設定で、FineWebを加えた5M Token以下のToken列だけを変更します。対照と処置のvalidationは同じ既存general validation Token列を使い、さらにFineWeb test由来のvalidationを別のdomain評価として記録します。これにより、既存general validationの改善と新sourceへの適応を分けて確認します。

## 実行前の再現情報

実験027を記録してpushした直後のHEADは`9f6347c`です。FineWeb取り込みスクリプト、設定、manifestの作成後にコードをcommitし、そのcommitを使ってデータ取得・Token化・学習を実行します。依存パッケージを追加する場合は、`.venv`へ導入したこととバージョンを記録し、`pyproject.toml`の任意extraにも再現用の指定を追加します。

開始前に、入力parquetのSHA-256、ファイルサイズ、row数、skip row数、抽出した文書数、抽出本文のSHA-256、Tokenizer SHA-256、選択Token数、source別の採用比率をmanifestへ保存します。学習開始前にはこのノートへ実行コマンドと使用commitを追記し、学習中は1,000 stepを超えない間隔でmetricsと生成文を保存します。今回は500 stepのため、100 stepごとのmetricsとsampleを残します。

## 成功・失敗の判断

parquetを指定commitから取得し、重複注意の先頭20,000行を除外し、入力と出力のhashをmanifestへ保存できればデータ準備を成功とします。既存元データに変更がなく、追加sourceを含むToken列が5,000,000 Token以下で作成され、モデルが500 stepまでNaNやshape errorなしに完走し、checkpoint・metrics・生成TXTを保存できることを学習成功の基準とします。

一般validation lossが対照より0.05以上低下し、FineWeb validation lossも有限値として計算できれば、データ追加が短期の言語モデリング指標へ有望な影響を与えたと判断します。固定会話評価は実験027のtest manifestを再利用し、EOS・生成長・Token overlapを保存します。ただし48例のうちtrain本文との完全一致候補が7例、履歴のcontext超過が33例あるため、会話能力の改善をこの実験だけで確定しません。

## 実行コマンド

データ取得、parquet抽出、混合、Token化、モデル形状確認、学習、domain評価、固定会話評価の順に実行します。実行前に導入スクリプトと設定のcommitを記録し、予定から変更があればこのノートへ追記します。

## 実験中の記録

2026-09-05に、dataset commit `180ca004c6a89b590daaad86cb062a07a5353c69`からtrain shardとtest shardを取得しました。導入したデータ処理依存は`pyarrow 25.0.1`で、`.venv`には`ensurepip`でpipを復旧してから導入しました。導入スクリプトと任意依存の定義は`c359082`（`feat: add pinned FineWeb Japanese importer`）としてGitHubへpush済みです。

train parquetは268,597,958 bytes、SHA-256は`a38a4b50e7aee2e9c2ca1eeb96858d794751c1a4c2d2f6f1119964fc8d4d6838`でした。先頭20,000行を除外し、空行と本文完全一致の重複を除きながら、現行Tokenizerで5,000,000 Tokenを超えないところまで9,796文書を抽出しました。実際の選択Token数は4,999,748、走査行数は29,797、空行は0、skip後の重複除去は0です。抽出本文のSHA-256は`471869caa73aa5987a52a2dbcfa28846441d0729ff03bb0c05db0fa461e3890f`で、条件は`artifacts/corpus/fineweb2-edu-japanese-v1-train.manifest.json`へ保存しました。

test parquetは9,843,060 bytes、SHA-256は`2dbc0824036cc083b4e52f249006c66d99b724be0ecb7d5e4ae0c3c7332dc534`でした。test側はskipせず全8,082行を走査し、本文完全一致の重複4,041行を除いて4,041文書、2,061,459 Tokenを抽出しました。抽出本文のSHA-256は`f7cb1fd629f75095399becdcff0eed9306bf339203131988addca14158d3a297`で、条件は`artifacts/corpus/fineweb2-edu-japanese-v1-test.manifest.json`へ保存しました。dataset cardの重複注意に従い、trainでは先頭20,000行をskipし、testでは重複除去の実績を記録しています。入力parquetと抽出本文はGit管理対象外ですが、manifestは追跡対象にします。

現時点ではデータ準備が成功し、学習はまだ開始していません。次に、既存3 sourceへFineWeb抽出本文を追加した5M Token以下の混合コーパスを作成し、source別Token比率とhashを確認してからToken化・学習へ進みます。

その後、`mix_corpora.py`を次の条件で実行しました。

```bash
.venv/bin/python scripts/mix_corpora.py \
  --source aozora=artifacts/corpus/aozora-general-v1.txt \
  --source fineweb=artifacts/corpus/fineweb2-edu-japanese-v1/train.txt \
  --source conversation=artifacts/corpus/conversation-v1/train.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --weight aozora=8 --weight fineweb=8 \
  --weight conversation=1 --weight medical=1 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --target-tokens 5000000 --seed 42 \
  --output artifacts/corpus/mixed-ja-token-budget-fineweb2-5m-v1.txt \
  --manifest artifacts/corpus/mixed-ja-token-budget-fineweb2-5m-v1.manifest.json
```

混合結果は5,000,000 Tokenの上限内で4,999,958 Token、14,392単位、35,767行、6,653,005文字でした。出力本文のSHA-256は`6259d9ada9cc92a1498942724d4365e5041501ff6a7dd0ac4f2d3a3068f1a0ff`です。Token比率はFineWeb 71.867983%、青空文庫10.165645%、会話8.982855%、医療8.983515%でした。FineWeb sourceの有限サイズに対してweightを8にしたため、希望weight比率そのものではなく、sourceの採用可能量を反映した結果になっています。混合manifestは`artifacts/corpus/mixed-ja-token-budget-fineweb2-5m-v1.manifest.json`へ保存します。

この時点では元の1Mコーパス、会話・医療の元split、FineWebの抽出本文を変更していません。次にこの混合本文を既存TokenizerでToken化し、Token数とSHA-256を確認してから学習を開始します。

## 結果と解釈

未実施です。

## 次に試すこと

データ追加の効果を確認した後、同じ5M Token前後の学習データとTokenizerを固定し、dim 384・8層程度の20M級モデルへ拡張します。モデル容量を変える実験ではデータ追加の処置を重ねず、今回の処置条件を基準にした別実験として記録します。
