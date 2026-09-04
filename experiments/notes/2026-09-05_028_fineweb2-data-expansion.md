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

未実施です。

## 結果と解釈

未実施です。

## 次に試すこと

データ追加の効果を確認した後、同じ5M Token前後の学習データとTokenizerを固定し、dim 384・8層程度の20M級モデルへ拡張します。モデル容量を変える実験ではデータ追加の処置を重ねず、今回の処置条件を基準にした別実験として記録します。
