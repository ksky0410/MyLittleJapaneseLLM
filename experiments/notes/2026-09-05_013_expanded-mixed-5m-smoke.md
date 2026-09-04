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

## Tokenizerとencodeの結果

Tokenizerは混合train 32,281行をすべて読み込み、skipなく完了した。実際の語彙数は4,096で、modelのSHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`、vocabのSHA-256は`bafd084943d817d27a1b873192471c5e5f6073c6768daee36e5d334eb3c4e180`である。

混合trainは1,336,619 token、一般validationは11,780 token、会話validationは1,104,647 token、医療validationは197,674 tokenへ符号化された。各binのSHA-256は、trainが`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`、一般validationが`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`、会話validationが`e78281538e34ad3b895ced68676b3f3386b1d265806cd5901fd777d4700bffc8`、医療validationが`649483667e3d2328dbb5b59188e637d60b43ef10344371b5ce7855dd6551bb2c`である。

混合出力の各単位を元sourceへ照合し、同じTokenizerとEOS挿入規則で数えたsource別token寄与は、一般488,856 token（36.57%）、会話530,471 token（39.69%）、医療317,292 token（23.74%）だった。単位数は80/10/10でも、token数は大きく異なる。原因は会話ブロックと医療1問の平均長であり、今後の比率実験では単位比率とtoken比率を別々の条件として扱う。

## 結果

学習は500 stepまでエラー、NaN、OOM、途中停止なく完走し、所要時間は94.19秒であった。stepごとの主要値は次のとおりである。

| step | train loss | validation loss | validation perplexity |
|---:|---:|---:|---:|
| 1 | 8.7928 | 8.8004 | 6,636.73 |
| 100 | 5.2139 | 6.6892 | 803.69 |
| 200 | 5.6807 | 6.3517 | 573.44 |
| 300 | 4.4124 | 6.0763 | 435.40 |
| 400 | 4.8887 | 5.8463 | 345.97 |
| 500 | 4.9487 | 5.7359 | 309.81 |

最良checkpointは`artifacts/checkpoints/expanded-mixed-ja-5m-smoke/step_000500.npz`で、validation loss 5.7359479268、perplexity 309.8065055140である。domain別評価は次のとおりで、詳細は`artifacts/evaluations/expanded-mixed-ja-5m-smoke-domains.json`に保存した。

| domain | token数 | validation loss | perplexity |
|---|---:|---:|---:|
| 一般 | 11,780 | 5.7359 | 309.81 |
| 会話 | 1,104,647 | 3.7156 | 41.08 |
| 医療 | 197,674 | 4.9252 | 137.71 |

実験009のabsolute基準と比べると、一般lossは5.7469から5.7359へ0.0110改善した。一方、会話は3.5532から3.7156へ、医療は4.9087から4.9252へ悪化した。Tokenizerが異なること、会話・医療のvalidationも異なるTokenizerで再符号化されていることから、この差を一般source追加の効果だけとは解釈しない。学習データが増えたことで、500 stepでの見かけの過学習が弱まった可能性はあるが、より長いstepと同一Tokenizerの対照実験が必要である。

source別に数えたtrain token寄与は、一般488,856 token（36.57%）、会話530,471 token（39.69%）、医療317,292 token（23.74%）だった。単位数は80/10/10でも、会話が長いためtoken比率は大きく異なる。この実験で、今後は単位quotaではなくTokenizer後のtoken budgetを基準にする必要性がさらに明確になった。token比率80/10/10を目標にするなら、会話・医療の採用単位を減らすか、source別のtoken budget抽出を追加する。

stepごとの生成文とreload後の生成文は`artifacts/samples/expanded-mixed-ja-5m-smoke/`へ保存した。`今日は`に対してstep 500では短い日本語文が出たが、reload後にすぐEOSとなる出力もあり、安定した生成には至っていない。`吾輩は`では複数作品由来と思われる文体が混ざった長い出力になった。会話promptは短い応答を返したが、医療promptは問題形式をほぼ維持せず、医療知識の正確性は確認できなかった。代表例は[reloaded-today.txt](../../artifacts/samples/expanded-mixed-ja-5m-smoke/reloaded-today.txt)、[reloaded-story.txt](../../artifacts/samples/expanded-mixed-ja-5m-smoke/reloaded-story.txt)、[reloaded-conversation.txt](../../artifacts/samples/expanded-mixed-ja-5m-smoke/reloaded-conversation.txt)、[reloaded-medical.txt](../../artifacts/samples/expanded-mixed-ja-5m-smoke/reloaded-medical.txt)である。

学習、reload、domain別評価は成功した。軽量artifactとノートはGitHubへpushし、巨大なToken・Tokenizer・checkpoint本体はGit管理対象外とする。次は、token budgetを指定できる混合処理を追加するか、同じ拡張データでRoPEを再比較するかを、今回のsource別token比率を踏まえて決める。
