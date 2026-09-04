# 実験ノート：拡張一般source上のRoPE 5M smoke

## 基本情報

- 実験番号：014
- 記録日：2026-09-05
- 担当者：ユーザーとCodex
- 実験開始時のGitコミット：`8dd401a`
- ブランチ：`main`
- 状態：学習開始前

## 仮説

実験010では、吾輩は猫である。を中心にした小さな混合コーパスでRoPE側のvalidation lossがabsolute側より低かった。今回は実験013の拡張一般sourceを使い、データの作品数を増やした状態でも同じ傾向が出るか確認する。データ、Tokenizer、モデルサイズ、optimizer、seed、step数は実験013と同じで、位置表現だけをabsoluteからRoPEへ変更する。

RoPEが拡張データでも一貫して低いなら、少なくともこの小型設定での候補として優先する価値がある。ただし、absoluteとRoPEでは学習可能なposition embeddingの有無により乱数初期化の順序が変わるため、同じseedでも完全に同じ初期値ではない。今回も探索的な比較として扱い、長い学習または初期値を厳密に揃えた追試を残す。

## 条件

- train Token：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`
- 一般validation：`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`
- 会話validation：`artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin`
- 医療validation：`artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin`
- Tokenizer：実験013と同じSentencePiece Unigram 4,096語彙
- 設定：`configs/expanded-rope-mixed-ja-5m-smoke.toml`
- モデル：RoPE、dim240、6層、6 heads、context256、MLP ratio4、推定5,136,480 parameters
- optimizer：AdamW。learning rate 3e-4から3e-5、warmup300、weight decay0.1
- batch size：8。最大500 step。seed42
- 出力：`artifacts/checkpoints/expanded-rope-mixed-ja-5m-smoke/`、`artifacts/samples/expanded-rope-mixed-ja-5m-smoke/`

学習コマンドは次のとおりである。

```bash
.venv/bin/python scripts/train.py \
  --config configs/expanded-rope-mixed-ja-5m-smoke.toml
```

学習後には、同じcheckpointを`evaluate_domains.py`へ渡し、一般・会話・医療のlossとperplexityを測定する。固定promptは`今日は`、`吾輩は`、会話marker付きprompt、`問題：`とし、stepごとの生成文も保存する。

## 成功条件

500 stepをNaN、OOM、途中停止なく完走し、RoPE checkpointをreloadできること。実験013と同じToken列・validationで比較できるmetrics、domain評価、生成文、所要時間が揃うこと。悪い生成も削除しない。

## 結果

RoPE学習は500 stepまでエラー、NaN、OOM、途中停止なく完走し、所要時間は120.40秒であった。stepごとの主要値は次のとおりである。

| step | train loss | validation loss | validation perplexity |
|---:|---:|---:|---:|
| 1 | 8.7796 | 8.8061 | 6,674.73 |
| 100 | 5.1355 | 6.6499 | 772.72 |
| 200 | 5.5513 | 6.3058 | 547.75 |
| 300 | 4.2604 | 5.9724 | 392.46 |
| 400 | 4.6055 | 5.6675 | 289.32 |
| 500 | 4.7141 | 5.5338 | 253.11 |

最良checkpointは`artifacts/checkpoints/expanded-rope-mixed-ja-5m-smoke/step_000500.npz`で、validation loss 5.5338133176、perplexity 253.1072514355である。domain別評価は次のとおりで、詳細は`artifacts/evaluations/expanded-rope-mixed-ja-5m-smoke-domains.json`に保存した。

| domain | token数 | validation loss | perplexity |
|---|---:|---:|---:|
| 一般 | 11,780 | 5.5338 | 253.11 |
| 会話 | 1,104,647 | 3.5064 | 33.33 |
| 医療 | 197,674 | 4.5083 | 90.77 |

実験013のexpanded absoluteと比較すると、一般lossは5.7359から5.5338へ0.2021、会話は3.7156から3.5064へ0.2092、医療は4.9252から4.5083へ0.4169下がった。今回もabsoluteとRoPEで学習可能パラメータの有無が違い、同じseedでも乱数初期化の順序が変わる。したがって、この差はRoPE単独の確定的効果ではなく、次の厳密な初期値固定または複数seed追試が必要である。

step 0から500までの生成文とreload後の固定prompt生成は`artifacts/samples/expanded-rope-mixed-ja-5m-smoke/`へ保存した。`今日は`では短い日本語の続き、会話promptでは`こんにちは!`、医療promptでは問題形式に近い語列が出た。一方、`吾輩は`から医療の選択肢解説へ流れる生成もあり、一般文書と医療形式の分離はできていない。これらは小さいデータ・モデルでの観測であり、医学的正確性や会話能力を示すものではない。代表例は[reloaded-today.txt](../../artifacts/samples/expanded-rope-mixed-ja-5m-smoke/reloaded-today.txt)、[reloaded-story.txt](../../artifacts/samples/expanded-rope-mixed-ja-5m-smoke/reloaded-story.txt)、[reloaded-conversation.txt](../../artifacts/samples/expanded-rope-mixed-ja-5m-smoke/reloaded-conversation.txt)、[reloaded-medical.txt](../../artifacts/samples/expanded-rope-mixed-ja-5m-smoke/reloaded-medical.txt)である。

学習、reload、domain別評価は成功した。metrics、summary、checkpoint metadata、生成文、domain別評価JSON、このノートはGitHubへpushする。巨大な`.npz`とToken・Tokenizer本体はGit管理対象外とする。次は、token budget基準の混合処理を追加してデータ源の寄与を制御することを優先する。RoPEについては、必要なら複数seed追試またはcontextを512へ伸ばす実験を行う。
