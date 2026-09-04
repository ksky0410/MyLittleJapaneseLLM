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

学習直後にstep別metrics、最良checkpoint、perplexity、domain別loss、固定promptの生成結果、実験013との差を追記する。軽量artifactとこのノートはGitHubへpushし、`.npz`・Token・Tokenizer本体はGit管理対象外とする。
