# 実験044：LayerNorm・context 512のToken予算対照

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験042ではcontext 512を500 step学習したため、context 256・500 stepの実験014より1 stepあたりの処理Token数が2倍でした。実験044では、実験042と同じLayerNorm・RoPE・context 512モデルを250 stepだけ学習し、context 256・500 step条件と総処理Token数を約1.024M Tokenで揃えます。これにより、実験042で見えた改善が長いcontextそのものによるのか、単に学習Token量が多かったためなのかを一部切り分けます。

今回の仮説は、学習Token予算を揃えるとcontext 512のvalidation loss改善幅は実験042より小さくなる、または消える可能性があるというものです。長いcontextが有効なら、処理Token数を揃えてもcontext 256と同等以上になると予想します。ただし、context 512は一度に見る系列が長い一方、250 stepではoptimizer更新回数が半分になるため、系列長の効果と更新回数の効果は完全には分離できません。この限界を明記した探索的対照として扱います。

実験042との差分は、context 512を維持したまま`max_steps = 250`とし、記録間隔を50 stepへ変更した点です。norm_typeはLayerNormで、実験042と同じです。dim 240、6層、6 heads、MLP倍率4、batch size 8、AdamW、学習率3e-4から3e-5、warmup 300、weight decay 0.1、seed 42、学習・検証Token列、Tokenizerを固定します。context 512・batch 8・250 stepでは、1,024,000 Token相当を処理します。モデルの概算parameter数は5,136,480です。

## 使用するデータ、Tokenizer、コード

学習Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`で、Token数は1,336,619、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`で、11,780 Token、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。会話validationは`artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin`、医療validationは`artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin`を使用します。一般80%、会話10%、医療10%の混合コーパスから作った既存Token列を読み取り専用で使用し、元の医師国家試験データや`medilink_analysis`の原本には変更を加えません。

TokenizerはSentencePiece Unigram、語彙数4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。実験設定は`configs/layernorm-context512-mixed-ja-5m-token-budget-smoke.toml`、SHA-256は`1c4542676a4732df465949a62d9cc177f0b18b6276ecd174d522284301d0a5a3`です。使用コードの基準commitは`c9098c5`で、RMSNorm対応とcheckpoint互換性の実装は`4d1b459`です。

予定している学習コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py \
  --config configs/layernorm-context512-mixed-ja-5m-token-budget-smoke.toml
```

学習後は、`scripts/evaluate_domains.py`でgeneral・conversation・medical・fineweb・wikipediaのvalidation lossを測定し、`scripts/evaluate_chat_prompts.py`でIssue #1の固定会話promptを評価します。checkpoint reload後には`今日は`、`吾輩は`、会話marker、`問題：`を生成します。学習途中の全生成文、崩れた出力、空の出力、評価JSON、checkpoint metadataを削除せず保存します。

## 成功条件と判定方法

250 stepをNaN、Metalエラー、OOM、Token列不足なく完走し、metrics、checkpoint metadata、stepごとの生成TXTが保存されれば実装上の成功とします。checkpoint reloadと固定prompt生成が成功することも確認します。品質比較では、主に実験014のcontext 256・LayerNorm・500 stepとgeneral validation lossを比較します。今回のgeneral lossが低くても、更新回数が少ないため「context 512が優れる」と即断しません。実験042との比較では、学習Token予算を揃えたときの差を記録します。

## 実験中の記録

2026-09-05 12:50 JST、学習開始前に設定、コードcommit、Tokenizer、学習Token列、general validation Token列のSHA-256を確認しました。実行環境はPython 3.13.1、macOS 15.5 arm64、MLXのdeviceは`Device(gpu, 0)`です。同じ出力先を使う別の学習プロセスがないことも確認しました。設定とノートはこの後のcommitで固定してから開始します。学習中は50 stepごとにloss、perplexity、生成文、所要時間を保存し、異常や予定変更があれば直ちに追記します。

（ここへ開始時刻、実行環境、stepごとの記録、警告、停止理由を追記する。）

## 結果と解釈

（ここへ実際の最終loss、最良checkpoint、学習時間、評価結果、生成例、実験014・042との差分を追記する。未実施または失敗の場合も、その事実と原因をそのまま記録する。）

## 次に試すこと

（結果に基づいて、次に変更する条件を一つか二つだけ具体的に記録する。）
