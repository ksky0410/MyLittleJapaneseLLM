# 実験046：RMSNorm + SwiGLU・RoPE・context 512の日本語5M smoke

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験042のLayerNorm・GELU条件、実験043のRMSNorm・GELU条件、実験045のLayerNorm・SwiGLU条件を基準にし、046ではRMSNormとSwiGLUを同時に使います。目的は、二つの現代的な構造要素を組み合わせたときに、単独導入の効果が保たれるか、相乗効果または干渉が見えるかを確認することです。

今回で、同じデータ・Tokenizer・モデル幅・層数・context・seed・学習条件による2×2の探索表が揃います。GELU/LayerNormは042、GELU/RMSNormは043、SwiGLU/LayerNormは045、RMSNorm/SwiGLUは046です。仮説は、RMSNormとSwiGLUが独立に有効なら、046のvalidation lossは043と045の少なくとも一方より低くなる可能性がある、というものです。一方、小規模な500 step smokeでは初期化差や学習揺らぎが支配的になり、組み合わせによる改善が見えない可能性もあります。結果をもって現代的構成全体の優劣とは断定しません。

実験045との差分は`model.norm_type = "rmsnorm"`だけで、`ffn_type = "swiglu"`は維持します。RoPE、dim 240、6層、6 heads、context length 512、MLP倍率4、batch size 8、最大500 step、評価・生成間隔100、AdamW、学習率3e-4から3e-5、warmup 300、weight decay 0.1、seed 42、学習・検証Token列、Tokenizerを固定します。SwiGLUの中間次元は640です。biasを含む実際のparameter数はおよそ5,142,480で、045の5,145,600より3,120少なくなります。概算parameter数は5,133,360です。

## 使用するデータ、Tokenizer、コード

学習Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`で、1,336,619 Token、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`で、11,780 Token、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。既存の一般・会話・医療混合コーパスから作ったToken列を読み取り専用で使用し、元の医師国家試験データや`medilink_analysis`の原本には変更を加えません。

TokenizerはSentencePiece Unigram、語彙数4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。設定は`configs/rmsnorm-swiglu-context512-mixed-ja-5m-smoke.toml`です。コードは`ffn_type`と`norm_type`をcheckpointのmodel signatureへ含め、過去のGELU・LayerNorm checkpointは省略項目を既定値として読み込めます。

予定している学習コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py \
  --config configs/rmsnorm-swiglu-context512-mixed-ja-5m-smoke.toml
```

学習後は`evaluate_domains.py`でgeneral・conversation・medical・FineWeb2 Edu Japanese・Wikipediaを評価し、`evaluate_chat_prompts.py`でIssue #1の固定会話promptを評価します。checkpoint reload後には`今日は`、`吾輩は`、会話marker、`問題：`を生成します。stepごとの生成文、崩れた出力、空の出力、評価JSON、checkpoint metadataは削除せず保存します。

## 成功条件と判定方法

500 stepをNaN、Metalエラー、OOM、Token列不足なく完走し、metrics、checkpoint metadata、stepごとの生成TXTが保存されれば実装上の成功とします。checkpoint reloadと固定prompt生成が成功することも確認します。品質は042、043、045のgeneral・conversation・medical lossと比較します。046が最良でも、500 stepの1 seedだけでRMSNormとSwiGLUの相乗効果を確定しません。生成文は自然さを過大評価せず、医学的正確性も主張しません。

## 実験中の記録

2026-09-05 13:03 JST、学習開始前に設定・入力Token列・TokenizerのSHA-256、現在のGit commit、Python・macOS・MLX device、同じ出力先を使う別プロセスの有無を確認しました。実行環境はPython 3.13.1、macOS 15.5 arm64、MLXのdeviceは`Device(gpu, 0)`です。コードcommitは`c7a6b8f`、設定のSHA-256は`e4c9144c634c4513746d5f63cac2e92bd408862f6272c65ead606b5d4b14db5d`です。同じ出力先を使う別プロセスはありませんでした。このノートと設定をcommit・pushしてから学習を開始します。学習中は100 stepごとにtrain loss、validation loss、perplexity、生成文、所要時間を保存し、異常や予定変更があればその時点で追記します。

（ここへ開始時刻、実行環境、stepごとの記録、警告、停止理由を追記する。）

## 結果と解釈

（ここへ実際の最終loss、最良checkpoint、学習時間、各domain評価、固定会話prompt、reload結果、042・043・045との差分を追記する。未実施または失敗の場合も、その事実と原因をそのまま記録する。）

## 次に試すこと

（結果に基づいて、次に変更する条件を一つか二つだけ具体的に記録する。）
