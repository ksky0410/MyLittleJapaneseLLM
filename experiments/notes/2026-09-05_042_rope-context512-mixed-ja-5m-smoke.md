# 実験042：RoPEとcontext length 512の日本語5M smoke

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。ColabのT4・L4割当が連続して利用できなかったため、MacBookのMLX/Metalで実行できる範囲の構造実験を先に進めます。実験014では、同じvocab 4,096、dim 240、6層、6 heads、500 stepのRoPEモデルをcontext length 256で学習し、general validation loss 5.5338133176を記録しました。

今回の仮説は、RoPEモデルのcontext lengthだけを256から512へ伸ばすと、長い文脈を一度に扱えるため、general validation lossまたは生成文の文脈維持が改善する可能性があるというものです。一方、学習stepは500のままなので、1 stepあたりの処理Token数が2,048から4,096へ倍増し、同じ計算step数でも学習総Token量が増えます。そのため、結果は「context長の効果」だけでなく「学習Token予算の増加」を含む探索的比較として扱います。長いcontextの利点がまだ現れず、attention計算の増加だけが表れる可能性も記録します。

## 使用するデータ、Tokenizer、モデル

学習Token列は実験014と同じ`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`で、Token数は1,336,619、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。混合manifestは`artifacts/corpus/mixed-ja-80-10-10-v2.manifest.json`で、単位比率は一般80%、会話10%、医療10%、Tokenizer後のToken比率は実験013記録どおり一般36.57%、会話39.69%、医療23.74%です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、11,780 Token、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。元の医師国家試験データと加工済みコーパスは変更せず、既存Token列を読み取り専用で使います。

TokenizerはSentencePiece Unigram、語彙数4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。モデルはRoPE、dim 240、6層、6 heads、context length 512、MLP倍率4です。RoPEのhead dimensionは40で偶数です。位置embeddingを持たないため、実験014のcontext 256モデルより概算parameter数は同じ5,136,480です。batch size 8、最大500 step、evaluation/sample interval 100、evaluation batches 20、AdamW、learning rate 3e-4から3e-5、warmup 300、weight decay 0.1、seed 42を使います。

設定は`configs/rope-context512-mixed-ja-5m-smoke.toml`です。学習前に設定、Token列、TokenizerのhashとGit commitを固定します。予定コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py \
  --config configs/rope-context512-mixed-ja-5m-smoke.toml
```

学習後は、general validation lossを実験014と比較し、context 512で保存したstepごとの生成文を削除せず確認します。生成結果だけで長期記憶や会話能力を主張せず、まず学習が正常に進んだか、長い入力が扱えるか、出力の崩れ方が変わったかを記録します。

## 成功条件

500 stepをNaN、Metalエラー、OOM、Token列不足なく完走し、step 0と100 step以下の間隔でmetrics、checkpoint metadata、生成TXTを保存することです。context length 512のforward、checkpoint reload、固定prompt生成が成功することも確認します。失敗した生成文や途中停止の成果物は削除しません。

## 実行前の再現情報

このノートと設定をcommit・pushした後、そのcommit SHAをここへ追記します。学習開始時のMLX、Python、macOS、Metal情報、設定SHA-256は開始直前に追記します。

## 実験中の記録

## 結果と解釈

## 次に試すこと

general lossが改善した場合は、同じcontext 512でstep数を変えた比較、または20M級への拡張を検討します。改善しない場合は、context長を固定したままRMSNormまたはSwiGLUを一つだけ導入し、構造変更の影響を分離します。Colab T4が再び利用可能になった場合は、041の新規kernel probe、bundle hash検証、10,000 step延長を優先します。
