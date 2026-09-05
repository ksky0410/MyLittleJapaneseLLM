# 実験043：RoPE・context 512・RMSNormの日本語5M CPU smoke

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験042では、vocab 4,096、dim 240、6層、6 headsの5M級モデルにRoPEとcontext length 512を使い、500 stepでgeneral validation loss 5.341098を記録しました。042はMLX/Metalで実行したため、今回のPyTorch CPU実行とはbackendが異なりますが、RMSNormを実装したモデルが学習・保存・再読込まで動作するかをまず確認します。

今回の仮説は、LayerNormをRMSNormへ置き換えてもcausal LMの学習が安定し、同じ小型モデルでvalidation lossと生成の崩れ方に差が現れる可能性があるというものです。RMSNormは平均の減算を行わず二乗平均平方根だけで正規化するため、LayerNormより計算とパラメータを少し減らせます。一方、500 stepの短いCPU smokeで得られるloss差は小さく、042との比較にはMLXとPyTorch、初期化と数値精度の差も含まれるため、構造の優劣を断定しません。

## 使用するデータ、Tokenizer、モデル

学習Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、1,336,619 Token、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、11,780 Token、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。一般・会話・医療の混合比率は既存の`mixed-ja-80-10-10-v2`を引き継ぎ、元の医師国家試験データは変更せず、既存の加工済みToken列を読み取り専用で使用します。

TokenizerはSentencePiece Unigram、語彙数4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。モデルはdim 240、6層、6 heads、context length 512、MLP倍率4、RoPE、RMSNormで、推定parameter数は5,133,360です。RMSNormによりLayerNorm条件より正規化のscale/bias構成が変わるため、parameter数も学習後のmetadataで確認します。

設定は`configs/rmsnorm-context512-mixed-ja-5m-smoke.toml`です。batch size 8、最大500 step、evaluation/sample interval 100、evaluation batches 20、AdamW、learning rate 3e-4から3e-5、warmup 300、weight decay 0.1、seed 42を使います。Metal deviceがこのシェルから見えないため、実行はPyTorch 2.14.0のCPU、`--device cpu --no-amp`で行います。CPU実行を理由に結果を042のMLX値へ直接順位付けしません。

## 実行前の再現情報

RMSNorm実装を追加したcommitは`4d1b459`、PyTorch forwardテストを追加したcommitは`0048b3a`、設定を追加したcommitは`7e871fc`です。実験開始時点のGit commit、設定SHA-256、実行環境はこのノートへ追記します。設定SHA-256は`8658c4c9b5977189f57536242d531d9f3be791394d90b7297def4a444fc471cc`です。

予定コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train_torch.py \
  --config configs/rmsnorm-context512-mixed-ja-5m-smoke.toml \
  --device cpu --no-amp
```

成功条件は、500 stepをNaN、shape error、Token列不足、checkpoint reloadエラーなく完走し、step 0と100 step間隔のmetrics・生成文・checkpoint metadataを保存することです。完了後は同じCPU条件でgeneral、conversation、medicalのdomain評価と固定chat評価を行います。生成文は良い結果だけでなく、崩れた出力も削除せず保存します。

## 実験中の記録

開始前に、MLXのMetal deviceがこのheadless実行環境から見えないことを確認しました。そのため、RMSNormのforwardと学習経路をPyTorch CPUで検証します。学習開始時刻、PyTorch version、CPU情報、stepごとのmetrics、異常の有無を完了後に追記します。

## 結果と解釈

未実施です。

## 次に試すこと

未実施です。完了後、042のLayerNorm・RoPE・context 512結果と比較し、backend差を明記します。実装が安定していれば、同じRMSNorm条件でSwiGLUを一つだけ追加するか、PyTorch CPUでLayerNorm対照をそろえて構造差を切り分けます。
