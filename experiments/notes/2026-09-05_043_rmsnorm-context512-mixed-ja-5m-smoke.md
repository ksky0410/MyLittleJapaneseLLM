# 実験043：RMSNorm・context 512の日本語5M smoke

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験042で使用したRoPE・context length 512の5Mモデルを基準にし、正規化層だけをLayerNormからRMSNormへ変更します。今回の目的は、現代的なdecoder-only Transformerで広く使われるRMSNormを、この小型日本語モデルへ導入したときに、学習の安定性、validation loss、生成結果、実行時間がどのように変わるかを確認することです。

仮説は、RMSNormは平均を引かず、LayerNormより計算とパラメータが少ないため、同じ学習条件でも少なくとも同程度に学習できる可能性がある、というものです。ただし、このモデルは非常に小さく、学習Token数も少ないため、差がノイズに埋もれる可能性があります。RMSNormが常に優れるとは仮定せず、validation lossの推移と生成文の崩れ方をLayerNormの実験042と並べて判断します。

実験042との差分は`model.norm_type = "rmsnorm"`だけです。RoPE、context length 512、dim 240、6層、6 heads、MLP倍率4、batch size 8、最大500 step、評価間隔100、生成間隔100、AdamW、学習率3e-4から3e-5、warmup 300、weight decay 0.1、seed 42、学習・検証Token列、Tokenizerを固定します。RMSNormでは正規化のscaleだけを持つため、概算parameter数は5,133,360で、実験042のLayerNormモデル5,136,480より3,120少なくなります。

## 使用するデータ、Tokenizer、コード

学習Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`で、Token数は1,336,619、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`で、11,780 Token、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。一般80%、会話10%、医療10%の混合コーパスから作った既存Token列を読み取り専用で使用し、元の医師国家試験データや`medilink_analysis`の原本には変更を加えません。

TokenizerはSentencePiece Unigram、語彙数4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。実験設定は`configs/rmsnorm-context512-mixed-ja-5m-smoke.toml`、SHA-256は`8658c4c9b5977189f57536242d531d9f3be791394d90b7297def4a444fc471cc`です。RMSNorm対応コードのcommitは`4d1b459`、実験設定を追加したcommitは`7e871fc`です。

予定している学習コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py \
  --config configs/rmsnorm-context512-mixed-ja-5m-smoke.toml
```

学習後は、`scripts/evaluate_domains.py`でgeneral・conversation・medicalのvalidation lossを測定し、`scripts/evaluate_chat_prompts.py`でIssue #1の固定会話promptを評価します。さらに、checkpointをロードして固定promptを再生成します。stepごとの生成文、崩れた出力、空の出力、評価JSON、checkpoint metadataは削除せず保存します。

## 成功条件と判定方法

500 stepをNaN、Metalエラー、OOM、Token列不足なく完走し、metrics、checkpoint metadata、stepごとの生成TXTが保存されれば実装上の成功とします。checkpoint reloadと固定prompt生成が成功することも確認します。モデル品質については、実験042のLayerNorm条件とvalidation lossを比較します。RMSNorm側のlossが低ければ探索上の改善、差が小さければ同等、明確に高ければこの条件では悪化と判定します。生成文が短く崩れていても削除せず、lossと併せて解釈します。

## 実験中の記録

学習開始前に、このノート、設定、コードcommit、入力Token列、TokenizerのSHA-256を確認します。学習中は設定の評価間隔に従って少なくとも100 stepごとにloss、perplexity、生成文、所要時間を保存します。異常や予定変更があれば、その時点で追記します。

（ここへ開始時刻、実行環境、stepごとの記録、警告、停止理由を追記する。）

## 結果と解釈

（ここへ実際の最終loss、最良checkpoint、学習時間、評価結果、生成例、実験042との差分を追記する。未実施または失敗の場合も、その事実と原因をそのまま記録する。）

## 次に試すこと

（結果に基づいて、次に変更する条件を一つか二つだけ具体的に記録する。）
