# 実験079：seed 123で長文応答25%層化SFTを対照比較

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。Issue #1の現代的な会話データを、一般日本語と医師国家試験由来の医療データを含む共通の日本語モデルへ組み込みます。医療専用モデルにはせず、一般・会話・医療・RPC・MRMPを分けて評価します。元の`/Users/koseki/projects/medilink_analysis`と医師国家試験の原データは変更・削除しません。

実験078では、075の50M・約5M Token事前学習済みbaseに標準response-only SFTをseed 123で行いました。実験077では、同じbase・データ・seed42でSFT部分の応答24 Token以上を2/6へ増やしましたが、077と076のseedが異なるため、長文比率の効果を特定できませんでした。本実験は078とbase、SFTデータ、rehearsal、seed、学習率、step、評価条件を完全に揃え、長文比率だけを0から0.25へ変更します。

仮説は、長文比率0.25にするとlong F1、平均生成長、長い履歴に対するToken overlapが改善する一方、short F1は低下することです。078との差が小さい、または悪化する場合は、50M・約5M Token基盤ではSFTバッチ内の長文oversamplingを採用せず、データ形式や評価方法へ進みます。今回は同じseedのpaired ablationなので、077よりも078との差を主な判断材料にします。

## 再現条件

モデルはRoPE・LayerNorm・SwiGLU、dim 576、12層、9 heads、context length 256、50,207,616 parametersです。設定上のseedは123、batch sizeは8、最大3,000 step、learning rateは5e-5から5e-6、warmup 100、weight decay 0.01、EOS loss weight 0.50です。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`、SFT trainは64,423例、validationは49,045例です。SFTとrehearsalは0.80対0.20で混ぜ、SFT部分6例から応答24 Token以上の長文を2例抽出します。

base checkpointは実験075のbestでSHA-256は`71931b2c689c2fbaa31c8c92c022a21fac571894ec2993a59be48644794e5e17`です。SFT trainは`645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、validationは`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`、rehearsal Token列は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`、Tokenizerは`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。設定ファイルのSHA-256は実験開始前に計算して追記します。
設定ファイル`configs/issue1-both-50m-sft-from-5m-long025-seed123-3k.toml`のSHA-256は`118c011929e67f963e93825bef4494780521b23c6891b9e31fa0a0ec3a059176`です。

再現コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-50m-sft-from-5m-long025-seed123-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-50m-sft-from-5m-long025-seed123-3k \
  --samples-dir artifacts/samples/issue1-both-50m-sft-from-5m-long025-seed123-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --long-response-ratio 0.25 \
  --long-response-min-tokens 24 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

Colab T4の割り当てを開始前に試し、HTTP 503などで失敗した場合は応答とsession状態を記録してMPSへ切り替えます。成功条件は3,000 stepを完走し、100 stepごとのmetricsと生成文、500 stepごとのcheckpoint metadata、summary、5領域評価、固定chat-test 48例、人手レビュー用JSONを保存することです。

## 実験中の記録

ここにはColab試行、bundle hash、MPSへの切り替え、step 1・500・1,000・1,500・2,000・2,500・3,000のloss・PPL・経過時間・学習率・固定prompt生成を時系列で追記します。警告、失敗、悪い生成も削除せず残します。

## 実験終了後の結果と解釈

実際のbackend、最良checkpoint、学習時間、5領域loss、固定chat-testのEOS・長さ・precision・recall・F1、長さ別集計、078との差、生成本文の質的観察を追記します。評価結果、生成全文、checkpoint metadataのSHA-256を残します。

## 次に試すこと

079で同一seedのpaired comparisonを完成させた後、長文比率を採用するか保留します。保留の場合はcontext length 512、データ量追加、蒸留または現代的なinstruction tuningの小規模実験へ進みます。採用する場合も別seedで再確認してから標準条件を更新します。
