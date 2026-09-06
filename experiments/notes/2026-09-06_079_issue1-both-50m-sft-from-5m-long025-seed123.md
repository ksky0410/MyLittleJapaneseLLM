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

Colab送信用bundleは`/tmp/small_llm-colab-079-XXXXXX.tar.gz`、236,435,471 bytes、SHA-256は`4d455e4fc73b803ae3a900ccf94fa46b050681b4a9f93d52acadcf3ad31f82da`です。bundleには079の実行コード、設定、075 best checkpoint、加工済みSFTデータ、rehearsal Token列、Tokenizerだけを含め、元JSONL、医師国家試験原本、`medilink_analysis`は含めていません。

## 実験中の記録

ここにはColab試行、bundle hash、MPSへの切り替え、step 1・500・1,000・1,500・2,000・2,500・3,000のloss・PPL・経過時間・学習率・固定prompt生成を時系列で追記します。警告、失敗、悪い生成も削除せず残します。

2026年9月6日09:09台に`colab sessions`を実行し、`No active sessions found on server.`を確認しました。その後、`colab new --session exp079-both-50m-sft-long025-seed123 --gpu T4`を実行しましたが、assignment endpointがHTTP 503 `Service Unavailable`を返しました。Colabセッションは作成されず、bundle upload、入力hash検証、モデル初期化、学習stepは発生していません。078までと同じ制約のため、079もMPSへ切り替えます。

同日09:11台に予定したMPSコマンドで学習を開始しました。step 1はtrain loss 4.082140、SFT loss 4.347381、rehearsal loss 3.021178、validation loss 4.342332、PPL 76.8866、learning rate 5e-7、経過時間4.14秒でした。step 100はtrain loss 3.966904、SFT loss 4.164704、rehearsal loss 3.175706、validation loss 3.837496、PPL 46.4091、learning rate 5e-5、経過時間66.09秒でした。step 200はtrain loss 3.440580、SFT loss 3.546707、rehearsal loss 3.016072、validation loss 3.802486、PPL 44.8125、learning rate 4.9871e-5、経過時間209.26秒、step 300はtrain loss 4.151612、SFT loss 4.388435、rehearsal loss 3.204319、validation loss 3.781122、PPL 43.8652、learning rate 4.9479e-5、経過時間334.58秒でした。step 400はvalidation loss 3.779117、PPL 43.7774、learning rate 4.8830e-5、経過時間458.18秒でした。

step 500はtrain loss 3.880764、SFT loss 4.131837、rehearsal loss 2.876472、validation loss 3.744829、PPL 42.3018、learning rate 4.7931e-5、経過時間587.21秒でした。step 500の固定生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんばんは`でした。step 0〜500の生成本文、step 500のcheckpoint metadata、metricsを保存し、GitHubへpushします。NaN、OOM、shape error、警告はありません。

step 600はvalidation loss 3.713040、PPL 40.9782、learning rate 4.6792e-5、経過時間716.76秒でした。step 700は3.713716、PPL 41.0059、learning rate 4.5427e-5、経過時間843.25秒、step 800は3.698069、PPL 40.3693、learning rate 4.3852e-5、経過時間969.19秒でした。step 900は3.675959、PPL 39.4865、learning rate 4.2085e-5、経過時間1,096.51秒、step 1,000はtrain loss 3.854930、SFT loss 4.185379、rehearsal loss 2.533137、validation loss 3.650785、PPL 38.5049、learning rate 4.0147e-5、経過時間1,233.94秒でした。step 1,000で最良validationを更新しています。

step 1,000の固定生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんにちは`でした。step 600〜1,000の生成本文、step 1,000のcheckpoint metadata、metricsを保存し、GitHubへpushします。学習は継続中です。

## 実験終了後の結果と解釈

実際のbackend、最良checkpoint、学習時間、5領域loss、固定chat-testのEOS・長さ・precision・recall・F1、長さ別集計、078との差、生成本文の質的観察を追記します。評価結果、生成全文、checkpoint metadataのSHA-256を残します。

## 次に試すこと

079で同一seedのpaired comparisonを完成させた後、長文比率を採用するか保留します。保留の場合はcontext length 512、データ量追加、蒸留または現代的なinstruction tuningの小規模実験へ進みます。採用する場合も別seedで再確認してから標準条件を更新します。
