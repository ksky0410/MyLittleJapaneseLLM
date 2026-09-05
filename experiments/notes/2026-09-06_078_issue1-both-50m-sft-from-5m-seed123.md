# 実験078：50M基盤への標準SFTをseed 123で再確認

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。Issue #1の会話データ、一般日本語、医師国家試験由来の医療データを同じ日本語モデルへ役割分担して使う方針を維持します。医療専用モデルにはせず、一般・会話・医療・RPC・MRMPを分けて評価します。元の`/Users/koseki/projects/medilink_analysis`と医師国家試験の原データは変更・削除しません。

実験076は50M・約5M Token事前学習済みbaseへ標準response-only SFTをseed 42で行い、実験077は同じbase・データ・seedでSFT部分の長文応答を2/6へ増やしました。077では長文F1が076とほぼ同じで、全体F1は低下しました。過去の20M実験でも長文層化はseedによる分散が大きかったため、077の比較を単一seedの結果にしないよう、今回は長文層化を行わない標準条件をseed 123で再実行します。

仮説は、seed 123の標準SFTが076と近い5領域lossを示し、固定chat-testのF1はseed分散の範囲に収まることです。この対照が得られた後、同じseed 123で長文比率0.25を実行すれば、base・データ・学習seedを固定した長文比率の対比較ができます。今回は長文比率を0、rehearsal ratioを0.20、EOS loss weightを0.50に固定し、他の変更を入れません。

## 再現条件

モデルはRoPE・LayerNorm・SwiGLU、dim 576、12層、9 heads、context length 256、50,207,616 parametersです。設定上のseedは123、batch sizeは8、最大3,000 step、learning rateは5e-5から5e-6、warmup 100、weight decay 0.01です。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`、SFT trainは64,423例、validationは49,045例です。SFTはresponse-only loss、SFTとrehearsalは0.80対0.20で混ぜます。

base checkpointは実験075のbestでSHA-256は`71931b2c689c2fbaa31c8c92c022a21fac571894ec2993a59be48644794e5e17`です。SFT trainは`645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、validationは`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`、rehearsal Token列は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`、Tokenizerは`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。設定ファイルのSHA-256は実験開始前に計算して追記します。
設定ファイル`configs/issue1-both-50m-sft-from-5m-seed123-3k.toml`のSHA-256は`7932e93671c901caac1fcdadfa58f80c48faa3d158f0e22a5d5109c9dbfadba3`です。

再現コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-50m-sft-from-5m-seed123-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-50m-sft-from-5m-seed123-3k \
  --samples-dir artifacts/samples/issue1-both-50m-sft-from-5m-seed123-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

Colab T4の割り当てを開始前に試し、HTTP 503などで失敗した場合は応答とsession状態を記録してMPSへ切り替えます。成功条件は3,000 stepを完走し、100 stepごとのmetricsと生成文、500 stepごとのcheckpoint metadata、summary、5領域評価、固定chat-test 48例、人手レビュー用JSONを保存することです。

Colab送信用bundleは`/tmp/small_llm-colab-078-XXXXXX.tar.gz`、236,435,549 bytes、SHA-256は`cc3d964406f04280469f690052b43459865506bc175ff9ccfd417e31b8305bea`です。bundleには078の実行コード、設定、075 best checkpoint、加工済みSFTデータ、rehearsal Token列、Tokenizerだけを含め、元JSONL、医師国家試験原本、`medilink_analysis`は含めていません。

## 実験中の記録

ここにはColab試行、bundle hash、MPSへの切り替え、step 1・500・1,000・1,500・2,000・2,500・3,000のloss・PPL・経過時間・学習率・固定prompt生成を時系列で追記します。警告、失敗、悪い生成も削除せず残します。

2026年9月6日07:59台に`colab sessions`を実行し、`No active sessions found on server.`を確認しました。その後、`colab new --session exp078-both-50m-sft-seed123 --gpu T4`を実行しましたが、assignment endpointがHTTP 503 `Service Unavailable`を返しました。Colabセッションは作成されず、bundle upload、入力hash検証、モデル初期化、学習stepは発生していません。078も同一条件をMPSへ切り替えます。

同日08:01台に予定したMPSコマンドで学習を開始しました。step 1はtrain loss 4.245380、SFT loss 4.537371、rehearsal loss 3.077420、validation loss 4.342219、PPL 76.8780、learning rate 5e-7、経過時間4.11秒でした。step 100はtrain loss 3.526594、SFT loss 3.504326、rehearsal loss 3.615668、validation loss 3.824654、PPL 45.8169、learning rate 5e-5、経過時間64.54秒でした。step 200はvalidation loss 3.800898、PPL 44.7413、learning rate 4.9871e-5、経過時間207.87秒、step 300はtrain loss 3.686939、SFT loss 3.650456、rehearsal loss 3.832873、validation loss 3.813421、PPL 45.3052、learning rate 4.9479e-5、経過時間327.62秒でした。step 400はvalidation loss 3.766543、PPL 43.2304、learning rate 4.8830e-5、経過時間450.85秒でした。

step 500はtrain loss 3.344829、SFT loss 3.413326、rehearsal loss 3.070840、validation loss 3.762769、PPL 43.0675、learning rate 4.7931e-5、経過時間582.20秒でした。step 500の固定生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> よろしくお願いします!`で、077のstep 500とは異なるseed依存の出力になっています。step 0〜500の生成本文、step 500のcheckpoint metadata、metricsを保存し、GitHubへpushします。NaN、OOM、shape error、警告はありません。

## 実験終了後の結果と解釈

実際のbackend、最良checkpoint、学習時間、5領域loss、固定chat-testのEOS・長さ・precision・recall・F1、長さ別集計、077との差、seed分散への解釈を追記します。評価結果と生成全文のSHA-256も残します。

## 次に試すこと

seed 123の標準条件を保存した後、同じseed・同じbaseでlong-response ratio 0.25を実行し、長文比率のpaired comparisonを完成させます。その結果を踏まえ、context length 512、データ量追加、蒸留または現代的なinstruction tuningのどれを先に進めるか決めます。
