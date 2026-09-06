# 実験080：50M日本語モデルの10M Token事前学習

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。実験079で長文応答のoversamplingは全体の会話性能を改善しなかったため、今回はSFTの細かな比率ではなく、事前学習のデータ量を増やすことを試します。強い教師モデルによる蒸留は行わず、同じ50Mモデルを日本語コーパスだけから学習します。

実験075では、同じ50M構造を約5M Token、2,500 stepで事前学習しました。実験080では、同じTokenizer・モデル構造・seed・optimizerを保ち、FineWeb2 Edu Japanese、青空文庫、Wikipedia、日本語会話、医師国家試験由来データを混ぜた約10M Token列を使い、5,000 stepまで学習します。batch size 8、context length 256なので、学習で見るToken数はおよそ10.24Mです。075の約5.12M Tokenに対して、学習予算をほぼ2倍にする比較です。

仮説は、学習Token数を約2倍にすると、generalだけでなくconversation、medical、RPC、MRMPのvalidation lossが改善し、固定chat-testの応答がより長く自然になることです。ただし、10M列ではWikipediaの比率が増え、075の5M列とsource比率が完全には一致しません。したがって結果を「Token数だけの因果効果」と断定せず、データ量増加とsource構成変更を合わせた実用的な主線候補として扱います。改善しなければ、次は同一5M列を複数周回した条件を別実験にして、データの多様性と反復学習を分離します。

成功条件は、5,000 stepを完走し、100 step間隔のmetricsと生成文、500 step間隔のcheckpoint metadata、最良checkpoint、summary、5領域評価、固定chat-test 48例を保存することです。validation lossだけでなく、EOS到達率、平均生成長、全体および長さ別のToken overlap F1、生成文の日本語としての自然さを確認します。

## 再現条件

モデルはRoPE・LayerNorm・SwiGLU、dim 576、12層、9 heads、context length 256、MLP倍率4、50M級の構造です。TokenizerはSentencePiece Unigram、vocab 4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`です。学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1-train.bin`、9,999,973 Token、SHA-256は`d043d06180d2c6deb0e0c14038fd1b3f736f86f062cf61260bd19282f8ce48e4`です。general validationは`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。元の`medilink_analysis`と医師国家試験原データは変更せず、small_llm側の加工済みToken列だけを読み取ります。

10Mコーパスのmanifestは`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.manifest.json`です。Token比率はaozora 5.08%、fineweb 42.18%、wikipedia 42.19%、conversation 5.27%、medical 5.27%です。075の5M列はaozora 10.17%、fineweb 71.87%、conversation 8.98%、medical 8.98%だったため、この差を結果の解釈に明記します。

設定ファイルは`configs/issue1-both-50m-pretrain-10m-5k.toml`です。学習条件はbatch size 8、最大5,000 step、eval/sample interval 100、checkpoint interval 500、eval batches 20、learning rate 3e-4から3e-5、warmup 500、weight decay 0.1、seed 42です。設定ファイルのSHA-256は学習開始前に計算して追記します。

開始前に確認したSHA-256は、設定ファイルが`1f3570dd38e13286e9dc3270e68f2b5f803dd8aa44410f2ce85abbede56b9447`、10M corpus manifestが`f9d17b36998671320ab69d7448fde10a7be0c2ba894ae0daa00fc437ac3e2c64`、Tokenizerが`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。新規スクリプトは`py_compile`を通過しました。

ローカルでの再現コマンドは次のとおりです。

```bash
uv run python scripts/train_torch.py \
  --config configs/issue1-both-50m-pretrain-10m-5k.toml \
  --device mps
```

Colab T4が利用できる場合は、同じ設定と入力を`colab_bootstrap_080.py`から実行します。Colab失敗時はHTTP status、session状態、bundle hashを残してMPSへ切り替えます。

開始時点で`colab sessions`は`No active sessions found on server.`でした。Colab送信用bundleは`/tmp/small_llm-colab-080.tar.gz`、12MB、SHA-256は`ec8d498e1956083df20333b094aba73b333fb32657c04987f9a7f7a8a51552c5`です。bundleには080の設定、学習スクリプト、srcパッケージ、Tokenizer、10M Token列、general validationだけを含め、元JSONL、医師国家試験原本、`medilink_analysis`は含めていません。

## 実験中の記録

学習開始前に設定・入力hash・Git commit・bundle hash・Colab試行結果を追記します。学習中は1,000 stepを超えて記録を空けず、原則100 stepごとにmetrics、生成文、異常を保存します。生成文は良いものだけでなく、崩れた出力も削除せずGitHubへ追加します。

Colab T4の新規session `exp080-both-50m-pretrain-10m`は作成に成功し、bundle uploadと入力hash検証も完了しました。Colab側のTorch/CUDA情報は学習完了後のsummaryから確定します。標準出力の転送は遅れましたが、リモートの`metrics.jsonl`を途中回収して学習継続を確認しています。step 1はtrain loss 8.790941、validation loss 8.819555、PPL 6765.26、learning rate 6.0e-7、経過3.48秒でした。step 100はvalidation loss 7.276060、PPL 1445.28、learning rate 6.0e-5、経過12.51秒、step 500はtrain loss 5.497313、validation loss 6.379793、PPL 589.81、learning rate 3.0e-4、経過56.60秒でした。

step 1,000はtrain loss 4.444293、validation loss 5.593394、PPL 268.65、learning rate 2.9189e-4、経過109.36秒でした。step 1,500はtrain loss 4.536695、validation loss 5.346462、PPL 209.86、learning rate 2.6848e-4、経過166.33秒、step 2,000はtrain loss 3.786160、validation loss 5.110367、PPL 165.73、learning rate 2.3258e-4、経過220.45秒でした。step 2,500はtrain loss 3.421191、validation loss 4.980713、PPL 145.58、learning rate 1.8854e-4、経過278.24秒、step 3,000はtrain loss 3.565266、validation loss 4.889075、PPL 132.83、learning rate 1.4165e-4、経過330.51秒でした。step 3,100はtrain loss 3.711011、validation loss 4.839065、PPL 126.35、learning rate 1.3243e-4、経過343.84秒でした。step 3,100時点でNaN、OOM、shape errorはなく、validation lossは継続的に改善しています。学習は継続中です。

## 実験終了後の結果と解釈

実際のbackend、完走または停止理由、最良checkpoint、学習時間、5領域loss、chat-test結果、075との差、source比率の影響、自然な日本語の質的観察を追記します。失敗した場合も、原因不明ならそのまま明記し、次の切り分けを残します。

## 次に試すこと

080で改善が確認できた場合は、同じ10M列でSFTへ進める前に、さらに学習Token数を増やすか、context length 512へ伸ばすかを比較します。改善が限定的な場合は、同じ5M列を複数周回する条件を実施し、データの多様性と反復学習の効果を分離します。どちらの場合も蒸留は主線へ入れず、教師なしの日本語事前学習で自然さがどこまで伸びるかを優先します。
