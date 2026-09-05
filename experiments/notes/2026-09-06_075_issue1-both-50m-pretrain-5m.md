# 実験075：Issue #1混合データを約5M Tokenへ増やした50M事前学習

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。GitHubの[Issue #1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)で候補になっている現代的な会話データと、医師国家試験由来の加工済みデータを、一般日本語と一緒に使う方針を引き継ぎます。モデルを医療専用にはせず、一般・会話・医療の領域別validationを分けて確認します。元の`/Users/koseki/projects/medilink_analysis`とその原データは変更・削除しません。

実験073では、50Mモデルを約100万Tokenの`issue1-both-1m-fineweb`へ2,500 step学習しました。20M基盤と比較して5領域のvalidation lossは改善せず、50M側でデータ不足または過学習が早く現れた可能性がありました。今回はモデル構造、Tokenizer、seed、optimizer、学習stepを固定し、学習Token列だけを約100万から約500万へ変更します。1 stepあたりの露出量はbatch 8 × context 256 = 2,048 Tokenであり、2,500 stepでは約512万Tokenです。今回の約500万Token列をほぼ一周するため、073の約5周相当と比較できます。

仮説は、50Mモデルへ十分な異なる日本語データを与えると、073よりgeneral・conversation・medical・RPC・MRMPのvalidation lossが下がり、データ不足による早い過学習が緩和されることです。反対に、validationが改善しない場合は、今回のモデル容量に対してさらに大きなデータ量、学習率や正則化、またはデータの品質・分布がボトルネックと判断します。固定chat-testの自動F1は短い相づちを過大評価する可能性があるため、領域別lossと生成本文を合わせて解釈し、自然な会話能力の向上を単独では主張しません。

## 使用するデータと再現条件

設定は[`configs/issue1-both-50m-pretrain-5m-2p5k.toml`](../../configs/issue1-both-50m-pretrain-5m-2p5k.toml)です。モデルはvocab 4,096、dim 576、12層、9 heads、context length 256、RoPE、LayerNorm、SwiGLUで、073と同じ実測約50.2M parametersを使います。学習条件はbatch size 8、最大2,500 step、evaluation・生成間隔100 step、checkpoint間隔500 step、evaluation batches 20、AdamW、learning rate 3e-4から3e-5、warmup 300、weight decay 0.1、seed 42です。MacBookではMPSかつAMP無効、ColabではT4などが割り当てられた場合にPyTorch CUDAを使います。

学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-5m-v1-train.bin`で、4,999,958 Token、ファイルサイズ19,999,832 bytes、SHA-256は`54eb3fab617c94bda59899db4f78e6ac65665606219414a710e40cc8ccb8603c`です。混合manifestは`artifacts/corpus/mixed-ja-token-budget-fineweb2-5m-v1.manifest.json`で、実際のToken比率は青空文庫約10.17%、FineWeb2 Edu Japanese約71.87%、会話約8.98%、医療約8.98%です。会話sourceにはIssue #1に関連する加工済み会話データを含み、医療sourceには`medilink_analysis`からコピーして加工済みの医師国家試験データを含みます。元データは読み取り専用で扱います。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。general validationは`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、SHA-256 `c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。完走後は同じcheckpointをconversation、medical、RPC、MRMP validationと固定chat-test-v1で評価します。

実行予定コマンドは次のとおりです。

```bash
uv run python scripts/train_torch.py \
  --config configs/issue1-both-50m-pretrain-5m-2p5k.toml \
  --device mps
```

先にColab CLIで`colab new -s exp075-both-50m-pretrain-5m --gpu T4`を試します。bundleにはコード、設定、Tokenizer、加工済みToken列、general validationだけを含め、元JSONL、医師国家試験原本、`medilink_analysis`のディレクトリは含めません。Colabが503、割当上限、認証、uploadなどで失敗した場合は、そのエラーとsession状態を残して同一条件のMPSへ切り替えます。

開始commitは`4824b09`で、GitHubの`origin/main`へpush済みです。設定ファイルのSHA-256は`74afeb79e08c1f96fa0954ccb1921fca767457536ab965170804f140151a61b4`、実験ノートのSHA-256は`e114706285e6897c74fbcd58fa1cb08d7759c7f55f77a865f6c57d4177ba9d50`です。Colab送信用bundleは`/tmp/small_llm-colab-075-XXXXXX.tar.gz`、6,989,615 bytes、SHA-256 `92c0862c64ca327b0b0e6930595af3afdc87b4be2c63c2fcd9cc95053e0d124c`です。bundleには`src`、PyTorch学習コード、075のbootstrapとpackage、設定、Tokenizer、約5M Tokenの学習列、general validationだけを含めています。bundle作成時にPython cacheは除外しました。

## 成功・失敗の判定基準

2,500 stepをNaN、OOM、shape error、Token列不足なく完走し、step 0から2,500まで100 step間隔のmetricsと生成本文、500 step間隔のcheckpoint metadata、summaryを保存できれば学習実験として成功とします。性能面では、073の50M・約1M Token条件に対してgeneralを含む複数領域のvalidation lossが改善するかを確認します。失敗、悪い生成、途中停止、Colabの利用不能は削除せず記録します。

## 実験中の記録

この節には、Colab試行、bundleのbytesとSHA-256、session状態、MPSへの切り替え、開始時の実測parameter数、100 stepごとのmetricsと生成文、500 stepごとのcheckpoint metadata、警告、メモリ問題、途中停止を時系列で追記します。実験開始前の設定変更や、ColabとMPSで実行backendが変わった場合も上書きせず残します。

2026年9月6日、MPS学習の前に`colab sessions`を実行し、`No active sessions found on server.`を確認しました。その後、`colab new --session exp075-both-50m-pretrain-5m --gpu T4`を実行しましたが、assignment endpointがHTTP 503 `Service Unavailable`を返して終了しました。Colab側のbundle upload、入力検証、Python初期化、学習stepは発生していません。今回のColab失敗を成功実験と混ぜず、同じcommit・config・入力・seedでMPSへ切り替えます。

## 実験終了後の結果と解釈

実験終了直後に、実際のruntime、最良・最終loss、PPL、学習時間、最大メモリまたは未計測の理由、best checkpointのhash、領域別評価、chat-test、固定promptの代表的な生成例、073との差、仮説との一致・不一致、次に変える条件を追記します。自動評価だけで自然さを断定せず、人手レビューが未実施ならその状態を明記します。

## 次に試すこと

本実験でデータ量の効果を確認した後、同じ50M best checkpointへのresponse-only SFTを比較します。validationがまだ改善しない場合は、50M構造のままさらに10M Tokenへ増やすか、学習率・weight decayを一つだけ変更します。改善した場合も、会話SFTの前後を分けて、pretrainingとinstruction tuningの効果を混同しないようにします。
