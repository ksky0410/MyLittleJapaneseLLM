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

同日、MPSで学習を開始しました。開始時の実測parameter数は50,207,616、PyTorch 2.14.0、AMP無効でした。step 1はtrain loss 8.895693、general validation loss 8.819042、PPL 6761.78、learning rate 1.0e-6、経過4.96秒でした。step 100はtrain loss 6.729032、general validation loss 7.089904、PPL 1199.79、learning rate 1.0e-4、経過60.74秒でした。NaN、OOM、shape error、警告は発生しておらず、学習を継続します。step 0とstep 100の生成文、metrics、step 100 metadataを保存しました。

step 200はtrain loss 6.332452、general validation loss 6.819515、PPL 915.54、経過174.55秒、step 300は5.761728、6.478393、PPL 650.92、経過279.86秒、step 400は5.644550、6.151804、PPL 469.56、経過403.54秒でした。step 500ではtrain loss 5.254088、general validation loss 5.943526、PPL 381.28、learning rate 2.9459e-4、経過529.81秒となり、step 100からvalidation lossが1.146378低下しました。step 500のcheckpoint metadataにはweights SHA-256 `c9f6661ccb29c390346342024e2af1c24c5dd6c139385ce636b7fa9e022a29fa`を記録しています。固定promptの生成はstep 0の漢字断片中心の出力から、step 500では数字・記号の長い列へ変化しましたが、自然な返答ではありません。ここまでNaN、OOM、shape error、警告はありません。学習を継続します。

step 600はtrain loss 4.860228、general validation loss 5.724469、PPL 306.27、経過654.43秒、step 700は4.535986、5.590585、PPL 267.89、経過779.78秒、step 800は4.461191、5.477937、PPL 239.35、経過907.17秒、step 900は4.317969、5.388396、PPL 218.85、経過1033.59秒でした。step 1,000ではtrain loss 4.366850、general validation loss 5.297207、PPL 199.78、learning rate 2.3815e-4、経過1165.51秒となりました。step 500からvalidation lossは0.646319低下し、step 1,000のcheckpoint weights SHA-256は`47bf32482a4d6552783c17e00faaeaecbc1d5bcd8a93672b9c77bf69b8ec5666`です。固定promptは、step 500の記号列から、step 1,000では助詞や文末表現を含む長い日本語風の断片へ変わりましたが、「今日なにしてた？」への一貫した返答にはなっていません。ここまでNaN、OOM、shape error、警告はありません。学習を継続します。

step 1,100はtrain loss 3.920229、general validation loss 5.250916、PPL 190.74、経過1298.73秒、step 1,200は3.816042、5.128983、PPL 168.85、経過1420.77秒、step 1,300は3.978298、5.102706、PPL 164.47、経過1540.99秒、step 1,400は3.664692、5.013380、PPL 150.41、経過1661.24秒でした。step 1,500ではtrain loss 3.635811、general validation loss 4.942713、PPL 140.15、learning rate 1.4598e-4、経過1790.18秒となりました。step 1,000からvalidation lossは0.354494低下し、step 1,500のcheckpoint weights SHA-256は`dcded2e301a983d314a8331b4cf72cd651af8a2d6fc500a855a58fc7d627222e`です。固定promptの生成は、step 1,000の断片からstep 1,500で「ニュース」「システム」などの語を含む記事風の連続へ変わりましたが、質問への会話応答にはなっていません。前回073の同じ50M構造・約1M Token条件ではstep 1,300以降にvalidationが悪化したため、現時点では約500万Token条件の方が過学習を遅らせている可能性があります。ここまでNaN、OOM、shape error、警告はありません。学習を継続します。

step 1,600はtrain loss 3.962660、general validation loss 4.914219、PPL 136.21、経過1923.12秒、step 1,700は3.697206、4.889252、PPL 132.85、経過2052.33秒、step 1,800は3.518451、4.849867、PPL 127.72、経過2181.83秒、step 1,900は3.627597、4.798238、PPL 121.30、経過2312.05秒でした。step 2,000ではtrain loss 3.981688、general validation loss 4.775174、PPL 118.53、learning rate 6.3100e-5、経過2444.92秒となりました。step 1,500からvalidation lossは0.167540低下し、step 2,000のcheckpoint weights SHA-256は`9b7846c34d0711ded3867505cbf9aa0ae5911c8d580e36e60ac3ed1e6a7f16df`です。固定promptの出力は、日本語の単語・助詞・敬語風の連続が増えましたが、「お金」「お客様」の反復が強く、質問への適切な会話応答にはなっていません。ここまでNaN、OOM、shape error、警告はありません。学習を継続します。

step 2,100はtrain loss 3.520096、general validation loss 4.746451、PPL 115.17、経過2577.62秒、step 2,200は3.713841、4.736478、PPL 114.03、経過2711.09秒、step 2,300は3.483684、4.719119、PPL 112.07、経過2840.55秒、step 2,400は3.440668、4.707030、PPL 110.72、経過2971.79秒でした。step 2,500ではtrain loss 3.295709、general validation loss 4.689170、PPL 108.76、learning rate 3.0000e-5、step処理時間3099.86秒、summary上の総時間3105.20秒で完走しました。step 2,000からvalidation lossは0.086004低下し、best checkpointはstep 2,500です。step 2,500のweights SHA-256は`f1e5654851c971fcb5435a551fc288be3157f25dc0219b14280dcd2def681a83`、`best.pt`のSHA-256は`71931b2c689c2fbaa31c8c92c022a21fac571894ec2993a59be48644794e5e17`です。固定promptは日本語風の連続と「お客様」「お金」などの反復が増えましたが、会話応答としては未成立です。NaN、OOM、shape error、途中停止、警告はありませんでした。

学習完走時点では、前回073の約1M Token条件のbest validation loss 6.228799に対して、本実験は4.689170で1.539629低下しました。学習中のvalidation曲線も、073がstep 1,300以降に悪化したのに対し、本実験はstep 2,500まで改善を続けました。このため「50Mモデルに約1M Tokenではデータ不足が強く、約5M Tokenへ増やすと少なくともgeneral language modelingは改善する」という仮説は支持されます。ただし、同じ学習Token列ではなく混合データの構成も変わっているため、約5倍のToken数だけの効果とは断定せず、領域別評価と固定chat-testを実行してから結論を確定します。次にbest checkpointをconversation・medical・RPC・MRMPを含むdomain評価へ回し、生成サンプルはGitHubへ追加します。

## 実験終了後の結果と解釈

実験終了直後に、実際のruntime、最良・最終loss、PPL、学習時間、最大メモリまたは未計測の理由、best checkpointのhash、領域別評価、chat-test、固定promptの代表的な生成例、073との差、仮説との一致・不一致、次に変える条件を追記します。自動評価だけで自然さを断定せず、人手レビューが未実施ならその状態を明記します。

## 次に試すこと

本実験でデータ量の効果を確認した後、同じ50M best checkpointへのresponse-only SFTを比較します。validationがまだ改善しない場合は、50M構造のままさらに10M Tokenへ増やすか、学習率・weight decayを一つだけ変更します。改善した場合も、会話SFTの前後を分けて、pretrainingとinstruction tuningの効果を混同しないようにします。
