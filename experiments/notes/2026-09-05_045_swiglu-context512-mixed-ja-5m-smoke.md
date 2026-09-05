# 実験045：SwiGLU・RoPE・context 512の日本語5M smoke

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験042のLayerNorm・RoPE・context 512条件を基準にし、実験045ではFFNだけをGELUからSwiGLUへ変更します。目的は、現代的なdecoder-only Transformerで広く使われるSwiGLUをこの日本語小型モデルへ導入したとき、同程度のparameter予算で学習の安定性、validation loss、生成結果、実行時間がどう変わるかを確認することです。

事前の仮説は、SwiGLUのgateによって表現力が増し、同じ学習条件でvalidation lossがGELU条件と同程度か少し低くなる可能性がある、というものです。ただし、500 stepの短いsmokeであり、モデルも約5M parameterにすぎないため、差がほとんど見えないか、初期化とsamplingの揺らぎに埋もれる可能性も高いと考えます。SwiGLUが常に優れるとは仮定せず、まず実装とcheckpoint reloadが成立するかを確認し、042との数値差は探索上の参考として扱います。

今回の042との差分は`model.ffn_type = "swiglu"`だけです。LayerNorm、RoPE、dim 240、6層、6 heads、context length 512、MLP倍率4、batch size 8、最大500 step、評価・生成間隔100、AdamW、学習率3e-4から3e-5、warmup 300、weight decay 0.1、seed 42、学習・検証Token列、Tokenizerを固定します。SwiGLUはgate・up・downの3射影を使うため、中間次元を`2/3 × dim × mlp_ratio`として640にし、GELUのup・downの中間次元960とほぼ同じ重み行列parameter数に揃えます。biasを含む実際のparameter数はGELUが5,143,680、SwiGLUが5,145,600で、SwiGLUが1,920多くなります。RoPEでは位置Embeddingを持たないため、概算parameter数はどちらも5,136,480です。

## 使用するデータ、Tokenizer、コード

学習Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`で、1,336,619 Token、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`で、11,780 Token、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。既存の一般・会話・医療混合コーパスから作ったToken列を読み取り専用で使用し、元の医師国家試験データや`medilink_analysis`の原本には変更を加えません。

TokenizerはSentencePiece Unigram、語彙数4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。設定は`configs/swiglu-context512-mixed-ja-5m-smoke.toml`です。SwiGLUをMLXとPyTorchの両方へ実装し、`ffn_type`をcheckpointのmodel signatureへ含めます。既存設定は既定値`gelu`として解釈し、過去のGELU checkpointはmetadataにこの項目がなくても読み込めるよう互換性を維持します。

予定している学習コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py \
  --config configs/swiglu-context512-mixed-ja-5m-smoke.toml
```

学習後は`evaluate_domains.py`でgeneral・conversation・medicalに加え、FineWeb2 Edu JapaneseとWikipediaのvalidation lossを測定し、`evaluate_chat_prompts.py`でIssue #1の固定会話promptを評価します。checkpointをロードして固定promptも再生成します。stepごとの生成文、崩れた出力、空の出力、評価JSON、checkpoint metadataは削除せず保存します。

## 成功条件と判定方法

500 stepをNaN、Metalエラー、OOM、Token列不足なく完走し、metrics、checkpoint metadata、stepごとの生成TXTが保存されれば実装上の成功とします。checkpoint reloadと固定prompt生成が成功することも確認します。品質については042のLayerNorm・RoPE・context 512条件とgeneral、conversation、medicalのvalidation loss、固定会話promptの停止状況を比較します。lossが低ければこの条件での改善候補、差が小さければ同等、明確に高ければ悪化と判定しますが、一回のsmokeだけでSwiGLU全体の優劣は結論づけません。

## 実験中の記録

2026-09-05 12:56 JST、学習開始前に設定・入力Token列・TokenizerのSHA-256と現在のGit commitを確認しました。実行環境はPython 3.13.1、macOS 15.5 arm64、MLXのdeviceは`Device(gpu, 0)`です。コードcommitは`540ba90`、設定のSHA-256は`3ba2fe173d9745e41d72f3142ecf98c3f9e001ed961adc04236a6569dffb0c31`です。学習中は100 stepごとにtrain loss、validation loss、perplexity、生成文、所要時間を保存します。Metal deviceが見えず実行できない場合も失敗として記録し、CPUや別backendで実行した場合は主実験と混ぜません。

## 結果と解釈

2026-09-05 12:56 JSTごろにMLX/Metalで学習を開始し、500 stepまで正常に完走しました。NaN、Metalエラー、OOM、Token列不足、checkpoint reloadエラーは発生しませんでした。step 1、100、200、300、400、500のvalidation lossは順に8.8384100596、6.7402544022、6.2039810816、5.7415204048、5.3970890045、5.2701706886でした。最終train lossは4.3003616333、最良checkpointはstep 500、general validation perplexityは194.4491498718、学習時間は174.69秒でした。

SwiGLUの重み行列parameter予算をGELUと近づけるため、中間次元を640としました。biasを含む実際のparameter数は5,145,600で、GELUの042の5,143,680より1,920多く、差は約0.037%です。概算parameter数はどちらも5,136,480ですが、厳密な比較ではこのbias差を考慮します。

実験042のGELU・LayerNorm・RoPE・context 512・500 step条件はgeneral loss 5.3410979907、conversation loss 3.3903319836、medical loss 4.0347564220でした。045のSwiGLUはそれぞれ5.2701706886、3.3716124694、3.9484332403となり、042からgeneralは0.0709273020、conversationは0.0187195142、medicalは0.0863231818低下しました。同じseed、Token列、Tokenizer、モデル幅、層数、context、学習step、MLX/Metal backendで、主な構造差をFFNへ限定できています。ただし初期化の乱数列はgate追加によって完全には同一ではなく、1回のsmokeのみでSwiGLUの一般的な優位性を証明したとは扱いません。

追加したfinewebとwikipediaのdomain lossはそれぞれ5.9042464892と6.5104347865でした。詳細は`artifacts/evaluations/swiglu-context512-mixed-ja-5m-smoke-domains.json`へ保存し、SHA-256は`67178074061294c5f21c1971a0f226b3170d9939b252092c236ef643611a9ea6`です。finewebとwikipediaは043のRMSNorm条件でも測定していますが、正規化方式が異なるため、ここでは主比較の補助値としてのみ扱います。

学習中の固定prompt生成は`artifacts/samples/swiglu-context512-mixed-ja-5m-smoke/step_000000.txt`から`step_000500.txt`まで保存しました。step 100では短い「ま?」、step 200では日本語らしい助詞を含む断片、step 300と400では医療問題形式の断片、step 500では「今日はにおられられるからないからます。」という短く崩れた文が出ました。学習lossは改善しても、自然な日本語や医学的な正確性は確認できません。

Issue #1の固定会話prompt 8件は`artifacts/evaluations/swiglu-context512-mixed-ja-5m-smoke-chat.json`と`artifacts/samples/swiglu-context512-mixed-ja-5m-smoke/chat-issue-1.txt`へ保存しました。8件中6件がEOSへ到達し、空completionは2件、平均completion長は45.0 Tokenでした。`それな`には「に分っきかけりになかんときょに思ひたい。」が返りましたが、`まじで`と`いやそれは`では英数字や医学問題の断片が160 Token続きました。042は8件すべてEOS、空completion 2件、平均4.75 Tokenでしたので、045のvalidation loss改善と固定会話promptの停止挙動改善は切り分けて扱います。chat JSONのSHA-256は`ccf325b04390e9bfd8aeae8cfc3ec4c77358c03648693a0a0335928632c5e227`、可読TXTのSHA-256は`63021275ae2874f946e24122ca5b8f7ab516fe23dfec32d414dd9a6aba462d80`です。

step 500 checkpointをロードした`今日は`、`吾輩は`、会話marker、`問題：`の生成はすべて成功しました。会話markerでは「こんにちは。」、story promptでは短い日本語らしい断片、todayとmedicalでは医療問題や数値の崩れた断片が出ました。reload生成は`artifacts/samples/swiglu-context512-mixed-ja-5m-smoke/reloaded-today.txt`、`reloaded-story.txt`、`reloaded-conversation.txt`、`reloaded-medical.txt`に保存しました。checkpointのSHA-256は`1e2382c102124379e516539b57dce0c496735350ddc5fa7d42eb9b4b5e8ea625`、metricsは`314e3f01f2284e760800c6e8c6554983072c05fac88e5795f0bc9dcbe255c3b3`、summaryは`4b7cf198acef2fab73cc3ffb27ad7fcc9baa62368b706bf504d7797dbf4f23a6`、step 500 metadataは`1786421be710bb71b79cec539464a628d65ed09594a25190f6bc1ac777c7b998`です。reload生成のSHA-256は順に、`fd23d9ade07b123c810294dd098d986aba29c443da3084f378fb8fdbdf94a095`、`623f6845ed8e5a8b5038edc07bd8ab0b21b77230a4fb69fb72f20a48921cbe3b`、`2a7d1208de91fd084fd965ebdb12ea421507ba7bb4bb83161e012bd8d8ab5e9a`、`e1d6814d6cbb33177e5fa9d6a708d69e06787b44787b8509e285d3bde0fba374`です。

以上から、SwiGLUは今回の5M・500 step・context 512条件で、general、conversation、medicalのvalidation lossをGELUより低下させましたので、次の候補として採用します。ただし、会話promptはEOS停止率と生成長が悪化し、出力内容も医学問題の断片や英数字の崩れを含みます。したがって「言語モデルのnext-token lossに対して有望」と判定し、「会話能力や医学能力が向上した」とは判定しません。SwiGLU実装、parameter予算の記録、checkpoint reload、全生成物の保存には成功しました。

## 次に試すこと

次は、同じSwiGLU・RoPE・context 512・seed 42でRMSNormへ切り替え、SwiGLUを固定したLayerNorm対RMSNormの比較を行います。その後、今回の構造を10Mから20M級へ拡大し、学習Token予算を増やしたときにもloss改善が残るかを確認します。データ追加を行う場合は、新しいdatasetの出所・revision・ライセンス・取得日時・hash・混合比率を先に記録し、元データを変更せずに加工済みdatasetを別パスへ保存します。
