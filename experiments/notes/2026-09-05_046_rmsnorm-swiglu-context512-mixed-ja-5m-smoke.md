# 実験046：RMSNorm・SwiGLU・RoPE・context 512の日本語5M smoke

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験045でGELUからSwiGLUへ変更したLayerNormモデルが、042のGELU条件より低いvalidation lossを示しました。実験046では、そのSwiGLUを固定したままLayerNormだけをRMSNormへ変更し、現代的なdecoder-only Transformerでよく使われるRMSNormとSwiGLUの組み合わせを確認します。目的は、正規化方式の差が学習安定性、validation loss、生成結果、実行時間へ与える影響を、FFN差と混同せずに記録することです。

仮説は、RMSNorm・SwiGLUの組み合わせが045のLayerNorm・SwiGLUと同程度か、わずかに低いvalidation lossになる可能性がある、というものです。RMSNormは平均を引かずscaleだけを持つため、parameter数と計算量を少し減らせます。一方、5M級・500 stepの短い実験では差がsamplingや初期化の揺らぎに埋もれる可能性があります。固定会話promptでは、lossが低くてもEOS停止や日本語の自然さが改善しない可能性をあらかじめ想定します。

045との差分は`model.norm_type = "rmsnorm"`だけです。SwiGLU、RoPE、dim 240、6層、6 heads、context length 512、MLP倍率4、batch size 8、最大500 step、評価・生成間隔100、AdamW、学習率3e-4から3e-5、warmup 300、weight decay 0.1、seed 42、学習・検証Token列、Tokenizerを固定します。SwiGLUの中間次元は640です。比較用の概算parameter数は5,133,360で、biasを含む実際のparameter数は5,142,480となる見込みです。045の実際の5,145,600より3,120少なくなります。

## 使用するデータ、Tokenizer、コード

学習Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`で、1,336,619 Token、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`で、11,780 Token、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。既存の一般・会話・医療混合コーパスから作ったToken列を読み取り専用で使用し、元の医師国家試験データや`medilink_analysis`の原本には変更を加えません。

TokenizerはSentencePiece Unigram、語彙数4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。設定は`configs/rmsnorm-swiglu-context512-mixed-ja-5m-smoke.toml`です。RMSNormとSwiGLUはcheckpointのmodel signatureに含め、既存のLayerNorm・GELU checkpointを誤ってロードしないようにします。過去metadataにこの設定がない場合はGELUとLayerNormを既定値として解釈する互換性を維持します。

予定している学習コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py \
  --config configs/rmsnorm-swiglu-context512-mixed-ja-5m-smoke.toml
```

学習後は`evaluate_domains.py`でgeneral・conversation・medical・FineWeb2 Edu Japanese・Wikipediaのvalidation lossを測定し、`evaluate_chat_prompts.py`でIssue #1の固定会話promptを評価します。checkpointをロードして`今日は`、`吾輩は`、会話marker、`問題：`を再生成します。stepごとの生成文、崩れた出力、空の出力、評価JSON、checkpoint metadataは削除せず保存します。

## 成功条件と判定方法

500 stepをNaN、Metalエラー、OOM、Token列不足なく完走し、metrics、checkpoint metadata、stepごとの生成TXTが保存されれば実装上の成功とします。checkpoint reloadと固定prompt生成が成功することも確認します。品質比較は045のLayerNorm・SwiGLU条件に対して行い、general・conversation・medical loss、perplexity、EOS到達数、空completion数、平均生成Token数を並べます。lossが低くても生成が崩れていれば、next-token loss上の差と会話品質を分けて解釈します。

## 実験中の記録

2026-09-05 13:04 JST、学習開始前に設定・入力Token列・TokenizerのSHA-256と現在のGit commitを確認しました。実行環境はPython 3.13.1、macOS 15.5 arm64、MLXのdeviceは`Device(gpu, 0)`です。準備commitは`c5a2d8d`、設定SHA-256は`e4c9144c634c4513746d5f63cac2e92bd408862f6272c65ead606b5d4b14db5d`です。学習中は100 stepごとにtrain loss、validation loss、perplexity、生成文、所要時間を保存します。Metal deviceが見えず実行できない場合も失敗として記録し、CPUや別backendで実行した場合は主実験と混ぜません。

13:04 JSTごろにMLX/Metalで学習を開始し、500 stepまで完走しました。NaN、OOM、Token列不足、Metalエラー、checkpoint reloadエラーは発生しませんでした。step 1、100、200、300、400、500のvalidation lossは順に8.8376019796、6.7399013837、6.2076776822、5.7528589567、5.4054004351、5.2747049332でした。

## 結果と解釈

MLX/Metalでの学習は500 stepまで正常に完走し、所要時間は185.720秒でした。最良checkpointはstep 500で、重みは`artifacts/checkpoints/rmsnorm-swiglu-context512-mixed-ja-5m-smoke/step_000500.npz`、SHA-256は`0674835fabc5b1b7d5f8a3d6637997b09509bd750cf2d2b72dbf5efc73fde05c`です。実際の学習parameter数は5,142,480、概算関数による比較用の値は5,133,360です。step 500 metadataのSHA-256は`5fb9754534768738453b13995ae4c91f677d38b481519b940e8cdb7182b472af`、metricsのSHA-256は`ea365b7b31ceb8735803d26a8f152ce28d8d0defb8b4c1de24ddf34396535058`、summaryのSHA-256は`caad4cc3e79181bcd88f0177b46227c7cb2e14b945f36b208376797c913473b2`です。

general validation lossは5.2747049332、perplexityは195.3328317683でした。domain評価ではconversationが3.3796505133（perplexity 29.3605082138）、medicalが3.9627672036（52.6026870820）、FineWeb2 Edu Japaneseが5.9117492040（369.3516620880）、Wikipediaが6.5071889559（669.9405358227）でした。評価JSONは`artifacts/evaluations/rmsnorm-swiglu-context512-mixed-ja-5m-smoke-domains.json`に保存し、SHA-256は`031ea619b4e5149cc9591d2792d07d0469d891be44bf2fca206fe61e8c55f608`です。

2×2比較の主なlossは、GELU/LayerNormの042がgeneral 5.3410979907、conversation 3.3903319836、medical 4.0347564220、GELU/RMSNormの043が5.3400775592、3.3905173937、4.0437904994、SwiGLU/LayerNormの045が5.2701706886、3.3716124694、3.9484332403、RMSNorm/SwiGLUの046が5.2747049332、3.3796505133、3.9627672036でした。046は043よりgeneralが0.0653726261、conversationが0.0108668804、medicalが0.0810232957低くなり、SwiGLUの改善は維持しました。一方、045よりgeneralが0.0045342445、conversationが0.0080380440、medicalが0.0143339634高くなりました。RMSNormによってparameter数は3,120減りましたが、今回の単一seed・500 step条件ではSwiGLUへ加える相乗効果は確認できませんでした。FineWebとWikipediaは補助値として記録し、構造差の結論には使いません。

Issue #1の固定会話prompt 8件は`artifacts/evaluations/rmsnorm-swiglu-context512-mixed-ja-5m-smoke-chat.json`と`artifacts/samples/rmsnorm-swiglu-context512-mixed-ja-5m-smoke/chat-issue-1.txt`へ保存しました。JSONのSHA-256は`e730abcfcb218db652a768baeb13ff67881355559a505bc63a07411032c2e564`、TXTのSHA-256は`0b2262921361bb1a7a25ba387b68885c2ad59863449647f63cd6fde09db2f285`です。8件中5件がEOSへ到達し、空completionは2件、平均completion長は68.125 Tokenでした。045は6件EOS、空completion 2件、平均45.0 Tokenでしたので、046では固定会話promptの停止挙動も改善していません。`それな`、`なんかさ`、`いやそれは`では医療問題や英数字を含む長い断片が続き、loss差を会話能力の向上とは解釈できません。

step 500 checkpointをロードした`今日は`、`吾輩は`、会話marker、`問題：`の生成はすべて成功しました。生成文は`reloaded-today.txt`、`reloaded-story.txt`、`reloaded-conversation.txt`、`reloaded-medical.txt`へ保存しています。会話markerでは「よろしくお願いします。」が返りましたが、storyやmedicalでは英数字、選択肢、医療問題の断片が混ざりました。reload生成のSHA-256は、todayが`3832d8bc0b16b0abf74a8e68e751b50777527943f98ff1ee4f783e5972ded953`、storyが`95217eba29a00ecfb796080aaa8f647a24f6e732fbfc0a28c2220532bee71896`、conversationが`817ae4d020b20e4b9538ae09dbc38342b880201ca6f3a0521b128fcfe8836a57`、medicalが`68a79add12b67986bdf5c6666fc736a4907b3a63d97b2fd9d17f8d08e99c091f`です。stepごとの生成文は`step_000000.txt`から`step_000500.txt`まで保存しました。

以上から、実装上の成功条件は満たしましたが、RMSNormをSwiGLUへ追加したことでvalidation lossや会話停止が改善したとは判定しません。043のRMSNorm・GELUよりはlossが下がりましたが、045のLayerNorm・SwiGLUをgeneral・conversation・medicalのすべてで下回りましたので、今回の小型・単一seed・500 step条件ではSwiGLU単独を採用候補とします。RMSNormはより長い学習Token予算または複数seedで再検証する対象とし、医学的能力や実用的な会話能力は主張しません。次回はSwiGLU・LayerNormを固定して20M級へ拡大し、学習Token数を増やしても今回のloss差が残るかを調べます。

## 次に試すこと

次はSwiGLU・LayerNorm・RoPEを固定して10Mから20M級へ拡大し、既存の混合Token列を増量してもSwiGLUのloss改善が残るかを確認します。Colab T4が確保できる場合はPyTorchで20M級を実行し、確保できない場合はMacBook MLXで10M級の短い予備実験を行います。データ追加を行う場合は、新しいdatasetの出所・revision・ライセンス・取得日時・hash・混合比率を先に記録し、元データを変更せずに加工済みdatasetを別パスへ保存します。
