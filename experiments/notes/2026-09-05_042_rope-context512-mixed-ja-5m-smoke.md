# 実験042：RoPEとcontext length 512の日本語5M smoke

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。ColabのT4・L4割当が連続して利用できなかったため、MacBookのMLX/Metalで実行できる範囲の構造実験を先に進めます。実験014では、同じvocab 4,096、dim 240、6層、6 heads、500 stepのRoPEモデルをcontext length 256で学習し、general validation loss 5.5338133176を記録しました。

今回の仮説は、RoPEモデルのcontext lengthだけを256から512へ伸ばすと、長い文脈を一度に扱えるため、general validation lossまたは生成文の文脈維持が改善する可能性があるというものです。一方、学習stepは500のままなので、1 stepあたりの処理Token数が2,048から4,096へ倍増し、同じ計算step数でも学習総Token量が増えます。そのため、結果は「context長の効果」だけでなく「学習Token予算の増加」を含む探索的比較として扱います。長いcontextの利点がまだ現れず、attention計算の増加だけが表れる可能性も記録します。

## 使用するデータ、Tokenizer、モデル

学習Token列は実験014と同じ`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`で、Token数は1,336,619、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。混合manifestは`artifacts/corpus/mixed-ja-80-10-10-v2.manifest.json`で、単位比率は一般80%、会話10%、医療10%、Tokenizer後のToken比率は実験013記録どおり一般36.57%、会話39.69%、医療23.74%です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、11,780 Token、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。元の医師国家試験データと加工済みコーパスは変更せず、既存Token列を読み取り専用で使います。

TokenizerはSentencePiece Unigram、語彙数4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。モデルはRoPE、dim 240、6層、6 heads、context length 512、MLP倍率4です。RoPEのhead dimensionは40で偶数です。位置embeddingを持たないため、実験014のcontext 256モデルより概算parameter数は同じ5,136,480です。batch size 8、最大500 step、evaluation/sample interval 100、evaluation batches 20、AdamW、learning rate 3e-4から3e-5、warmup 300、weight decay 0.1、seed 42を使います。

設定は`configs/rope-context512-mixed-ja-5m-smoke.toml`です。学習前に設定、Token列、TokenizerのhashとGit commitを固定します。予定コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py \
  --config configs/rope-context512-mixed-ja-5m-smoke.toml
```

学習後は、general validation lossを実験014と比較し、context 512で保存したstepごとの生成文を削除せず確認します。生成結果だけで長期記憶や会話能力を主張せず、まず学習が正常に進んだか、長い入力が扱えるか、出力の崩れ方が変わったかを記録します。

## 成功条件

500 stepをNaN、Metalエラー、OOM、Token列不足なく完走し、step 0と100 step以下の間隔でmetrics、checkpoint metadata、生成TXTを保存することです。context length 512のforward、checkpoint reload、固定prompt生成が成功することも確認します。失敗した生成文や途中停止の成果物は削除しません。

## 実行前の再現情報

設定を追加したGit commitは`dfcdb61a5efa3df73b6701c27b360e1db85cc534`、学習開始前に条件を固定したcommitは`596eb38`です。設定SHA-256は`99e6e4335451565db02f270a139f15daa9f23ccb2a637fa24ea8d00c43dc8458`です。学習Token列、validation Token列、TokenizerのSHA-256は前節へ記録済みです。実行環境はPython 3.13.1、macOS 15.5 arm64、MLX import成功、`mx.default_device()`は`Device(gpu, 0)`でした。Metalデバイスが見えていることを確認してから学習を開始しました。

## 実験中の記録

2026-09-05 12:19 JST、開始前に公開commit、設定SHA-256、入力Token列とTokenizerのSHA-256、Metal GPU deviceを確認しました。これから予定コマンドで学習を開始します。

学習完了後の評価は、`scripts/evaluate_domains.py`でgeneral・conversation・medicalの3 domainを同じv2 Token列に対して測定し、`scripts/evaluate_chat_prompts.py`でIssue #1の固定prompt 8件を評価します。reload確認として`generate.py`から`今日は`、`吾輩は`、会話marker付きprompt、`問題：`を生成し、既存stepサンプルと別ファイルへ保存します。評価JSON、可読TXT、reload生成文はすべてGitHubへ追加します。

12:22 JST、500 stepの学習、domain評価、固定chat prompt評価、checkpoint reload後の4種類の生成を完了しました。NaN、Metalエラー、OOM、Token列不足、reloadエラーは発生しておりません。

## 結果と解釈

学習は500 stepまで完走し、所要時間は122.53秒でした。step 500が最良checkpointで、general validation lossは5.3410979907、perplexityは208.7417810357でした。学習中のvalidation lossはstep 1で8.7620855967、step 100で6.6710745494、step 200で6.1675168673、step 300で5.7626544635、step 400で5.4578887622、step 500で5.3410979907と、全記録点で低下しました。step 500のcheckpoint本体は約20MBで、SHA-256は`eadff62449ae195335c66319fdca31a9808c626d88b99dd9642a4ea8de9515f6`です。

実験014のcontext 256 RoPE条件と同じToken列・Tokenizer・モデル幅・層数・seed・最大stepで比較すると、general lossは5.5338133176から0.1927153269低下しました。今回のdomain評価では、conversation lossが3.3903319836、medical lossが4.0347564220となり、context 256の3.5064193408、4.5083060265からそれぞれ0.1160873572、0.4735496044低下しました。

この改善は有望ですが、context 512ではbatch sizeを変えていないため、1 stepあたりの処理Token数が2,048から4,096へ増え、500 stepで約2.048M Tokenを処理しています。context 256条件の約1.024M Tokenに比べて学習Token予算が倍であり、長いcontextそのものの効果と学習量の効果は分離できません。また、attention計算量も増えているため、所要時間はcontext 256の実験014の120.40秒とほぼ同じに見えますが、実行時のMetalコンパイルや環境差を含む参考値です。

固定Issue #1 prompt 8件はすべてEOSへ到達し、平均completion長は4.75 Tokenでした。`まじで`には「かくとくんでます。」、`それな`には「つてねた。」、`おつかれ`には「がっていた。」などを返し、`今日なにしてた？`と`明日ひま？`は空のcompletionでした。context 256条件も生成文の崩れと空応答があったため、今回のEOS停止率だけから会話能力の改善とは解釈しません。評価結果のJSONと全生成文は`artifacts/evaluations/rope-context512-mixed-ja-5m-smoke-chat.json`、`artifacts/samples/rope-context512-mixed-ja-5m-smoke/chat-issue-1.txt`へ保存しました。

checkpointをロードした固定prompt生成も4件すべて成功しました。`今日は`では比較的長い日本語風の続き、`吾輩は`では短い説明風の続き、会話markerでは「こんにちは!」という短い返答、`問題：`ではBase64のように見える崩れた文字列が出ました。これらはすべて削除せず、`artifacts/samples/rope-context512-mixed-ja-5m-smoke/reloaded-*.txt`へ保存しました。特に医療promptが崩れたことから、医師国家試験データを含む混合コーパスを学習していても、医学的な正確性や実用性は示されていないと判断します。

metricsは`artifacts/checkpoints/rope-context512-mixed-ja-5m-smoke/metrics.jsonl`、summaryは`artifacts/checkpoints/rope-context512-mixed-ja-5m-smoke/summary.json`、checkpoint metadataは同ディレクトリのJSONへ保存しました。domain評価JSONのSHA-256は`ac80a243407a1a458a643536c207cdab12812ba0550df500234ac9a8b5db7b12`、chat評価JSONは`ef5a16a007e4e73c8d33da42c794abfeb378769ea9943da0131d77271efc7edd`、chat可読TXTは`2c0f72c8402cab066ecab78c04dbcad089e0f155a7af8f735fae80d5b235c1df`です。

以上から、context 512 RoPEは今回の小規模条件でvalidation lossと複数domain lossの低下を示しましたので、探索上は有望と判定します。ただし、学習Token量が倍になった非対照要因があり、固定会話promptの意味的な応答や医療内容の正確性は改善したと認定できません。次はcontext長を固定してRMSNormまたはSwiGLUを一つだけ導入するか、同じcontext 512でstep数を半分にしてToken予算を揃える対照を作る必要があります。

## 次に試すこと

同じcontext 512で学習stepを250へ下げ、context 256・500 stepとToken予算を近づける対照を作ると、context長と学習量の影響を一部切り分けられます。その後はcontext 512を固定したままRMSNormまたはSwiGLUを一つだけ導入し、現代的な構造変更を独立に比較します。Colabの新規T4 kernelが確保できた場合は、まずPython・PyTorch・T4のprobe、実行コードcommit、bundle hashを検証した後に041を再試行し、Colabが使えない間はMacBookの小型構造実験を続けます。
