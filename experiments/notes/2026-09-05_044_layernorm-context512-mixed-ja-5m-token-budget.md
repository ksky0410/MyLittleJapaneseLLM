# 実験044：LayerNorm・context 512のToken予算対照

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験042ではcontext 512を500 step学習したため、context 256・500 stepの実験014より1 stepあたりの処理Token数が2倍でした。実験044では、実験042と同じLayerNorm・RoPE・context 512モデルを250 stepだけ学習し、context 256・500 step条件と総処理Token数を約1.024M Tokenで揃えます。これにより、実験042で見えた改善が長いcontextそのものによるのか、単に学習Token量が多かったためなのかを一部切り分けます。

今回の仮説は、学習Token予算を揃えるとcontext 512のvalidation loss改善幅は実験042より小さくなる、または消える可能性があるというものです。長いcontextが有効なら、処理Token数を揃えてもcontext 256と同等以上になると予想します。ただし、context 512は一度に見る系列が長い一方、250 stepではoptimizer更新回数が半分になるため、系列長の効果と更新回数の効果は完全には分離できません。この限界を明記した探索的対照として扱います。

実験042との差分は、context 512を維持したまま`max_steps = 250`とし、記録間隔を50 stepへ変更した点です。norm_typeはLayerNormで、実験042と同じです。dim 240、6層、6 heads、MLP倍率4、batch size 8、AdamW、学習率3e-4から3e-5、warmup 300、weight decay 0.1、seed 42、学習・検証Token列、Tokenizerを固定します。context 512・batch 8・250 stepでは、1,024,000 Token相当を処理します。モデルの概算parameter数は5,136,480です。

## 使用するデータ、Tokenizer、コード

学習Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`で、Token数は1,336,619、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`で、11,780 Token、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。会話validationは`artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin`、医療validationは`artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin`を使用します。一般80%、会話10%、医療10%の混合コーパスから作った既存Token列を読み取り専用で使用し、元の医師国家試験データや`medilink_analysis`の原本には変更を加えません。

TokenizerはSentencePiece Unigram、語彙数4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。実験設定は`configs/layernorm-context512-mixed-ja-5m-token-budget-smoke.toml`、SHA-256は`1c4542676a4732df465949a62d9cc177f0b18b6276ecd174d522284301d0a5a3`です。使用コードの基準commitは`c9098c5`で、RMSNorm対応とcheckpoint互換性の実装は`4d1b459`です。

予定している学習コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py \
  --config configs/layernorm-context512-mixed-ja-5m-token-budget-smoke.toml
```

学習後は、`scripts/evaluate_domains.py`でgeneral・conversation・medical・fineweb・wikipediaのvalidation lossを測定し、`scripts/evaluate_chat_prompts.py`でIssue #1の固定会話promptを評価します。checkpoint reload後には`今日は`、`吾輩は`、会話marker、`問題：`を生成します。学習途中の全生成文、崩れた出力、空の出力、評価JSON、checkpoint metadataを削除せず保存します。

## 成功条件と判定方法

250 stepをNaN、Metalエラー、OOM、Token列不足なく完走し、metrics、checkpoint metadata、stepごとの生成TXTが保存されれば実装上の成功とします。checkpoint reloadと固定prompt生成が成功することも確認します。品質比較では、主に実験014のcontext 256・LayerNorm・500 stepとgeneral validation lossを比較します。今回のgeneral lossが低くても、更新回数が少ないため「context 512が優れる」と即断しません。実験042との比較では、学習Token予算を揃えたときの差を記録します。

## 実験中の記録

2026-09-05 12:50 JST、学習開始前に設定、コードcommit、Tokenizer、学習Token列、general validation Token列のSHA-256を確認しました。実行環境はPython 3.13.1、macOS 15.5 arm64、MLXのdeviceは`Device(gpu, 0)`です。同じ出力先を使う別の学習プロセスがないことも確認しました。ノートと設定はcommit `e9b0a17`でpush済みです。学習中は50 stepごとにloss、perplexity、生成文、所要時間を保存し、異常や予定変更があれば直ちに追記します。

学習開始後、別プロセスとの競合、NaN、Metalエラー、OOM、Token列不足は発生しませんでした。step 1、50、100、150、200、250のvalidation lossは順に8.7620855967、7.1652158101、6.6710743904、6.4310317039、6.1675168673、5.9643232028でした。最終step 250まで完走し、所要時間は63.16秒でした。

（ここへ開始時刻、実行環境、stepごとの記録、警告、停止理由を追記する。）

## 結果と解釈

MLXでの学習は250 stepまで正常に完走し、step 250が最良checkpointでした。最終train lossは4.7627973557、general validation lossは5.9643232028、perplexityは389.2894688582でした。モデルの概算parameter数は5,136,480です。step 0から250まで50 step間隔で6件の生成文を保存し、step 250では医療問題の形式と英数字の断片が混ざる崩れた出力になりました。metrics、summary、checkpoint metadata、生成文は削除していません。

Token予算を揃えた主比較では、context 256・500 stepの実験014のgeneral loss 5.5338133176に対して、context 512・250 stepの044は5.9643232028で0.4305098852高くなりました。perplexityも実験014の253.1072514355から389.2894688582へ上昇しました。context 512・500 stepの実験042はgeneral loss 5.3410979907でしたので、044は0.6232252121高くなっています。この結果は、実験042の改善が長いcontextだけで説明できず、約2.048M Tokenを処理した学習量と、500回のoptimizer更新が大きく寄与した可能性を示します。

domain評価では、general loss 5.9643232028、conversation loss 3.9171713193、medical loss 4.9621599515、fineweb loss 6.3432046572、wikipedia loss 6.8143019676でした。詳細は`artifacts/evaluations/layernorm-context512-mixed-ja-5m-token-budget-smoke-domains.json`へ保存し、SHA-256は`d383ea9f1206a54083d84a46d81fd56d1d1ffe76d5dec1544834ba571b1c520e`です。実験014と比べるとconversationは3.5064193408から0.4107519785、medicalは4.5083060265から0.4538535295、それぞれ悪化しました。044は更新回数が半分で、同じToken数でも学習の進みが足りない可能性があります。finewebとwikipediaは044で追加測定しましたが、014では同じ条件の値を保存していないため、構造差の判定には使いません。

Issue #1の固定会話prompt 8件は`artifacts/evaluations/layernorm-context512-mixed-ja-5m-token-budget-smoke-chat.json`と`artifacts/samples/layernorm-context512-mixed-ja-5m-token-budget-smoke/chat-issue-1.txt`へ保存しました。044では8件中5件がEOSへ到達し、空completionは1件、平均completion長は100.625 Tokenでした。`まじで`、`それな`、`いやそれは`では医学問題や数値の断片が最大長まで続き、短い会話応答としては崩れています。実験042の8件中8件EOS、平均4.75 Tokenと比べて停止挙動も悪化しましたが、この固定promptだけからモデル品質を断定しません。chat JSONのSHA-256は`c554f5c8e5d5871f1a1ad221099dca84e2aab6da98a25202577a87d0bc5e2074`、可読TXTのSHA-256は`a69811cbcf17b91599985e1613f4ebfc53c09eba3d07f9da3d93a45b6f62712c`です。

checkpoint reload後の`今日は`、`吾輩は`、会話marker、`問題：`の生成はすべて成功し、それぞれ`reloaded-today.txt`、`reloaded-story.txt`、`reloaded-conversation.txt`、`reloaded-medical.txt`へ保存しました。会話markerでは「はい、小いっし」と返りましたが、他の出力には日本語らしい断片、英数字の連続、`G`の連続などの崩れが残っています。step 250 checkpointのSHA-256は`a2051cc9794a97171569e7f33831d03307d80d6808f9a75fabea95a4dec3c596`です。metricsのSHA-256は`2aed0060cb77dfd4148bfda6198a066a83e830bada0d2d03870518e0f4efa6fa`、summaryは`4cbd0ca5cfe0faec971362532f1ad07ef1993a7c1ebc3d90849dc0e81a10adff`、step 250 metadataは`f7d1e1ac33d84ef50713a94fe4b4afae38c9bf1e3ef6ee2f3ef0ff43b51e5d60`です。reload生成のSHA-256は順に、`d7ea714b090a4cfb671f2c3a9d7dd4ea69d86ef1f46d8754535eeabf4dc328f5`、`ede7e239df4ec33609d1a335cfc7e53d35cfba8aae0dd81f34d38a512386b64c`、`9354ac017af8e58e036ab3c0f69f0b39379b450fd6d51863beb9e0c69d4391b5`、`5f86e7f969646204300de5999fb9ea05c13433b43862099061403130bccf15b3`です。

以上から、今回の仮説は支持されました。少なくともこの5Mモデル、混合日本語コーパス、seed 42、学習Token予算約1.024Mの条件では、context 512へ伸ばしただけではcontext 256・500 stepを上回れませんでした。ただし、context 512側はoptimizer更新回数が半分であり、context長の純粋な効果を完全に検証したわけではありません。次回は更新回数と総Token数を同時に揃える比較、または複数seedでの再試行が必要です。

## 次に試すこと

次は、context 512・250 stepの044を基準に、SwiGLUを一つだけ導入してMLP構造の影響を測定します。ただし、SwiGLUだけの効果を見たい場合はnorm_typeをLayerNormに戻し、今回と同じToken予算・seed・評価手順を維持します。その後、RMSNormとSwiGLUを同時に使う現代的構成へ進みますが、変更を一度に増やしすぎないよう実験番号を分けて記録します。
