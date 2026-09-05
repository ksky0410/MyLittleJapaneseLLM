# 実験061：Issue #1会話SFTのrehearsal ratio 0.25/0.75比較

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。GitHub Issue [#1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)では、RealPersonaChatとMulti-Relational Multi-Party Chat Corpusを候補にし、標準文と会話データの混合、事前学習と会話SFTの分離、話者境界の保持、固定promptと生成本文の保存を求めています。既存の057〜060では両sourceを含む会話SFTとrehearsalを実施しており、060を基準にrehearsal比率の影響を比較します。

061ではEOS weight 0.50、6,000 step、cosine学習率の終点3,000 stepを固定し、rehearsal ratioだけを0.25と0.75へ変更します。0.25は会話応答を優先し、0.75は一般・医療・標準文の保持を優先する条件です。仮説は、0.25ではchat-test-v1のF1と生成長が上がる一方でdomain lossが悪化し、0.75ではdomain lossが改善する一方で短い定型応答へ戻るというトレードオフです。060の0.50と合わせて、Issue #1の「会話らしさ」と基盤保持のバランスを判断します。

## 条件と再現情報

060と同じbundle、base checkpoint、会話train/validation、rehearsal Token列、Tokenizer、モデル、batch、seed、EOS weight、max steps、learning-rate scheduleを使います。変更するのは`--rehearsal-ratio`だけです。モデルはRoPE・LayerNorm・SwiGLU、19,308,032 parameters、batch size 8、context length 256、max steps 6,000、lr schedule steps 3,000、seed 42です。元の医師国家試験データと`/Users/koseki/projects/medilink_analysis`は変更しません。

実行前にwrapper、package、bundle検証、noteをcommit・pushします。重いcheckpoint本体はGitへ追加せず、入力hashとColab manifestのcheckpoint hashを残します。生成文は各条件でstep 0から6,000まで100 step間隔で保存し、悪い出力も削除せずGitHubへ保存します。

## 成功基準

2条件がNaN、OOM、shape errorなく完走し、各条件のmetrics、summary、checkpoint metadata、生成TXT、domain評価、固定48例chat評価を回収できれば実装上の成功とします。060と同じ評価条件で、EOS到達率、平均生成長、token overlap F1、short・medium・long別F1、general・conversation・medical・RPC・MRMP lossを比較します。

## 実験中の記録

2026-09-05 17時台に、060と同じbundleを061用のファイル名へ付け替えてColab T4へ投入しました。再結合後のサイズは271,827,848 bytes、SHA-256は`44d2f1cad370dd346af6f68b05a67ab2c70d75617b90bd5583b1341fa0f43e07`で一致しました。実験wrapperと計画のコミットは`8bc6d04`、評価wrapperのコミットは`c4d21e2`です。

ratio 0.25と0.75の両条件が、PyTorch 2.11.0+cu128、CUDA 12.8、Tesla T4上で完走しました。ratio 0.25の学習時間は490.305211秒、ratio 0.75は495.506295秒で、最大allocated memoryはいずれも750,051,840 bytes（約715 MiB）でした。NaN、OOM、shape errorは発生しませんでした。

各条件についてstep 0から6,000まで100 step間隔の生成文、500 step間隔のcheckpoint metadataを回収し、059・060と同じ5領域評価と固定48例chat-test-v1評価を行いました。評価終了後にColab sessionを停止し、`colab sessions`でアクティブなsessionがないことを確認しました。

## 結果と解釈

ratio 0.25ではbest checkpointがstep 5,800、validation lossが3.2595858335、perplexityが26.0387505321でした。ratio 0.75ではbest checkpointがstep 6,000、validation lossが3.2887503266、perplexityが26.8093397954でした。060のratio 0.50はstep 5,800でvalidation loss 3.2746284962でしたので、SFT validationだけでは0.25が最良、0.75が最も高い値となりました。

domain validation lossは次のとおりでした。ratio 0.25はgeneral 4.3242033323、conversation 2.4545847575、medical 2.4473261038、RealPersonaChat 2.3677238623、MRMP 2.0909896692でした。ratio 0.75はgeneral 4.2561046282、conversation 2.4583896796、medical 2.3595569928、RealPersonaChat 2.3863271872、MRMP 2.0934098562でした。060のratio 0.50と比べると、ratio 0.25はgeneralとmedicalをそれぞれ0.035475、0.053344悪化させる一方、RPCを0.006165改善しました。ratio 0.75はgeneralを0.032624、medicalを0.034425改善しましたが、conversation、RPC、MRMPは悪化しました。

固定48例では、ratio 0.25がEOS 48/48、平均生成長12.4167 token、overall token overlap F1 0.244982でした。short・medium・longのF1はそれぞれ0.333940、0.213987、0.187020でした。ratio 0.75はEOS 48/48、平均生成長10.4792 token、overall F1 0.212745で、short・medium・longは0.331786、0.157623、0.148826でした。060のratio 0.50は平均11.4167 token、overall F1 0.233049でしたので、ratio 0.25は全層で会話F1を上げ、ratio 0.75はmediumとlongで下げました。EOS停止率はいずれも維持されており、ratioを下げたことで未停止が増える現象は見られませんでした。

Issue #1の仮説どおり、rehearsal ratioは会話らしさと基盤保持のトレードオフを動かしました。ratio 0.25はchat-test-v1の自動F1と生成長で最良でしたが、一般・医療domain lossが悪化し、単純な会話専用化に近づいています。ratio 0.75はgeneral・medical lossで最良でしたが、会話のmedium・long応答が短くなり、overall F1も下がりました。ratio 0.50は両者の中間で、総合的な標準条件としては依然もっとも扱いやすい候補です。ただし、用途を会話応答へ限定するならratio 0.25、一般・医療の保持を優先するならratio 0.75を選ぶ余地があります。

固定prompt`<|startofconversation|><|speaker:DA|>こんにちはー！<eos:3><|speaker:DC|>`は、両条件ともstep 0から6,000まで「こんにちは!」で停止しました。このpromptは条件差をほとんど捉えないため、評価判断には使わず、全生成ファイルとheld-out chat本文を確認できる補助記録として扱います。生成文は[ratio 0.25のサンプル](../../artifacts/samples/issue1-056base-rehearsal-ratio025-eos050-colab-6k-fixedlr3k/)と[ratio 0.75のサンプル](../../artifacts/samples/issue1-056base-rehearsal-ratio075-eos050-colab-6k-fixedlr3k/)に保存しています。

軽量アーカイブのSHA-256は`a6db780e54ebb31685da1e516098984cabae5bc35b62fcbcf7bf910b852fe8e0`、manifestのSHA-256は`a25729a368939e9d5ab292df7159e0cd54d995994f420b180f2b2d24d506d3a1`です。ratio 0.25 best checkpointのSHA-256は`9efb30c8b3480449fe3eaae3fb756225ecb89265f0360921eb362dbf99ae56a3`、ratio 0.75 best checkpointは`e7461e89d0c582d9a3a0489cad6c479377c089d9aee0f512d077e1380ab07de0`です。checkpoint本体はGitの追跡対象外ですが、metadata、manifest、生成文、評価JSON/TXTは追跡対象として保存します。

## 次に試すこと

ratio 0.50を標準条件として保持し、用途別に0.25と0.75を使い分けられる状態にします。次はIssue #1のsource差を直接検証するため、同じ総会話Token予算でRealPersonaChat単独とMRMP単独を比較します。これまでのsource ablationはデータ構成の差が残っていたため、今回のrehearsal比率比較で得た条件を固定し、sourceだけを変える設計へ進みます。その後、20Mでsource差を確認してから50Mへ拡大し、Issue #1の会話能力がモデル容量でも再現するかを確認します。
