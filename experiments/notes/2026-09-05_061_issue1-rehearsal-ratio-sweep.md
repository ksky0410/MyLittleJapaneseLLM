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

bundle hash検証、Colab割当、各条件の学習開始・途中・完了、評価、回収、session停止を時系列で追記します。失敗、停止、生成崩れも削除せずに残します。

## 結果と解釈

実験終了直後に、ratio 0.25と0.75のloss、PPL、runtime、生成例を追記します。060のratio 0.50を中央条件として、会話評価とdomain保持のトレードオフを解釈します。

## 次に試すこと

比率の差が明確なら、chat F1とdomain lossの両方を見ながら採用比率を決めます。差が小さい場合は、会話sourceをRealPersonaChat単独・MRMP単独へ分けるsource ablation、または会話データの品質フィルタ比較へ進みます。その後、20Mで条件を固め、50Mへ拡大してIssue #1の会話能力がモデル容量でも再現するかを確認します。
