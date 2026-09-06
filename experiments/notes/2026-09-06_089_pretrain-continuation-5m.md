# 実験089：50M日本語事前学習checkpointへ追加5,000 stepを行う

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。086・087・088では会話SFTのデータ構成を比較しましたが、会話形式の固定promptでは挨拶への縮退が残りました。087では長文例を増やすことでheld-out chat F1の一部が改善しましたが、5領域validation lossは悪化しました。このことから、SFTだけで自然な日本語の基礎能力を作るのではなく、事前学習段階で同じ日本語コーパスを追加で何周も見せる効果を確認します。

089では、実験081の50M・5M token事前学習のbest weightから重みだけを読み込み、同じ5M token train列を追加5,000 step学習します。これは新しい教師モデルや蒸留ではなく、既存の日本語コーパスを追加で約2周見る継続事前学習です。初期checkpointはoptimizer状態を引き継がず、低いlearning rate 5e-5から5e-6、warmup 250、weight decay 0.01で新しいoptimizerを開始します。seedはデータ抽出の再現性と区別するため84に固定します。

仮説は、同じ日本語データを追加で学習することで、一般validation lossと固定日本語生成が改善し、SFT前の基礎モデルが自然な文末・助詞・文脈を安定して生成することです。成功条件は、081よりgeneral validation lossを改善し、医療・会話・Issue #1固定promptの評価でSFT後の基礎に使える兆候が出ることです。損失だけが改善して生成が反復・崩壊する場合は失敗とします。

## 再現条件

実験開始時点のGit commitは`3b284a7`です。設定ファイルのSHA-256は`669fddbfb2fbba31cdbc79f90d1afae5cb0185c830cd293c9d38afc5043dfa07`、bootstrapは`6c643fd0ac3703b04c67566c043cae6aaa94cbf8560a6f1cd9a6c4f9957563f5`、bundle結合スクリプトは`d0a4754f66d4e24f739e5d1283639b4f2d22167a87a280ca2cd27b82727d163f`です。bundleは193,600,722 bytes、SHA-256は`4967234c12d5c7104d64f5e91b90fc1d11b5da192264171b305949d647604118`です。初期checkpointのSHA-256は`1e09a7386a630133503c12052f7701216d505cb3bca8765cade38c879ba5e8cb`、初期checkpoint metadataは`4b6b56ad60730cc75a938dd8ef99aba6e713e03852043e8fe9175ef5d5c2813b`、学習Token列は`54eb3fab617c94bda59899db4f78e6ac65665606219414a710e40cc8ccb8603c`です。モデルは50,207,616 parameter、dim 576、12層、9 heads、context length 256、RoPE、LayerNorm、SwiGLUです。batch size 8、5,000 step、learning rate 5e-5から5e-6、warmup 250、weight decay 0.01です。

## 実験中の記録

ここに入力検証、stepごとのloss、生成文、異常、途中停止を追記します。生成文は100 step間隔で保存し、1,000 stepを超えて記録を空けません。

## 実験終了後の結果と解釈

ここにbest checkpoint、学習時間、validation loss、5領域評価、固定prompt、artifactのSHA-256、081・086との差を追記します。継続学習で悪化した場合も削除せず、学習率・データ反復・過学習のどこが原因かを残します。

## 次に試すこと

改善した場合はこのcheckpointをSFTの新しいbaseにし、同じ086データでSFTを再実行して、pretraining改善が会話性能へ伝播するか確認します。改善しない場合は同じデータの反復を続けず、FineWeb2日本語とWikipediaの品質・重複・配分を見直してから追加データを作ります。
