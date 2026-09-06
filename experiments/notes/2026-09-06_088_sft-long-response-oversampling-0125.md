# 実験088：長い応答例を12.5%へ穏やかに層化してSFTする

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。086の通常サンプリングではresponse 24 token以上が6.65%、087で25%へ増やすとheld-out chat F1は0.203292から0.216545へ改善しました。しかし087はgeneral 4.354792→4.374137、conversation 2.405646→2.433337など、5領域すべてのvalidation lossを悪化させました。長文例の増加には会話の話題継続を促す効果がある一方、25%は過剰と考えられます。

088では、086と087の中間としてSFT batchの12.5%をresponse 24 token以上の例から抽出します。モデル、base checkpoint、SFTデータ、rehearsal、EOS loss weight、seed、learning rate、学習stepは086・087と同一で、`--long-response-ratio 0.125 --long-response-min-tokens 24`だけを変更します。

仮説は、12.5%ならvalidation分布の長文比率に近づけつつ、087で見えたheld-out chatのshort・long改善を一部維持できることです。成功条件は、086よりheld-out chat F1が高く、087ほど5領域lossを悪化させないことです。固定promptで入力に応じた応答が出るかも確認しますが、生成が長くなっただけでは成功としません。強いLLMの蒸留、外部教師データ、生成データは使いません。

## 再現条件

実験開始時点のGit commit、入力ハッシュ、Colab bundle、session名、実行コマンドを実行前に追記します。モデルは50,207,616 parameter、dim 576、12層、9 heads、context length 256、RoPE、LayerNorm、SwiGLUです。batch size 8、10,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 123、rehearsal ratio 0.20、EOS loss weight 0.50です。

## 実験中の記録

ここに入力検証、stepごとのloss、生成文、異常、途中停止を追記します。生成文は100 step間隔で保存し、1,000 stepを超えて記録を空けません。

## 実験終了後の結果と解釈

ここにbest checkpoint、学習時間、validation loss、5領域評価、48例chat-test、Issue #1固定prompt、artifactのSHA-256、086・087との差を追記します。長文比率が中間条件として有効でなかった場合も、そのまま記録します。

## 次に試すこと

088が最も良ければ12.5%を暫定採用し、次に応答機能別の層化へ進みます。改善しなければ長さの再重み付けを打ち切り、一般日本語pretrainingのデータ量と学習回数、会話SFTの書式を優先して見直します。すべて蒸留なしで進めます。
