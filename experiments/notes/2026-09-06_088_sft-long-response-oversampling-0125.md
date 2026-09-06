# 実験088：長い応答例を12.5%へ穏やかに層化してSFTする

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。086の通常サンプリングではresponse 24 token以上が6.65%、087で25%へ増やすとheld-out chat F1は0.203292から0.216545へ改善しました。しかし087はgeneral 4.354792→4.374137、conversation 2.405646→2.433337など、5領域すべてのvalidation lossを悪化させました。長文例の増加には会話の話題継続を促す効果がある一方、25%は過剰と考えられます。

088では、086と087の中間としてSFT batchの12.5%をresponse 24 token以上の例から抽出します。モデル、base checkpoint、SFTデータ、rehearsal、EOS loss weight、seed、learning rate、学習stepは086・087と同一で、`--long-response-ratio 0.125 --long-response-min-tokens 24`だけを変更します。

仮説は、12.5%ならvalidation分布の長文比率に近づけつつ、087で見えたheld-out chatのshort・long改善を一部維持できることです。成功条件は、086よりheld-out chat F1が高く、087ほど5領域lossを悪化させないことです。固定promptで入力に応じた応答が出るかも確認しますが、生成が長くなっただけでは成功としません。強いLLMの蒸留、外部教師データ、生成データは使いません。

## 再現条件

実験開始時点のGit commitは`c1bbfca`です。設定ファイルのSHA-256は`70978d302359c7803f1ce86ff04844a64fb2dbdfbae127790e85a5e6ea27961f`、bootstrapは`5c83dd27c0bec2b2fee1ef311b7d82d8372a39f0e70ff88228153e031240f534`、bundle結合スクリプトは`b5fb932d12aee0fd40b02a5c1f85adf3d1b5887e517fb38e56131408f7b56103`です。bundleは263,513,831 bytes、SHA-256は`cf9dc878a24b1733e1144ef06eea0b3ff9bdcfb3faf45310d7a9d3f2bc7ba361`です。入力は086・087と同じで、base checkpointは`1e09a7386a630133503c12052f7701216d505cb3bca8765cade38c879ba5e8cb`、SFT trainは`001dc022a998abc5756f641b199988112db77ff42903485ff7a6fd6bd0e028a3`、validationは`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`です。モデルは50,207,616 parameter、dim 576、12層、9 heads、context length 256、RoPE、LayerNorm、SwiGLUです。batch size 8、10,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 123、rehearsal ratio 0.20、EOS loss weight 0.50です。

## 実験中の記録

Colab session `exp088-long0125`の作成をT4で2回試みましたが、Colab APIが`Service Unavailable`（HTTP 503）を返してsessionを作成できませんでした。L4もquotaまたはentitlement不足で拒否されました。既存のColab sessionはなく、087までの成果物や088のbundleは変更されていません。GPUが復旧するまで学習は開始せず、条件を変更しません。

ここに、GPU sessionが確保できた後の入力検証、stepごとのloss、生成文、異常、途中停止を追記します。生成文は100 step間隔で保存し、1,000 stepを超えて記録を空けません。

## 実験終了後の結果と解釈

ここにbest checkpoint、学習時間、validation loss、5領域評価、48例chat-test、Issue #1固定prompt、artifactのSHA-256、086・087との差を追記します。長文比率が中間条件として有効でなかった場合も、そのまま記録します。

## 次に試すこと

088が最も良ければ12.5%を暫定採用し、次に応答機能別の層化へ進みます。改善しなければ長さの再重み付けを打ち切り、一般日本語pretrainingのデータ量と学習回数、会話SFTの書式を優先して見直します。すべて蒸留なしで進めます。
