# 実験087：長い応答例を層化してSFTする

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。086では、RPCとMRMPのSFT response tokenを各770kへ増やした結果、5領域のvalidation lossはすべて改善しました。しかし、48例のheld-out chat-test F1は085より低下し、Issue #1固定プロンプトでは入力に関係なく挨拶へ収束しました。単純な例数やresponse tokenの増加だけでは、自然な日本語の会話性能につながらないことが分かりました。

今回は、同じ086データを使い、応答の長さ分布だけを変えます。086のtrainはresponse長が24 token以上の例が6.65%、32 token以上の例が1.13%しかありません。一方、変更しないvalidationは24 token以上が14.05%あります。ランダムに例を選ぶだけでは長い応答や話題継続を学ぶ機会が少ないため、batch内の25%をresponse 24 token以上の例から抽出します。学習step、モデル、base checkpoint、seed、learning rate、rehearsal、EOS重み、SFTデータの総量は086と揃え、`--long-response-ratio 0.25 --long-response-min-tokens 24`だけを変更します。

仮説は、長い応答を意図的に増やすことで、モデルが短い定型挨拶だけを選ぶ傾向を弱め、質問への回答、前の話題の継続、自然な文末を学びやすくなることです。成功条件は、086よりIssue #1固定プロンプトの入力別応答が増え、held-out chat-testのF1が改善することです。validation lossだけが下がっても、応答が長くなるだけでも成功とはしません。強いLLMの蒸留、外部教師データ、生成データは使いません。

## 再現条件

実験開始時点のGit commitは`aa65b06`です。実験086のbaseは`artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt`で、SHA-256は`1e09a7386a630133503c12052f7701216d505cb3bca8765cade38c879ba5e8cb`です。SFT trainは`artifacts/sft/issue1-both-balanced-770k-each-v1/train.npz`、SHA-256は`001dc022a998abc5756f641b199988112db77ff42903485ff7a6fd6bd0e028a3`です。validation、rehearsal token、tokenizerは086と同じものを使います。

設定ファイルは`configs/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each-long025.toml`です。モデルは50,207,616 parameter、dim 576、12層、9 heads、context length 256、RoPE、LayerNorm、SwiGLUです。batch size 8、10,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 123、rehearsal ratio 0.20、EOS loss weight 0.50です。SFT batchの25%をresponse 24 token以上に固定する点だけが086と異なります。

学習前に次のコマンド、設定・データ・成果物のSHA-256、Colab session名を記録します。学習中は100 stepごとのlossと生成文を保存し、1,000 stepを超えてノート更新を空けません。終了後は5領域、48例chat-test、Issue #1固定promptのraw/conversationを086と同じ条件で評価します。

## 実験中の記録

ここにColabでの入力検証、学習経過、異常、停止理由、生成文の変化を追記します。

## 実験終了後の結果と解釈

ここにbest checkpoint、学習時間、validation loss、held-out chat、Issue #1固定prompt、artifactのSHA-256、086との差を追記します。長い応答の過剰学習によってvalidation lossが悪化した場合も、そのまま記録します。

## 次に試すこと

改善した場合は、response長だけでなく、質問・相づち・否定・話題継続・終了発話の機能別サンプリングへ進みます。改善しない場合は、長さの層化を主軸にせず、一般日本語pretrainingを5Mから20M token以上へ増やし、事前学習能力そのものを強化します。次の条件も蒸留なしで比較します。
