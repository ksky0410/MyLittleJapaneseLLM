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

Colab sessionは`exp087-long-response`、T4 GPUです。bundleは263,504,635 bytes、SHA-256は`b67e0b307cb94926225ca0b868dc0b1b47283e54237e3e03e0aa711139e34d73`で、Colab側の再結合後にも一致しました。bootstrapは9入力のハッシュ検証を通過しました。学習はstep 10,000まで完走し、OOM、NaN、shape error、途中停止はありませんでした。学習時間はsummary上1,273.95秒、peak allocated memoryは1,486,776,832 bytesでした。

validation lossはstep 500で3.488618、1,000で3.412264、2,000で3.346446、3,000で3.257151、4,000で3.185020、5,000で3.167959、6,000で3.143825、7,000で3.119076、8,000で3.100458、9,000で3.094433、9,500で3.102101、10,000で3.097132でした。bestはstep 9,900で、best validation lossは3.087942です。086は最終stepがbestでvalidation loss 3.046459だったため、長文例の過剰サンプリングはvalidationでは不利でした。

学習中の固定生成はstep 0で`こんにちは`、step 5,000で`こんにちは!よろしくおねがいします!`、step 10,000で`こんばんはー!こちらこそよろしくお願いします!`でした。086のstep 10,000は`こんにちはー!`だけでしたので、長文条件はEOSまでの継続を促しました。ただし、この差だけでは意味に沿った応答とは判断しません。

## 実験終了後の結果と解釈

best checkpointはstep 9,900、`artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each-long025/best.pt`、SHA-256は`7f4627e2459c50dd7cbac2e24b7e1c99c512f731e69b8892da51bdb1ad053605`です。best checkpoint archiveは200,878,080 bytes、SHA-256は`92007c6455b0e660983659961abc114b7526ddfbc00f034eff24bb4de195d007`、軽量archiveは19,385 bytes、SHA-256は`06344d6419ebdd79fbdf7997abcdb072a6b6b3201bd51408863524f7eb42941b`、manifestは`8b1f833fa0c9ebf3693b33422af065e6f5410afd3619cc29b6adcfd3b4d5edf8`です。summaryのSHA-256は`2c1b0c3a752abf3f5d505e8206046478a63a162a5ffa39396b55d6022a4935ce`、metricsは`433b0841e24a8e4c10b5e5e2c0e7aafa92458f3892f3b428c1661ec09361aa21`、best metadataは`564c69bf25a75397be755875b443ff4054e69ff04e7ce912262e32107284653d`です。

5領域のvalidation lossは、086からgeneral 4.354792→4.374137、conversation 2.405646→2.433337、medical 2.521558→2.528366、RPC 2.355622→2.385357、MRMP 2.001416→2.007640となり、すべて悪化しました。長い応答を25%へ増やしたことでSFT train側の応答機能は広がりましたが、086の分布に対しては過剰な再重み付けとなり、次トークン予測の汎化を損ねています。

一方、48例のheld-out chat-test全体F1は0.216545で、086の0.203292から0.013253改善しました。shortは0.352868→0.381520、mediumは0.139040→0.136888、longは0.117967→0.131226です。平均生成token数は7.896→8.438へ増え、EOS到達率は48/48のままでした。したがって、長文例の層化は特に短い応答と長い履歴への語彙的な重なりを改善しましたが、validation lossと中間長の応答を犠牲にしています。085の全体F1 0.236752、short 0.418210にはまだ及ばず、086より良いという理由だけで採用条件とはしません。

Issue #1固定promptのconversation形式では8/8がEOSへ到達しましたが、`まじで→こんにちは`、`それな→こんばんはー`、`今日なにしてた？→こちらこそ、よろしくです`、`やば→こんにちは`、`なんかさ→こんばんは!`、`いやそれは→おはようございます`、`おつかれ→よろしくお願いします`、`明日ひま？→こんにちは!`となりました。入力の意味に応答した例はなく、086と同じ挨拶縮退です。raw形式では`まじで`や`今日なにしてた？`から長い一般文、`いやそれは`や`おつかれ`から古風な文が続きましたが、自然な口語応答にはなっていません。全生成結果は`artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each-long025-issue1-prompts-raw.txt`、`...-conversation.txt`、held-out全文は`...-chat-test-v1.txt`に保存しています。学習途中の100 step間隔の生成文も`artifacts/samples/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each-long025/`へ保存し、Git管理対象にします。

評価JSONのSHA-256は、5領域評価が`4fc01da45e972f501319e39aeb7a5f318a13d1696faa6a84bd0f6740175b361c`、held-out chatが`cd43b8b43d51b254c828241b822f555c95158d0a3deef2a0d637c19609376f89`、レビュー用テンプレートが`c241324f36ad9cd0301d93695797a87418e29642a4d9bb3deb7cb1acd84b5187`、Issue #1 conversationが`2aad0e9e4d95a8711ff9df7df3e6551fe91a96f2e00d84c1d720f58cecbff969`、rawが`a8933ae1e47b33c2a2bacfa8c9bc53a3d65c188c0525e747381228244051c28a`です。テキスト出力のSHA-256はchat-testが`b4c182f9e9f49889e8ac81b608f614a2794aee7b27a8271a41ef07b6a729d3b7`、conversationが`c362907c37f4444f04526ab45321f48b697c58a283ca835c09c9fed6b280a4ba`、rawが`c849649ae0b230aba4b5eb4833280c4e1960c111411d5e61be6e1a195f57644e`です。

事前の仮説は一部だけ支持されました。長い応答の割合を増やすと、held-out chat-testのshort・long F1と生成の継続性は改善しました。しかし、固定promptでは意味応答が現れず、5領域validation lossはすべて悪化しました。これは、086の単純なランダムサンプリングが短文挨拶へ偏りすぎていたという問題と、長文を一律に25%へ増やすと会話分布全体から外れるという問題が同時に存在することを示します。自然な日本語性能を高めるには、長さだけでなく、質問に答える、相づちを返す、否定する、話題を継続する、会話を終えるという応答機能を分け、validationの比率と合わせてサンプリングする必要があります。

## 次に試すこと

087は086よりheld-out chatが改善したため、長文サンプリングは部分的な候補として残します。ただし25%固定ではなく、validationのresponse長分布に合わせた穏やかな層化、たとえば24 token以上を10〜15%へする条件を次に比較します。同時に、データを応答機能別に分けるため、元の会話JSONLとSFT配列を対応付け、定型挨拶だけの例を抑えた品質管理版を作ります。さらに、SFTで会話形式へ寄せる前に、一般日本語pretrainingを5Mから20M token以上へ増やした50M checkpointを作り、基礎的な自然文生成能力そのものを強化します。どの条件でも強いLLMの蒸留や生成教師は使わず、公開コーパスと既存会話データの構成・学習回数・サンプリングだけで改善を追います。
