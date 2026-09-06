# 実験086：RPC・MRMPのSFTデータを各1M response tokenへ増やす

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。実験085では、082と同じSFTデータを10,000 stepまで反復すると、5領域validation lossと48例chat-test全体F1が改善しました。一方、Issue #1固定プロンプトでは「こんにちは」「こんばんは」「よろしくお願いします」への偏りが残り、同じbalanced subsetを長く見るだけでは口語応答の多様性が増えないことが分かりました。

現在の`issue1-both-balanced-v1/train.npz`は、RPCとMRMPから各385k response tokenを選んだ64,423例です。元のtrain NPZにはRPC約5.13M、MRMP770,975 response tokenがあります。086では、元のtrain NPZからRPCとMRMPを可能な範囲で同じ770,975 response tokenへseed固定で抽出し、`concat_sft_npz.py`で連結します。当初は各1,000,000 tokenを予定しましたが、MRMPの入力総量が770,975 tokenしかないため成立せず、RPC各770,975へ計画を修正しました。sourceごとの予算を等しくし、MRMP全量と同量のRPCを使うことで、現在の約2倍のresponse token、より多い会話例を使います。validation NPZと評価セットは変更しません。

仮説は、085のように学習stepを増やして同じ例を再利用するより、同じ10,000 stepで未観測の会話例を増やす方が、Issue #1固定プロンプトの定型挨拶への収束を弱め、意味に沿った相づち・質問応答・口語表現を増やすことです。モデル、base、learning rate、seed、rehearsal ratio、EOS loss weight、学習stepは085と完全に揃えます。成功条件は、085より5領域lossを大きく悪化させず、chat-test全体F1またはshort・medium・longの複数層を改善し、Issue #1固定プロンプトの応答が入力別に分かれることです。出力が長くなるだけ、EOSだけが維持されるだけ、挨拶の種類が変わるだけなら成功としません。強いLLMの蒸留や外部教師データは使いません。

## 再現条件

実験開始時点のGit commitは`c630321`です。作業ツリーには別作業由来の`README.md`、`web/`、`scripts/serve_chat.py`、`src/my_little_japanese_llm/chat.py`、`tests/test_chat.py`の未コミット変更がありますが、086では変更せず、コミットへ混ぜません。設定は`configs/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-1m-each.toml`です。

抽出前の入力は`artifacts/sft/issue1-rpc-full-v1/train.npz`と`artifacts/sft/issue1-mrmp-full-v1/train.npz`です。RPCの入力は315,584例・5,132,071 response token、MRMPの入力は81,382例・770,975 response tokenです。修正版ではRPC target 770,975、seed 8601、MRMP target 770,975、seed 8602で抽出し、出力を`artifacts/sft/issue1-both-balanced-770k-each-v1/train.npz`へ連結します。validationは`artifacts/sft/issue1-both-full-v1/validation.npz`を使います。

085と同じ50,207,616 parameter、dim 576、12層、9 heads、context length 256、RoPE・LayerNorm・SwiGLU、batch size 8、10,000 step、eval/sample interval 100、checkpoint interval 500、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 123、rehearsal ratio 0.20、EOS loss weight 0.50です。baseは081 best checkpointで、SHA-256は`1e09a7386a630133503c12052f7701216d505cb3bca8765cade38c879ba5e8cb`です。Tokenizerは`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`、rehearsal Token列は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`、Issue #1 promptは`b538af4f00668e60e712aa796b1de5d51e0f677c8b9a19bc4445d40a90929594`です。

ローカルでのデータ作成コマンドは次のとおりです。

```bash
uv run python scripts/select_sft_npz.py --input artifacts/sft/issue1-rpc-full-v1/train.npz --output artifacts/sft/issue1-rpc-770k-v1/train.npz --manifest artifacts/sft/issue1-rpc-770k-v1/manifest.json --target-response-tokens 770975 --seed 8601
uv run python scripts/select_sft_npz.py --input artifacts/sft/issue1-mrmp-full-v1/train.npz --output artifacts/sft/issue1-mrmp-770k-v1/train.npz --manifest artifacts/sft/issue1-mrmp-770k-v1/manifest.json --target-response-tokens 770975 --seed 8602
uv run python scripts/concat_sft_npz.py --input rpc=artifacts/sft/issue1-rpc-770k-v1/train.npz --input mrmp=artifacts/sft/issue1-mrmp-770k-v1/train.npz --output artifacts/sft/issue1-both-balanced-770k-each-v1/train.npz --manifest artifacts/sft/issue1-both-balanced-770k-each-v1/manifest.json
```

SFTのローカル再現コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-1m-each.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-770k-each-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each \
  --samples-dir artifacts/samples/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each \
  --lr-schedule-steps 10000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

## 実験中の記録

当初のRPC・MRMP各1,000,000 response token抽出を実行したところ、RPCは61,587例・1,000,005 tokenを作成できましたが、MRMPは入力総量770,975 tokenのため`ValueError: 入力のresponse Token数が不足しています: 770975 < 1000000`で停止しました。RPC 1M出力は削除せず、当初計画の未使用成果物として残します。この失敗を受け、MRMP全量の770,975 tokenに合わせてRPCも770,975 tokenへ抽出する条件へ変更します。

修正版の抽出はRPC 47,518例・770,981 response token、MRMP 81,382例・770,975 response tokenとなり、連結後は128,900例・1,541,956 response tokenです。修正版train NPZのSHA-256は`001dc022a998abc5756f641b199988112db77ff42903485ff7a6fd6bd0e028a3`、manifestは`8ba77285ffed2a3738a45f6adf2ad7350056eebf9e1aedc55ee5d42d38a6a382`です。設定ファイルのSHA-256は`919d1acee3929091b399dc8863d1906b4097ee90abea57baedcf0a7990f2f726`、Colab bootstrapは`60f520d94a40c875b082ada97b5553d37b95599435cc5becd0c61cc76ea06941`、成果物回収スクリプトは`53a27481110a9d5af689221b2d509ca56f0e6a3f36a184e3d850ea8a375e9c19`です。修正版bundleは`/tmp/small_llm-colab-086.XXXXXX.tar.gz`、263,083,412 bytes、SHA-256は`fa282fabac842a094697db59dacd8c3befcf6e3a3985cf23d9364bcae37ba117`です。60 MiB以下の分割片5個を用意し、Colab側で結合後に同じSHA-256を検証します。学習中はmetricsと生成文を100 stepごとに確認し、1,000 stepを超えてノート更新を空けません。データ抽出が失敗した場合、入力不足、例数、response token数、ハッシュ不一致、OOM、validation悪化、Issue #1固定promptの挨拶偏重を削除せず記録します。

Colab session `exp086-issue1-diverse-sft`をT4で作成し、bundle再結合後のSHA-256一致を確認しました。bootstrapは9入力のhash検証を通過し、086本学習を開始しています。

086本学習はColab T4で10,000 stepを完走しました。OOM、NaN、shape error、途中停止はありませんでした。step 500のvalidation lossは3.504923、1,000は3.465738、1,500は3.407572、2,000は3.365918、2,500は3.309586、3,000は3.263908、3,500は3.240569、4,000は3.217721、4,500は3.203465、5,000は3.170082でした。その後もstep 6,000で3.118876、7,000で3.096408、8,000で3.066016、9,000で3.058462、9,500で3.048642まで改善し、最終step 10,000で3.046459となりました。今回のbest checkpointは最終stepです。

学習時間はsummary上1,185.99秒、PyTorch 2.11.0+cu128、CUDA 12.8、Tesla T4、AMP有効、peak allocated memory 1,491,208,704 bytesでした。best weightのSHA-256は`10d59191d680f6e5f31ea1220048c62656b5d89002e47552e2539b4b1ac62ccb`、best checkpoint archiveは200,878,080 bytes・SHA-256 `f033ba06bac6cc98663320a89f9862dc78c3377ca81d63b07f3da7dd5de6b4a6`、lightweight archiveは19,035 bytes・SHA-256 `5e89319efa22d54e1fc1c3919278f1000244cf6e56ac558f101d4e99be6fcbff`です。metrics、checkpoint metadata、summary、100 step間隔の生成文を回収しました。

学習中の固定生成はstep 0で「こんにちは」、step 1,000で「こんばんは!」、step 3,000から9,000では「こんにちは!」、step 10,000では「こんにちはー!」でした。085と同じく、単一の固定生成は挨拶へ偏っており、データを増やした効果はこのサンプルから判定できません。予定どおり5領域・held-out chat・Issue #1固定promptを評価します。

## 実験終了後の結果と解釈

抽出後のSFTデータはRPC 47,518例・770,981 response token、MRMP 81,382例・770,975 response tokenで、連結後は128,900例・1,541,956 response tokenでした。085の64,423例・約770k response tokenから、例数と入力の多様性を増やしています。1Mずつという当初計画はMRMPの総量不足で成立しなかったため、実際の比較は「RPCとMRMPを各770k tokenで揃えた条件」となります。

学習はT4で10,000 stepを完走し、OOM、NaN、shape error、途中停止はありませんでした。best checkpointはstep 10,000、validation lossは3.046459、perplexityは21.040704、学習時間は1,185.99秒でした。085のvalidation loss 3.084671から0.038212改善しています。best weightは`artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each/best.pt`に保存し、SHA-256は`10d59191d680f6e5f31ea1220048c62656b5d89002e47552e2539b4b1ac62ccb`です。summaryは`f9d98a0c9349b05d5ed180cdebd065bb5efd77b092fd70ccf61f4aa72fc26eaa`、metricsは`0886348d4ed064b32f978dde99d8b6504329394dc1464cc10953e53fd4dda1f4`、best metadataは`ba8fc22e367084ff5aa9574f2bfaa0731d4392f503adf6e5b122d61f2f8b64cf`です。

085と同じ検証分割で測ったvalidation lossは、generalが4.361762から4.354792、conversationが2.441312から2.405646、medicalが2.522089から2.521558、RPCが2.403770から2.355622、MRMPが2.024484から2.001416となりました。5領域すべてが悪化せず、特に会話データとRPCで改善したため、データを増やして同じstep数で学習することは、少なくとも次トークン予測の汎化には有効でした。ただしmedicalの改善は0.000531に留まり、会話形式だけで一般的な対話能力が得られるとは判断できません。

48例のheld-out chat-testでは、086の全体F1が0.203292で、085の0.236752より0.033460低下しました。shortは0.418210から0.352868、mediumは0.154845から0.139040、longは0.137200から0.117967へいずれも低下しています。EOS到達率は両条件とも48/48で、平均生成token数は7.729から7.896へわずかに増えました。したがって、今回のvalidation loss改善は、未知の会話への意味的な応答性能の改善を意味しません。追加したRPC・MRMPの分布がvalidationには適合した一方、held-outの話題や応答の選択には十分な多様性がなかった可能性があります。

Issue #1固定プロンプトの会話テンプレートでは、8/8がEOSに到達したものの、`まじで`と`それな`に「おはようございます」、`今日なにしてた？`から`こんばんは!`、`明日ひま？`から`こんにちはー!`を出すなど、入力の意味をほぼ利用せず挨拶へ収束しました。086のconversation出力は次のとおりです。

```text
まじで → おはようございます
それな → おはようございます。
今日なにしてた？ → こんばんは!
やば → こんばんはー!
なんかさ → こんばんは!
いやそれは → こんばんは!
おつかれ → おはようございます。
明日ひま？ → こんにちはー!
```

raw形式では、`まじで`から医療文、`それな`から夏目漱石風の文、`今日なにしてた？`から古風な長文が生成されました。一方で`やば`から`い。`、`なんかさ`から`らいなさそうでした。`のような接続は見られます。全出力は`artifacts/evaluations/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each-issue1-prompts-raw.txt`と`...-conversation.txt`、学習途中の全100 step間隔の生成文は`artifacts/samples/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each/`に保存しています。生成文は省略せずGit管理対象にしました。

評価JSONのSHA-256は、5領域評価が`38ac988d35dae028e9f0a035a5b5ed9600cbaa40bf63e0e6611943cf0a2127a5`、held-out chatが`7384650d98b34271b882633b7b456b190a1f111a96ee1ed420eddf2dfba2ee25`、人手レビュー用テンプレートが`7c9083b8e867d966dfcbb58df33b43122a535cb846a7f0731c537dcf8c7d3b62`、Issue #1 conversation promptが`fa2fd6a89b3361151e2bb192d6381ccc4a262232a41b5b2b5f04fd6a8d2e96be`、raw promptが`7d3753917ed99af26613d7f6d9a9b88014f28b3dd23f4504ec13623b0b124e1b`です。対応するテキスト出力のSHA-256はchat-testが`1c9323b8aeff86bd530bd809934700b034714ec43ab3b3aecd3144bfce29783a`、conversation promptが`8b034c29dca4d407098251bcf835a72b8468e1be0812caa788317d5db73370c0`、raw promptが`ab1a87be83402d438498cf54d00bccfc37cfd6bedf3ad45f212473df76fcbb74`です。

以上から、086の仮説は「SFT例を増やせば固定プロンプトの挨拶偏重が弱まる」という部分では支持されませんでした。データを増やすこと自体はvalidation lossを下げましたが、自然な日本語を話す能力を高めるには、会話例の件数だけでなく、入力に応じた応答の多様性、質問への回答、相づち、話題継続、終了の仕方を分けて管理する必要があります。また、raw形式で古典・医療文へ漂流し、conversation形式で挨拶へ縮退するという差から、pretraining分布とSFTの会話書式の接続にも課題があります。強いLLMからの蒸留を使わずに性能を上げるという本プロジェクトの目的に照らすと、次は「より多く同じ分布を見る」よりも、一般日本語pretrainingのトークン量を増やしつつ、SFT側では重複・定型挨拶を抑え、応答機能別に層化したデータを作る方が妥当です。

## 次に試すこと

086で確認できた「validation lossは改善するが、固定promptとheld-out chatは改善しない」という差を出発点に、次は次の順で研究します。第一に、086のベースcheckpointからSFTを継ぎ足すのではなく、一般日本語pretrainingを5Mから少なくとも20M tokenへ増やした50Mモデルを作り、自然な日本語の基礎能力を先に伸ばします。第二に、RPC・MRMPのSFTデータから、定型挨拶の連続例を間引き、質問応答・相づち・短い返答・話題継続・終了発話の比率を明示した品質管理版を作ります。第三に、同じデータを1周だけ見る条件と、複数周見る条件を比較し、反復がvalidation lossだけを下げて会話の多様性を失わせていないか確認します。第四に、学習中の固定生成を挨拶だけでなく、質問、否定、話題継続、医療以外の一般話題を含むプロンプト集合へ拡張します。すべてランダム初期値からの事前学習、公開コーパス、既存データの再構成だけで実施し、強いLLMの蒸留は使いません。
