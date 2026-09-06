# 実験079：seed 123で長文応答25%層化SFTを対照比較

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。Issue #1の現代的な会話データを、一般日本語と医師国家試験由来の医療データを含む共通の日本語モデルへ組み込みます。医療専用モデルにはせず、一般・会話・医療・RPC・MRMPを分けて評価します。元の`/Users/koseki/projects/medilink_analysis`と医師国家試験の原データは変更・削除しません。

実験078では、075の50M・約5M Token事前学習済みbaseに標準response-only SFTをseed 123で行いました。実験077では、同じbase・データ・seed42でSFT部分の応答24 Token以上を2/6へ増やしましたが、077と076のseedが異なるため、長文比率の効果を特定できませんでした。本実験は078とbase、SFTデータ、rehearsal、seed、学習率、step、評価条件を完全に揃え、長文比率だけを0から0.25へ変更します。

仮説は、長文比率0.25にするとlong F1、平均生成長、長い履歴に対するToken overlapが改善する一方、short F1は低下することです。078との差が小さい、または悪化する場合は、50M・約5M Token基盤ではSFTバッチ内の長文oversamplingを採用せず、データ形式や評価方法へ進みます。今回は同じseedのpaired ablationなので、077よりも078との差を主な判断材料にします。

## 再現条件

モデルはRoPE・LayerNorm・SwiGLU、dim 576、12層、9 heads、context length 256、50,207,616 parametersです。設定上のseedは123、batch sizeは8、最大3,000 step、learning rateは5e-5から5e-6、warmup 100、weight decay 0.01、EOS loss weight 0.50です。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`、SFT trainは64,423例、validationは49,045例です。SFTとrehearsalは0.80対0.20で混ぜ、SFT部分6例から応答24 Token以上の長文を2例抽出します。

base checkpointは実験075のbestでSHA-256は`71931b2c689c2fbaa31c8c92c022a21fac571894ec2993a59be48644794e5e17`です。SFT trainは`645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、validationは`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`、rehearsal Token列は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`、Tokenizerは`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。設定ファイルのSHA-256は実験開始前に計算して追記します。
設定ファイル`configs/issue1-both-50m-sft-from-5m-long025-seed123-3k.toml`のSHA-256は`118c011929e67f963e93825bef4494780521b23c6891b9e31fa0a0ec3a059176`です。

再現コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-50m-sft-from-5m-long025-seed123-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-50m-sft-from-5m-long025-seed123-3k \
  --samples-dir artifacts/samples/issue1-both-50m-sft-from-5m-long025-seed123-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --long-response-ratio 0.25 \
  --long-response-min-tokens 24 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

Colab T4の割り当てを開始前に試し、HTTP 503などで失敗した場合は応答とsession状態を記録してMPSへ切り替えます。成功条件は3,000 stepを完走し、100 stepごとのmetricsと生成文、500 stepごとのcheckpoint metadata、summary、5領域評価、固定chat-test 48例、人手レビュー用JSONを保存することです。

Colab送信用bundleは`/tmp/small_llm-colab-079-XXXXXX.tar.gz`、236,435,471 bytes、SHA-256は`4d455e4fc73b803ae3a900ccf94fa46b050681b4a9f93d52acadcf3ad31f82da`です。bundleには079の実行コード、設定、075 best checkpoint、加工済みSFTデータ、rehearsal Token列、Tokenizerだけを含め、元JSONL、医師国家試験原本、`medilink_analysis`は含めていません。

## 実験中の記録

ここにはColab試行、bundle hash、MPSへの切り替え、step 1・500・1,000・1,500・2,000・2,500・3,000のloss・PPL・経過時間・学習率・固定prompt生成を時系列で追記します。警告、失敗、悪い生成も削除せず残します。

2026年9月6日09:09台に`colab sessions`を実行し、`No active sessions found on server.`を確認しました。その後、`colab new --session exp079-both-50m-sft-long025-seed123 --gpu T4`を実行しましたが、assignment endpointがHTTP 503 `Service Unavailable`を返しました。Colabセッションは作成されず、bundle upload、入力hash検証、モデル初期化、学習stepは発生していません。078までと同じ制約のため、079もMPSへ切り替えます。

同日09:11台に予定したMPSコマンドで学習を開始しました。step 1はtrain loss 4.082140、SFT loss 4.347381、rehearsal loss 3.021178、validation loss 4.342332、PPL 76.8866、learning rate 5e-7、経過時間4.14秒でした。step 100はtrain loss 3.966904、SFT loss 4.164704、rehearsal loss 3.175706、validation loss 3.837496、PPL 46.4091、learning rate 5e-5、経過時間66.09秒でした。step 200はtrain loss 3.440580、SFT loss 3.546707、rehearsal loss 3.016072、validation loss 3.802486、PPL 44.8125、learning rate 4.9871e-5、経過時間209.26秒、step 300はtrain loss 4.151612、SFT loss 4.388435、rehearsal loss 3.204319、validation loss 3.781122、PPL 43.8652、learning rate 4.9479e-5、経過時間334.58秒でした。step 400はvalidation loss 3.779117、PPL 43.7774、learning rate 4.8830e-5、経過時間458.18秒でした。

step 500はtrain loss 3.880764、SFT loss 4.131837、rehearsal loss 2.876472、validation loss 3.744829、PPL 42.3018、learning rate 4.7931e-5、経過時間587.21秒でした。step 500の固定生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんばんは`でした。step 0〜500の生成本文、step 500のcheckpoint metadata、metricsを保存し、GitHubへpushします。NaN、OOM、shape error、警告はありません。

step 600はvalidation loss 3.713040、PPL 40.9782、learning rate 4.6792e-5、経過時間716.76秒でした。step 700は3.713716、PPL 41.0059、learning rate 4.5427e-5、経過時間843.25秒、step 800は3.698069、PPL 40.3693、learning rate 4.3852e-5、経過時間969.19秒でした。step 900は3.675959、PPL 39.4865、learning rate 4.2085e-5、経過時間1,096.51秒、step 1,000はtrain loss 3.854930、SFT loss 4.185379、rehearsal loss 2.533137、validation loss 3.650785、PPL 38.5049、learning rate 4.0147e-5、経過時間1,233.94秒でした。step 1,000で最良validationを更新しています。

step 1,000の固定生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんにちは`でした。step 600〜1,000の生成本文、step 1,000のcheckpoint metadata、metricsを保存し、GitHubへpushします。学習は継続中です。

step 1,100はvalidation loss 3.665414、PPL 39.0723、learning rate 3.8061e-5、経過時間1,379.41秒でした。step 1,200は3.654846、PPL 38.6616、learning rate 3.5851e-5、経過時間1,509.36秒、step 1,300は3.641851、PPL 38.1624、learning rate 3.3543e-5、経過時間1,644.97秒でした。step 1,400は3.615888、PPL 37.1844、learning rate 3.1164e-5、経過時間1,807.50秒、step 1,500はtrain loss 3.538447、SFT loss 3.512605、rehearsal loss 3.641814、validation loss 3.614141、PPL 37.1195、learning rate 2.8742e-5、経過時間1,956.57秒でした。step 1,500で最良validationを更新しています。

step 1,500の固定生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> えー、よろしくお願いします。`でした。step 1,100〜1,500の生成本文、step 1,500のcheckpoint metadata、metricsを保存し、GitHubへpushします。学習は継続中です。

step 1,600はvalidation loss 3.605783、PPL 36.8105、learning rate 2.6306e-5、経過時間2,084.30秒でした。step 1,700は3.598783、PPL 36.5537、learning rate 2.3884e-5、経過時間2,202.08秒、step 1,800は3.572048、PPL 35.5894、learning rate 2.1504e-5、経過時間2,330.61秒でした。step 1,900は3.593619、PPL 36.3655、learning rate 1.9195e-5、経過時間2,458.91秒、step 2,000はtrain loss 3.314458、SFT loss 3.282593、rehearsal loss 3.441917、validation loss 3.571540、PPL 35.5713、learning rate 1.6982e-5、経過時間2,587.86秒でした。

step 2,000の固定生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんにちは!`でした。step 1,600〜2,000の生成本文、step 2,000のcheckpoint metadata、metricsを保存しました。step 2,000時点ではvalidation lossの最良はstep 1,500の3.614141からstep 2,000の3.571540へ更新されました。学習は継続中です。

step 2,100はtrain loss 2.880641、SFT loss 2.946008、rehearsal loss 2.619175、validation loss 3.577139、PPL 35.7710、learning rate 1.4893e-5、経過時間2,722.14秒でした。step 2,200はtrain loss 3.656598、SFT loss 3.657779、rehearsal loss 3.651873、validation loss 3.573679、PPL 35.6475、learning rate 1.2952e-5、経過時間2,851.88秒でした。step 2,300はtrain loss 3.098670、SFT loss 2.905879、rehearsal loss 3.869832、validation loss 3.567085、PPL 35.4132、learning rate 1.1182e-5、経過時間2,981.63秒、step 2,400はtrain loss 4.024920、SFT loss 4.145559、rehearsal loss 3.542368、validation loss 3.560665、PPL 35.1866、learning rate 9.6027e-6、経過時間3,114.47秒でした。

step 2,500はtrain loss 3.071373、SFT loss 2.686692、rehearsal loss 4.610097、validation loss 3.542961、PPL 34.5691、learning rate 8.2333e-6、経過時間3,249.27秒でした。固定生成は引き続き`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんにちは!`でした。step 2,100〜2,500の生成本文、step 2,500のcheckpoint metadata、metricsを保存しました。step 2,500で最良validationを更新しました。

step 2,600はtrain loss 2.747316、SFT loss 2.950996、rehearsal loss 1.932600、validation loss 3.540011、PPL 34.4673、learning rate 7.0898e-6、経過時間3,386.20秒でした。step 2,700はtrain loss 3.328378、SFT loss 3.080291、rehearsal loss 4.320726、validation loss 3.541172、PPL 34.5073、learning rate 6.1856e-6、経過時間3,519.20秒、step 2,800はtrain loss 3.660130、SFT loss 3.598723、rehearsal loss 3.905756、validation loss 3.543153、PPL 34.5758、learning rate 5.5313e-6、経過時間3,649.06秒でした。step 2,900はtrain loss 3.119025、SFT loss 3.224555、rehearsal loss 2.696908、validation loss 3.531731、PPL 34.1831、learning rate 5.1345e-6、経過時間3,777.17秒でした。step 2,900で最良validationを更新しました。

step 3,000はtrain loss 2.949194、SFT loss 3.074243、rehearsal loss 2.448998、validation loss 3.536578、PPL 34.3492、learning rate 5.0000e-6、経過時間3,910.17秒でした。固定生成は`<|startofconversation|> <|speaker:DA|> こんにちは! <|speaker:DC|> こんばんは!`でした。step 2,600〜3,000の生成本文、step 3,000のcheckpoint metadata、metrics、summaryを保存しました。NaN、OOM、shape errorは発生していません。

## 実験終了後の結果と解釈

実際のbackend、最良checkpoint、学習時間、5領域loss、固定chat-testのEOS・長さ・precision・recall・F1、長さ別集計、078との差、生成本文の質的観察を追記します。評価結果、生成全文、checkpoint metadataのSHA-256を残します。

学習はMPSで3,000 stepを完走しました。Torchは2.14.0、CUDAは未使用、AMPは無効、パラメータ数は50,207,616、総経過時間は3,911.49秒でした。最良checkpointはstep 2,900で、validation lossは3.5317310214042665、PPLは34.18308808773353です。最終step 3,000はvalidation loss 3.536577832698822、PPL 34.34916922199306であり、評価には最良の`best.pt`を使います。`best.pt`のSHA-256は`03bd492d82fa31b1ee13e492d1fe2d7f1d69d3e4f122e32a4ac41370a2867bcb`、`best.json`は`847e6b6d27113694d72a494994ec40dabe12e0bc1008b6793aaf7a4f25e30afb`、`summary.json`は`84fbdbce70d697ffabab6c6b9620a4d844847891017f072c2116e00cb6a532d7`、`metrics.jsonl`は`7c7787fdd0c62ad5f5fc49f1812bdb02aa8bcfb4afa12351fcd93791c54ff6ec`です。

同一条件で評価した実験078との差は次のとおりです。5領域のvalidation lossは、generalが4.673127から4.682027へ+0.008900、conversationが2.724320から2.724248へ-0.000071、medicalが2.942104から2.952771へ+0.010667、RPCが2.706494から2.693884へ-0.012610、MRMPが2.253748から2.269132へ+0.015384でした。長文比率を変えただけで全体の言語モデル性能が一貫して上がったわけではなく、会話・RPCではわずかに改善した一方、general・medical・MRMPでは悪化しました。

固定chat-test 48例では、079もEOS到達は48例中48例でした。平均生成Token数は11.0625から13.5208へ+2.4583伸び、long層では12.3750から17.8125へ+5.4375伸びました。しかし全体F1は0.216804から0.181798へ-0.035005、short F1は0.341643から0.234270へ-0.107372、medium F1は0.174441から0.141558へ-0.032883と低下しました。long F1だけは0.134327から0.169567へ+0.035240改善しました。つまり、長文比率25%は長く生成する傾向とlong層の重複率を高めましたが、短い自然な応答と全体の正確さを犠牲にしています。

生成全文は`artifacts/evaluations/issue1-both-50m-sft-from-5m-long025-seed123-3k-chat-test-v1.txt`、機械評価は`artifacts/evaluations/issue1-both-50m-sft-from-5m-long025-seed123-3k-chat-test-v1.json`、人手レビュー用の48例は`artifacts/evaluations/issue1-both-50m-sft-from-5m-long025-seed123-3k-chat-review.json`に保存しました。SHA-256は順に`1551ce5aa52ca14aedf7b6a7eaf1e030089a6ba9a4bdc52915777744d9e372d9`、`edce353020be067c6afc6b78bec13d7480ad018b6bf388550bd1f1f36af2d33f`、`9c8aa3c204c304f9463f5015dab5225fd6f1e9f957f064edc154c63dba55d350`です。領域評価JSONは`artifacts/evaluations/issue1-both-50m-sft-from-5m-long025-seed123-3k-domains.json`で、SHA-256は`883c9de2cd172fab9a9866a83132a68d2e27a9741d7364b5eb326470f45c0891`です。

今回の結論は、長文比率25%を標準設定として採用しない、です。long層に絞った改善は確認できましたが、ユーザーが求めている「自然な日本語を話す強いモデル」という主目的に対して全体F1とshort層の低下が大きく、長文化と性能向上を混同してはいけません。次の主線では、この設定を使わず、同じ50Mモデルに対して学習Token数を増やし、同じデータを複数周回する方法、データ源の品質と比率、pretrainingからSFTへ移る順序を優先して調べます。

step 2,500以降もvalidation lossが改善し、step 2,900で最良になった一方、固定生成は短い挨拶からほとんど伸びませんでした。この段階では長文比率25%が「自然に長く話す」ことを保証したとは言えず、validation lossと会話生成長の両方を評価して判断する必要があります。

評価開始前の予定として、実験078と同じCPU・同じ5領域・同じ48例のchat-test・同じseed 42・max-new-tokens 64で評価します。実行コマンドは次のとおりです。

```bash
uv run python scripts/evaluate_torch.py domains \
  --config configs/issue1-both-50m-sft-from-5m-long025-seed123-3k.toml \
  --checkpoint artifacts/checkpoints/issue1-both-50m-sft-from-5m-long025-seed123-3k/best.pt \
  --device cpu \
  --domain general=artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin \
  --domain conversation=artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin \
  --domain medical=artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin \
  --domain RPC=artifacts/tokens/issue1-real-persona-chat-validation.bin \
  --domain MRMP=artifacts/tokens/issue1-mrmp-validation.bin \
  --eval-batches 20 \
  --output artifacts/evaluations/issue1-both-50m-sft-from-5m-long025-seed123-3k-domains.json
```

```bash
uv run python scripts/evaluate_torch.py chat \
  --config configs/issue1-both-50m-sft-from-5m-long025-seed123-3k.toml \
  --checkpoint artifacts/checkpoints/issue1-both-50m-sft-from-5m-long025-seed123-3k/best.pt \
  --selection experiments/evaluation/chat-test-v1.json \
  --input artifacts/corpus/conversation-v1/test.jsonl \
  --device cpu --max-new-tokens 64 --seed 42 \
  --output artifacts/evaluations/issue1-both-50m-sft-from-5m-long025-seed123-3k-chat-test-v1.json \
  --text-output artifacts/evaluations/issue1-both-50m-sft-from-5m-long025-seed123-3k-chat-test-v1.txt
uv run python scripts/create_chat_review_template.py \
  --evaluation artifacts/evaluations/issue1-both-50m-sft-from-5m-long025-seed123-3k-chat-test-v1.json \
  --output artifacts/evaluations/issue1-both-50m-sft-from-5m-long025-seed123-3k-chat-review.json
```

## 次に試すこと

079で同一seedのpaired comparisonを完成させた後、長文比率を採用するか保留します。ただし、この比較は最終目的ではなく、主線は蒸留に頼らずにモデルを強くすることです。次はcontext length 512、学習Token数の追加、同じデータを複数周回する学習、データsource比率の見直し、自然な会話形式のSFTを優先順位を付けて試します。蒸留や教師LLM由来データは主線の成功条件にせず、必要な場合だけ別系統の比較として記録します。
