# 実験081：50M日本語モデルの5M Token二周相当事前学習

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。080では約10M Tokenの新しい混合コーパスを5,000 step学習し、075の約5M Token・2,500 stepより5領域validation lossと固定chat F1が改善しました。ただし080はデータsource比率も変わっていたため、Tokenの多様性を増やした効果と、同じデータを繰り返し学習した効果を分離できていません。

081では、075と同じ約5M Token列を使い、080と同じ5,000 step・同じ50M構造・同じseed・同じlearning-rate scheduleで学習します。batch size 8、context length 256なので、学習で見るToken数は約10.24Mとなり、同じ5M Token列を概ね二周見せる条件です。080の「10M unique tokensを一周」と081の「5M tokensを二周」を比較し、データの多様性と反復回数のどちらがvalidation性能と自然な日本語に効くかを調べます。強い教師モデルの蒸留は行いません。

仮説は、5M列を二周すると、075で得られた一般・会話・医療データを繰り返し最適化できるため、080と同程度のvalidation lossになる一方、重複学習による過学習で固定chat-testの自然さや生成多様性が伸びにくい、です。もし081が080を上回る場合は、限られたMacBook規模ではデータを増やす前に高品質データを複数周回する方が有効な可能性があります。逆に080が上回る場合は、重複を増やさずデータを広げる方を主線へ採用します。

成功条件は、5,000 stepを完走し、100 step間隔のmetricsと生成文、500 step間隔のcheckpoint metadata、summary、5領域評価、固定chat-test 48例を保存することです。良い出力だけでなく、空応答、文字崩れ、繰り返し、文脈無視も含めてGitHubへ保存します。

## 再現条件

モデルは080と同じRoPE・LayerNorm・SwiGLU、dim 576、12層、9 heads、context length 256、MLP倍率4、50,207,616 parametersです。Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab 4,096、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-5m-v1-train.bin`、4,999,958 Token、SHA-256は`54eb3fab617c94bda59899db4f78e6ac65665606219414a710e40cc8ccb8603c`です。general validationは`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。

設定ファイルは`configs/issue1-both-50m-pretrain-5m-5k.toml`、seed 42、batch size 8、最大5,000 step、eval/sample interval 100、checkpoint interval 500、eval batches 20、learning rate 3e-4から3e-5、warmup 500、weight decay 0.1です。設定ファイルのSHA-256は学習開始前に計算して追記します。元の`medilink_analysis`と医師国家試験原データは変更せず、small_llm側の既存加工済みToken列を読み取り専用で使います。

開始前に確認したSHA-256は、設定ファイルが`db5604539e1da48122f160c55cffa5eb1b46d690738f038ceab0ef2696e8f32b`です。新規Colabスクリプトは`py_compile`を通過しました。

ローカルでの再現コマンドは次のとおりです。

```bash
uv run python scripts/train_torch.py \
  --config configs/issue1-both-50m-pretrain-5m-5k.toml \
  --device mps
```

Colab T4が利用できる場合は、`scripts/colab_bootstrap_081.py`から同じ設定を実行し、終了後に`colab_package_081.py`で軽量成果物とbest checkpointを回収します。Colab失敗時は同じ条件のMPSへ切り替え、失敗ログも削除しません。

開始時点で`colab sessions`は`No active sessions found on server.`でした。Colab送信用bundleは`/tmp/small_llm-colab-081.tar.gz`、約6.7MB、SHA-256は`4565b8aaf5d32635d528aff631d7bffb6cba320acff98ec8df879e78ce89f0c0`です。元JSONL、医師国家試験原本、`medilink_analysis`はbundleに含めていません。

## 実験中の記録

開始前にGit commit、設定hash、入力hash、bundle hash、Colab session状態を記録します。学習中は100 step間隔のmetricsと生成文を保存し、1,000 step以上ノート更新を空けません。途中停止、OOM、warning、空応答もそのまま記録します。

Colab T4の新規session `exp081-both-50m-pretrain-5m-two-pass`は作成に成功し、bundle uploadと入力hash検証が完了しました。step 1はtrain loss 8.895750、validation loss 8.819675、PPL 6766.06、learning rate 6.0e-7、経過2.71秒でした。step 100はtrain loss 6.931328、validation loss 7.205592、PPL 1346.94、learning rate 6.0e-5、経過11.96秒、step 200はvalidation loss 6.941548、PPL 1034.37、learning rate 1.2e-4、経過22.05秒でした。step 300はvalidation loss 6.646078、PPL 769.76、learning rate 1.8e-4、経過33.20秒、step 400はtrain loss 5.919453、validation loss 6.347234、PPL 570.91、learning rate 2.4e-4、経過44.40秒でした。NaN、OOM、shape errorは発生しておらず、学習は継続中です。

step 500はtrain loss 5.476576、validation loss 6.112171、PPL 451.32、learning rate 3.0e-4、経過53.85秒でした。step 600はvalidation loss 5.843829、PPL 345.10、learning rate 2.9968e-4、経過68.77秒、step 700は5.709621、PPL 301.76、learning rate 2.9870e-4、経過80.15秒、step 800はtrain loss 4.542830、validation loss 5.551695、PPL 257.67、learning rate 2.9707e-4、経過90.00秒でした。step 900はvalidation loss 5.450522、PPL 232.88、learning rate 2.9480e-4、経過102.09秒、step 1,000はtrain loss 4.442537、validation loss 5.368098、PPL 214.45、learning rate 2.9189e-4、経過111.89秒、step 1,100はtrain loss 3.985813、validation loss 5.303004、PPL 200.94、learning rate 2.8837e-4、経過122.76秒でした。step 1,100時点でもvalidation lossは改善しており、学習は継続中です。

step 1,200はtrain loss 3.892087、validation loss 5.169767、PPL 175.87、learning rate 2.8424e-4、経過133.29秒でした。step 1,300はvalidation loss 5.151368、PPL 172.67、learning rate 2.7954e-4、経過143.30秒、step 1,400は5.080469、PPL 160.85、learning rate 2.7427e-4、経過153.45秒、step 1,500はtrain loss 3.697270、validation loss 4.999947、PPL 148.41、learning rate 2.6848e-4、経過165.46秒でした。

step 1,600はvalidation loss 4.977218、PPL 145.07、learning rate 2.6218e-4、経過178.05秒、step 1,700は4.929737、PPL 138.34、learning rate 2.5540e-4、経過190.06秒、step 1,800はtrain loss 3.580178、validation loss 4.908610、PPL 135.45、learning rate 2.4819e-4、経過201.11秒でした。step 1,900はvalidation loss 4.854780、PPL 128.35、learning rate 2.4057e-4、経過213.53秒、step 2,000はtrain loss 4.015949、validation loss 4.840270、PPL 126.50、learning rate 2.3258e-4、経過223.94秒でした。step 2,000時点でもvalidation lossは改善中で、過学習による明確な悪化はまだ見えていません。学習は継続中です。

step 2,100はtrain loss 3.623113、validation loss 4.790753、PPL 120.39、learning rate 2.2426e-4、経過234.99秒でした。step 2,200はvalidation loss 4.803434、PPL 121.93、learning rate 2.1566e-4、経過245.08秒と一時的に悪化しましたが、step 2,300は4.781965、PPL 119.34、learning rate 2.0681e-4、経過256.09秒、step 2,400はtrain loss 3.445386、validation loss 4.710782、PPL 111.14、learning rate 1.9775e-4、経過267.79秒、step 2,500はtrain loss 3.306795、validation loss 4.682791、PPL 108.07、learning rate 1.8854e-4、経過279.69秒でした。step 2,500では080の同じstepのvalidation loss 4.980713を下回っています。学習は継続中です。

step 2,600はtrain loss 3.700928、validation loss 4.681067、PPL 107.89、learning rate 1.7921e-4、経過292.31秒でした。step 2,700は4.652793、PPL 104.88、learning rate 1.6981e-4、経過304.26秒、step 2,800はtrain loss 3.158581、validation loss 4.621344、PPL 101.63、learning rate 1.6038e-4、経過316.11秒でした。step 2,900は4.611706、PPL 100.66、learning rate 1.5098e-4、経過328.00秒、step 3,000はtrain loss 3.626999、validation loss 4.599061、PPL 99.39、learning rate 1.4165e-4、経過339.77秒でした。

step 3,100はtrain loss 3.247999、validation loss 4.575391、PPL 97.07、learning rate 1.3243e-4、経過352.50秒、step 3,200はtrain loss 3.545044、validation loss 4.533112、PPL 93.05、learning rate 1.2337e-4、経過364.30秒でした。step 3,200では080の同じstepのvalidation loss 4.841980より明確に低く、現時点で反復学習条件が優勢です。学習は継続中です。

step 3,300はtrain loss 3.393661、validation loss 4.523731、PPL 92.18、learning rate 1.1452e-4、経過375.69秒でした。step 3,400は4.494060、PPL 89.48、learning rate 1.0590e-4、経過387.57秒、step 3,500はtrain loss 3.240968、validation loss 4.500024、PPL 90.02、learning rate 9.7582e-5、経過397.82秒でした。step 3,600は4.492153、PPL 89.31、learning rate 8.9587e-5、経過410.49秒、step 3,700は4.481474、PPL 88.36、learning rate 8.1960e-5、経過420.95秒、step 3,800はtrain loss 3.429658、validation loss 4.456258、PPL 86.16、learning rate 7.4737e-5、経過431.42秒でした。

step 3,900はtrain loss 3.086437、validation loss 4.445015、PPL 85.20、learning rate 6.7955e-5、経過442.30秒、step 4,000はtrain loss 3.184059、validation loss 4.436998、PPL 84.52、learning rate 6.1645e-5、経過454.07秒でした。step 4,000まで大きな過学習は観測されず、080との差も維持されています。学習は継続中です。

step 4,100はtrain loss 2.883411、validation loss 4.431080、PPL 84.02、learning rate 5.5838e-5、経過466.42秒でした。step 4,200は4.428928、PPL 83.84、learning rate 5.0563e-5、経過476.42秒、step 4,300は4.420510、PPL 83.14、learning rate 4.5846e-5、経過488.23秒、step 4,400はtrain loss 3.003122、validation loss 4.418502、PPL 82.97、learning rate 4.1710e-5、経過498.24秒でした。step 4,500はtrain loss 3.184942、validation loss 4.414973、PPL 82.68、learning rate 3.8174e-5、経過510.27秒でした。

step 4,600はtrain loss 3.129592、validation loss 4.401906、PPL 81.61、learning rate 3.5256e-5、経過523.49秒、step 4,700はtrain loss 2.901073、validation loss 4.389427、PPL 80.59、learning rate 3.2970e-5、経過535.47秒、step 4,800はtrain loss 3.083824、validation loss 4.388238、PPL 80.50、learning rate 3.1327e-5、経過546.15秒でした。step 4,900はtrain loss 3.406056、validation loss 4.371881、PPL 79.19、learning rate 3.0335e-5、経過558.68秒で最良を更新しました。step 5,000はtrain loss 3.005190、validation loss 4.376420、PPL 79.55、learning rate 3.0000e-5、経過571.34秒でした。最終stepでも最良からの悪化は小さく、学習後半に明確な過学習は見られません。

学習はColab T4・PyTorch 2.11.0+cu128・CUDA 12.8・AMP有効で5,000 stepを完走しました。パラメータ数は50,207,616、summary上の総時間は571.34秒です。最良checkpointはstep 4,900、validation loss 4.371880690256755、PPL 79.19242816373284です。best weightのSHA-256は`1e09a7386a630133503c12052f7701216d505cb3bca8765cade38c879ba5e8cb`、`best.json`は`4b6b56ad60730cc75a938dd8ef99aba6e713e03852043e8fe9175ef5d5c2813b`、`summary.json`は`73c18287af27bbcfd8eb8fa1a7faa229c7e87014e3241a7955724ac7ced393cc`、`metrics.jsonl`は`bf36605740eb6a18f3f983d147b0e95b3a1755bd7606ede640c6db4f00b35f8e`です。Colab軽量archiveは64ファイル、SHA-256は`d94218e2a140496bb5a49894e5f314244ba070fa961fb58fa202c83097582d9b`、best checkpoint archiveのSHA-256は`147eebc3fa0c6c93fd4566aec9035762bdd43c35d9f692579ced346dac8e3131`、checkpoint hash manifestは`artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/colab_checkpoint_manifest.json`です。

固定promptの生成では、step 0は080と同じ文字崩れでした。step 2,500では「私はその後の自分には一日も何度も大事にしました」のような文法的に崩れた長文が出ましたが、step 5,000ではprompt直後にEOSとなり、空応答へ戻りました。validation lossの改善は確認できても、生成の自然さと安定性はまだ不足しています。step 0からstep 5,000まで100 step間隔の生成本文はすべて保存しました。

次に080と同じ5領域validation、固定chat-test-v1 48例、生成全文を評価します。評価にはstep 4,900のbest checkpointを使います。

## 実験終了後の結果と解釈

実際に実行した評価コマンドは次のとおりです。

```bash
uv run python scripts/evaluate_torch.py domains \
  --config configs/issue1-both-50m-pretrain-5m-5k.toml \
  --checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt \
  --device cpu \
  --domain general=artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin \
  --domain conversation=artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin \
  --domain medical=artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin \
  --domain RPC=artifacts/tokens/issue1-real-persona-chat-validation.bin \
  --domain MRMP=artifacts/tokens/issue1-mrmp-validation.bin \
  --eval-batches 20 \
  --output artifacts/evaluations/issue1-both-50m-pretrain-5m-5k-domains.json
uv run python scripts/evaluate_torch.py chat \
  --config configs/issue1-both-50m-pretrain-5m-5k.toml \
  --checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt \
  --selection experiments/evaluation/chat-test-v1.json \
  --input artifacts/corpus/conversation-v1/test.jsonl \
  --device cpu --max-new-tokens 64 --seed 42 \
  --output artifacts/evaluations/issue1-both-50m-pretrain-5m-5k-chat-test-v1.json \
  --text-output artifacts/evaluations/issue1-both-50m-pretrain-5m-5k-chat-test-v1.txt
uv run python scripts/create_chat_review_template.py \
  --evaluation artifacts/evaluations/issue1-both-50m-pretrain-5m-5k-chat-test-v1.json \
  --output artifacts/evaluations/issue1-both-50m-pretrain-5m-5k-chat-review.json
```

080との差を比較すると、081のvalidation lossはgeneralが4.607053から4.371884へ-0.235169、conversationが2.703728から2.588882へ-0.114846、medicalが2.732444から2.573062へ-0.159381、RPCが2.649845から2.544832へ-0.105013となりました。MRMPだけは2.363312から2.365846へ+0.002534とわずかに悪化しましたが、4領域では反復学習が改善しました。075の5M・2,500 stepから081の5M・5,000 stepへ総学習Tokenを増やした条件では、全領域のvalidation lossが大きく改善しており、同じデータをもう一周見せることがこの規模では有効でした。

固定chat-test 48例では、081もEOS到達は48例中48例でした。平均生成Token数は080の8.8750から9.4792へ+0.6042、precisionは0.110874から0.148852へ+0.037979、recallは0.067281から0.078555へ+0.011275、全体F1は0.074119から0.094911へ+0.020792改善しました。short F1は0.081483から0.124037へ+0.042554、long F1は0.073293から0.100298へ+0.027005改善しました。一方、medium F1は0.067581から0.060397へ-0.007184、平均生成長も9.7500から7.7500へ短くなりました。081は長文化だけではなく、短い応答の重複率を改善しており、079の長文oversamplingより主目的に合っています。

それでも生成文は完成していません。事前学習の固定promptはstep 5,000で空応答へ戻り、chat-testにも「大量で、どうでしょう?」「男性には女性が好きなんですが、もう子供の女性は、」のような文脈を外した出力が残っています。したがって、反復学習はvalidation性能と簡易chat F1を改善する有望な手法ですが、自然な日本語の安定生成を保証しません。今後はSFTへ進む前に、EOSや文書境界を含む学習形式、会話データの混合比、学習途中での生成評価を重視します。

評価JSON、生成全文、人手レビュー用JSONのSHA-256は、領域評価が`26c094237c80fdece236cc30b2bfb66391efc357b3e78e20dec375d7dee0225e`、chat JSONが`af0c36e5361e5a5603aebb9c0405df787db87cd301a6c0e7decc4f1710f4ac52`、生成TXTが`b85b99b9ec1f7db74a6e6433d429e9d62a05ed9428d26f9fdf7d3249202cd0c5`、review JSONが`622d9cd981fe0a403427ddbebffb6d1b8ac29db5cfd6130580a8aafea9484b9e`です。

今回の判断は、081の「5M Tokenを二周相当学習する」条件を、現状の50Mモデルの主線として採用する、です。080の10M unique tokens一周より、同じ総学習Token予算でvalidation lossと全体chat F1が良くなりました。次は、081のbest checkpointからSFTへ進める前に、5M列の重複をさらに減らす品質改良、会話・医療・一般文書の比率、EOSを含む文書境界の扱いを一つずつ調べます。蒸留は使わず、フルスクラッチのデータ品質・反復回数・SFTの組合せで自然な日本語を伸ばします。

## 次に試すこと

081で080に近いか上回る結果が出た場合は、5M列をさらに高品質化し、会話・医療・一般文書の比率を一つずつ調整します。080が明確に上回る場合は、FineWeb2・Wikipediaなどの追加データを増やし、10Mから20M、さらにcontext length 512へ進みます。どちらの場合も、蒸留は主線へ入れず、フルスクラッチのデータ量・反復回数・自然な日本語生成の関係を優先して研究します。
