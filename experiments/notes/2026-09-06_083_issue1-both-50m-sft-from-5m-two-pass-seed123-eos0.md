# 実験083：081反復事前学習基盤へのEOS loss weight 0.0 SFT

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。実験082では、081の5M Token反復事前学習checkpointへ標準SFTを行い、078より5領域のvalidation lossをすべて改善しました。一方、chat-test全体F1は078とほぼ横ばいで、平均生成Token数は短くなり、short F1が下がる一方でlong F1が上がりました。083では、同じデータ・同じモデル・同じ学習率・同じ3,000 step・同じrehearsal ratioを保ち、EOS loss weightだけを0.50から0.0へ変更します。

仮説は、応答末尾EOSを強く学習させないことで、モデルが短い定型応答を早く終えず、履歴に沿った自然な応答をもう少し続けられることです。成功条件は、082より平均生成Token数とlong F1が改善し、領域validation lossを大きく悪化させないことです。EOSを全く学習しないため、終了不能や冗長化が起きる可能性も失敗条件として記録します。強いモデルの蒸留、reasoning生成データ、long response oversamplingは使いません。

## 再現条件

081のbest checkpoint `artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt`を初期値にします。モデルは50,207,616 parameters、dim 576、12層、9 heads、context length 256、RoPE・LayerNorm・SwiGLUです。SFT trainは64,423例、validationは49,045例、rehearsal ratioは0.20です。seedは123、batch sizeは8、最大3,000 step、learning rateは5e-5から5e-6、warmup 100、weight decay 0.01です。083の設定ファイルは`configs/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k.toml`、SHA-256は`cd2b7f0df1a01073f764556891b8707ba996f6b7272bf0db5efae5310e402a2f`です。

Tokenizer `artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`のSHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`、base checkpointのSHA-256は`1e09a7386a630133503c12052f7701216d505cb3bca8765cade38c879ba5e8cb`、base metadata `best.json`のSHA-256は`4b6b56ad60730cc75a938dd8ef99aba6e713e03852043e8fe9175ef5d5c2813b`です。SFT trainのSHA-256は`645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、validationのSHA-256は`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`、rehearsal Token列のSHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。共通train scriptのSHA-256は`c8fb40406ec74635ba63159f86fcd55ef71724edc7cb8ffda53453222640203e`です。

ローカルでの再現コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k \
  --samples-dir artifacts/samples/issue1-both-50m-sft-from-5m-two-pass-seed123-eos0-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.0 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

## 実験中の記録

学習前にColab bundleのサイズ・SHA-256とsession状態を追記します。学習中は100 step間隔のmetricsと生成文を保存し、1,000 stepを超えてノート更新を空けません。EOSが出なくなる、文が冗長になる、validation lossが悪化する、途中停止する場合も削除せず記録します。

実行用bundleは`/tmp/small_llm-colab-083.tar.gz`、236,747,311 bytes（226 MiB）、SHA-256は`7ae0a7caa91c33578ae145d187b1a0f027fbe2a4f4e65eb4275c98a7a84029fb`です。082で判明した依存漏れを修正し、`train_torch.py`とbase metadata `best.json`をbundleへ含めました。HTTP upload制限を避けるため64 MiB以下の分割片で送信し、`scripts/colab_join_083_bundle.py`で結合後にSHA-256を検証します。学習開始前のsessionは`No active sessions found on server.`でした。

本番SFTはT4・CUDA AMP有効で開始し、入力9件のハッシュ検証を通過しました。step 400のvalidation lossは3.615486、500は3.606614、600は3.618175、700は3.619773、800は3.669414でした。082の同じstepのvalidation lossはそれぞれおよそ3.52、3.496、3.469、3.475、3.472であり、083は開始直後から悪化しています。step 800のperplexityは39.228926です。EOS weight 0.0が生成の長さを伸ばす一方で、学習目標全体との不均衡を生んでいる可能性があります。原因を決めつけず、3,000 stepまで実行して最終生成と領域評価で確認します。

step 1,200のvalidation lossは3.613898、1,300は3.627349、1,400は3.602808、1,500は3.612270、1,600は3.572864でした。step 1,600のperplexityは35.618444です。途中で3.60台まで下がりましたが、082のstep 1,600は約3.36であり、083のvalidation差はまだ大きいままです。step 800の固定生成は「こんにちは!お願いします!よろしくお願いします!」となり、EOS weight 0.0で応答が長くなる仮説は支持されますが、validation悪化との交換条件になっています。

step 1,900のvalidation lossは3.551040、2,000は3.550312、2,100は3.508189、2,200は3.530277、2,300は3.523702でした。step 2,300のperplexityは33.909717です。後半に改善しているものの、082の同じ時点より約0.24高く、EOS lossを完全に外す条件はvalidation性能では不利なままです。最終stepまで実行し、best checkpointを保存します。

step 2,500のvalidation lossは3.527650、2,600は3.538727、2,700は3.510081、2,800は3.531154、2,900は3.538751、最終step 3,000は3.523709でした。最終perplexityは33.909980、経過時間は350.88秒です。監視用のColab CLIはstep 2,300付近で応答待ちtimeoutとなりましたが、Colab sessionは存続しており、metricsの再downloadで学習完走を確認しました。timeoutは学習失敗ではありません。082の最終validation loss 3.275936との差は+0.247773で、EOS weight 0.0はvalidation性能を明確に悪化させました。

Colab packageの軽量archiveは40ファイル、SHA-256は`3ccbe100f4b34bad026d260fed3330fada94760237677eb9c02d2927984e9c60`、best checkpoint archiveは`2a201493f66e300705f1f00b5c8e98f947fbdf01d0fd7a16c2ba4385be682d9e`、manifestは`17075316c82464f8254aaca02cd44fa8ae465a17833f61ebb8394ea61d93713b`です。best weightはstep 2,100、validation loss 3.508188581466675、perplexity 33.38773381450272、elapsed 247.78秒、SHA-256は`6a79b40ed1ccc00706dd0859d36d5edce490ef5b1faca8acf2c8c09b7fee0c85`です。peak allocated memoryは1,491,208,704 bytesでした。

step 1,500の固定生成は「こんにちわー!わがんは!???!いいえ。あなたは??????????????????? ??? ??!??? ?!?? なんですよ!!?」で、出力が長くなった代わりに文字崩れが増えました。step 3,000は「こんにちはー!よろしくお願いします!よろしくお願いします!」で、繰り返しが目立ちます。EOSを抑えるだけでは自然な日本語にならず、終了記号を含めた学習信号が品質に必要という反証になっています。

## 実験終了後の結果と解釈

082と同じCPU・同じ5領域・同じ48例chat-test・同じgeneration seed 42で評価します。領域loss、平均生成Token数、EOS到達数、short・medium・long F1を比較し、EOS loss weight 0.0が自然な日本語を改善したかを判断します。

## 次に試すこと

083が改善した場合はEOS weight 0.0または0.25を候補にし、5M Token列の反復回数を増やす実験へ進みます。悪化した場合は0.25または082の0.5へ戻し、10Mから20Mへデータを増やす条件と会話・一般・医療の混合比を調べます。どちらの場合も蒸留を主線にせず、自前データの量・品質・反復学習・終了記号の設計を優先します。
