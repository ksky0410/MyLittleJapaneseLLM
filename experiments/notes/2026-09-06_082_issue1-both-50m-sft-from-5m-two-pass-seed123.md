# 実験082：081の反復事前学習基盤への標準SFT

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。実験081では、075と同じ約5M Token列を5,000 step、二周相当学習し、080の10M unique Token一周よりvalidation lossと固定chat F1が改善しました。082では、その081のbest checkpointへ標準SFTを行い、反復事前学習の改善が自然な会話応答へ移るかを確認します。強い教師モデルの蒸留、long response oversampling、reasoning生成データは使いません。

比較対象は実験078です。078と同じseed 123、SFT train・validation、rehearsal ratio 0.20、EOS loss weight 0.50、learning rate、3,000 step、固定chat-testを保ち、base checkpointだけを075の5M・2,500 stepから081の5M・5,000 stepへ変更します。仮説は、081の基盤がgeneral・conversation・medical・RPC・MRMPのvalidation lossと、short・medium・longの会話F1を全体的に押し上げ、自然な応答の崩れを減らすことです。

成功条件は、3,000 stepを完走し、100 step間隔のmetricsと生成文、500 step間隔のcheckpoint metadata、最良checkpoint、summary、5領域評価、固定chat-test 48例、人手レビュー用JSONを保存することです。長文を出すことではなく、履歴に対応した自然な日本語と全体F1を主な判断基準にします。

## 再現条件

モデルはRoPE・LayerNorm・SwiGLU、dim 576、12層、9 heads、context length 256、50,207,616 parametersです。SFT設定は`configs/issue1-both-50m-sft-from-5m-two-pass-seed123-3k.toml`、seed 123、batch size 8、最大3,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01です。設定ファイルのSHA-256は`c64652986d1b2b9efd75674902e59c2640e43ca8604ebaa3c42d2feffaed61d5`です。新規Colab実行・回収スクリプトは`py_compile`を通過しています。Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab 4,096、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

base checkpointは`artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt`、081 step 4,900、SHA-256は`1e09a7386a630133503c12052f7701216d505cb3bca8765cade38c879ba5e8cb`です。SFT trainは`artifacts/sft/issue1-both-balanced-v1/train.npz`、SHA-256は`645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、validationは`artifacts/sft/issue1-both-full-v1/validation.npz`、SHA-256は`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`、rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。設定ファイルのSHA-256は学習開始前に計算して追記します。

ローカルでの再現コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-50m-sft-from-5m-two-pass-seed123-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-3k \
  --samples-dir artifacts/samples/issue1-both-50m-sft-from-5m-two-pass-seed123-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

## 実験中の記録

Colab T4が利用できる場合は、`scripts/colab_bootstrap_082.py`から実行します。学習用bundleは`/tmp/small_llm-colab-082.tar.gz`、サイズは226 MiB（236,556,712 bytes）、SHA-256は`9571c667294c29018f5297e67c8db6587e3aae29850617c53a8a00ff69e7aa4c`です。開始前のColabセッションは存在しません。最初のuploadはHTTP 400で失敗しました。bundleを64 MiB以下の分割片で再送し、`scripts/colab_join_082_bundle.py`で結合後に同じSHA-256を検証する方式へ切り替えます。学習中は1,000 stepを超えてノート更新を空けず、metricsと固定生成を保存します。途中停止、警告、空応答、自然さの悪化も削除せず記録します。

## 実験終了後の結果と解釈

078と同じCPU・同じ5領域・同じ48例のchat-test・同じgeneration seed 42で評価します。081基盤の改善がSFT後にも残るか、またはSFTが会話データへ過適応して一般・医療文書を忘れるかを比較します。

## 次に試すこと

082で改善した場合は、081基盤を標準モデル候補として、SFTのデータ品質、EOS扱い、会話source比率を調べます。改善しない場合は、事前学習からSFTへの切り替え時のlearning rate、rehearsal ratio、SFTデータの重複を切り分けます。どちらの場合も、蒸留を主線にせず、自前データと反復学習で自然な日本語を伸ばします。
