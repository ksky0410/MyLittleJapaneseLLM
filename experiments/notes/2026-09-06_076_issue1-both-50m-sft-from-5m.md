# 実験076：約5M Tokenで事前学習した50Mモデルへのresponse-only SFT

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。Issue #1の現代的な会話データを含む日本語混合コーパスで事前学習した50Mモデルへ、response-only SFTを適用します。医師国家試験由来のデータは一般・医療領域の保持を測るrehearsalと評価へ使いますが、医療専用モデルにはしません。元の`/Users/koseki/projects/medilink_analysis`とその原データは変更・削除しません。

実験075では、約500万Tokenで事前学習した50Mモデルが、約100万Token条件の073よりgeneral・conversation・medical・RPC・MRMPのvalidation lossをすべて改善しました。一方、固定chat-test-v1のToken overlap F1は073より低く、事前学習のlanguage modeling改善だけでは会話応答の改善が確認できませんでした。今回は075のbest checkpointを初期値にして、074と同じresponse-only SFT条件を適用します。074は073の約100万Token基盤からSFTしたため、075基盤でSFT後の会話応答がどこまで変わるかを比較できます。

仮説は、より十分な日本語基盤を持つ075へSFTすると、074と同じSFT条件でも固定chat-testのoverall・medium・long F1が改善し、短い応答だけでなく話題への適合性が上がることです。反対に、SFT後もF1が伸びない場合は、現行SFTデータ量・応答形式・Tokenizer/context lengthがボトルネックであり、基盤Token数を増やすだけでは不十分と判断します。SFTは一般文・会話・医療の基盤を失わないよう、rehearsal ratio 0.20を固定します。

## 使用するデータと再現条件

設定は[`configs/issue1-both-50m-sft-from-5m-rehearsal020-3k.toml`](../../configs/issue1-both-50m-sft-from-5m-rehearsal020-3k.toml)です。モデル構造は075と同じvocab 4,096、dim 576、12層、9 heads、context length 256、RoPE、LayerNorm、SwiGLU、実測50,207,616 parametersです。初期checkpointは`artifacts/checkpoints/issue1-both-50m-pretrain-5m-2p5k/best.pt`で、SHA-256は`71931b2c689c2fbaa31c8c92c022a21fac571894ec2993a59be48644794e5e17`です。

SFT trainは`artifacts/sft/issue1-both-balanced-v1/train.npz`、SHA-256 `645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、validationは`artifacts/sft/issue1-both-full-v1/validation.npz`、SHA-256 `fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`です。rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。Tokenizerは075と同じ`mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

学習条件はbatch size 8、最大3,000 step、evaluation・生成間隔100、checkpoint間隔500、AdamW、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 42、response-only loss、EOS loss weight 0.50、rehearsal ratio 0.20です。SFT train batchとrehearsal batchの損失を分けて保存します。生成サンプルは`<|startofconversation|><|speaker:DA|>こんにちは！<eos:3><|speaker:DC|>`形式で保存します。

予定コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-50m-sft-from-5m-rehearsal020-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-50m-sft-from-5m-rehearsal020-3k \
  --samples-dir artifacts/samples/issue1-both-50m-sft-from-5m-rehearsal020-3k \
  --lr-schedule-steps 3000 \
  --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 \
  --sample-template conversation \
  --sample-speaker-a DA \
  --sample-speaker-b DC \
  --device mps
```

Colab CLIでT4割当を先に試し、bundleには075 best checkpoint、加工済みSFTデータ、rehearsal Token列、Tokenizer、コード、設定だけを含めます。元JSONL、医師国家試験原本、`medilink_analysis`は含めません。503や割当上限で失敗した場合は、エラーとsession状態をノートへ残して同一条件のMPSへ切り替えます。

開始commitは`6b8adf4`で、GitHubの`origin/main`へpush済みです。設定ファイルのSHA-256は`42b378bef6c16cdd6335acf92fc6e6ab2afd5e17e24db0e63108d2b782c32a27`、bootstrapのSHA-256は`eb3473ec0428f51afda2d0bfdd40572b78dc1105c6bd3c9d9889c9ec586a67c9`です。Colab送信用bundleは`/tmp/small_llm-colab-076-XXXXXX.tar.gz`、236,430,874 bytes、SHA-256 `622c5e5586409f4c726abc6b274e1f3999737061bf363c220528be9f7586636b`です。bundleには075 best重み、入力hash検証に必要なmetadata、SFTデータ、rehearsal Token列、Tokenizer、コードと設定だけを含め、原JSONL・医師国家試験原本・`medilink_analysis`は含めていません。

## 成功・失敗の判定基準

3,000 stepをNaN、OOM、shape error、base checkpoint signature不一致なく完走し、100 step間隔のlossと生成文、500 step間隔のcheckpoint metadata、summaryを保存できれば実装上の成功とします。品質比較では074と同じ5領域、固定chat-test-v1、EOS到達率、平均生成長、short・medium・long別F1を記録します。lossやToken overlapだけで自然さ、医学的正確性、会話能力を断定しません。失敗や崩れた生成も削除せず保存します。

## 実験中の記録

この節にはColab試行、bundleのhash、MPS切り替え、開始時のparameter数、100 stepごとのSFT・rehearsal・validation loss、生成本文、500 stepごとのmetadata、警告、途中停止を追記します。075のbest checkpointとデータhashを上書きせず、実行backendが変わった場合も時系列で残します。

2026年9月6日、MPS学習前に`colab sessions`を実行し、`No active sessions found on server.`を確認しました。その後、`colab new --session exp076-both-50m-sft-from-5m --gpu T4`を実行しましたが、assignment endpointがHTTP 503 `Service Unavailable`を返して終了しました。Colab側のbundle upload、入力hash検証、SFT初期化、学習stepは発生していません。この失敗を成功実験と混ぜず、同一条件のMPSへ切り替えます。

同日、MPSで076 SFTを開始しました。step 1のtrain lossは4.183418、SFT train lossは4.239130、rehearsal train lossは3.960572、validation lossは4.342213、PPL 76.88、learning rateは5.0e-7、経過4.12秒でした。実測parameter数は075と同じ50,207,616で、SFT設定はrehearsal ratio 0.20、EOS loss weight 0.50、lr schedule終点3,000 stepです。NaN、OOM、shape error、base checkpoint signature不一致はありません。学習を継続します。

## 実験終了後の結果と解釈

完走後に実際のruntime、最良・最終loss、SFTとrehearsalの損失、学習時間、best checkpoint hash、5領域評価、固定chat評価、生成本文を追記します。074との差は、初期checkpointとSFT条件を分けて記載し、075基盤の効果とSFT条件の効果を混同しません。

## 次に試すこと

076で075基盤のSFT効果を確認した後、会話データの長い応答比率を変更するか、現代的なinstruction形式と通常pretrainingの混合比率を一つだけ変更します。SFT後も長文会話が改善しない場合は、context length 512への拡張を別実験として検証します。
