# 実験067：both-SFTへ一般日本語rehearsal 0.20を加える

## 開始前の計画

実施日は2026年9月5日、担当者はCodexです。GitHubの[Issue #1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)は現在もOpenで、RealPersonaChatとMulti-Relational Multi-Party Chat Corpusを使い、一般日本語を保ちながら自然な会話表現を学習し、標準文・会話データ・SFT・rehearsalを分けて比較する方針を示しています。実験064ではRPCとMRMPを均等に混ぜたboth-SFT、実験065ではrehearsal ratio 0.25、実験066ではratio 0.10を同じbaseから測定しました。

実験067では064〜066と同じbase checkpoint、Tokenizer、RPC/MRMPのbalanced SFT train・validation、モデル構造、seed、学習率、EOS loss weight、3,000 stepを使い、rehearsal ratioだけを0.20へ変更します。batch size 8ではSFT 6例、rehearsal 2例となり、SFT lossに0.80、rehearsal lossに0.20を掛けて合算します。0.10はSFT 7例・rehearsal 1例、0.25はSFT 6例・rehearsal 2例ですので、0.20は両条件の間を埋めるだけでなく、rehearsal batch数が変わる境界も含む測定点です。

仮説は、ratio 0.20なら0.10より5領域のvalidation lossが改善し、0.25ほどchat-test F1を失わず、一般文書保持と会話応答のPareto上で実用的な中間点になることです。逆に、batchの丸めにより0.20と0.25の差が小さく、ratioを連続量として扱っても挙動が滑らかでない可能性があります。単一のlossやToken overlap F1だけで判断せず、general・conversation・medical・RPC・MRMPのloss、EOS、生成長、short・medium・long別F1、生成本文を比較します。

## 再現条件

実験開始前の実行コードとconfigの基準commitは`8d0bd5d`です。bundle hashと入力hashを記録したノート更新のみ、その後のcommit `3a74740`へ分離しています。configのSHA-256は`e657c439f4297e7339a4f14339254385e785be89c2ce8dfa056272a855dbaf6d`、base checkpointのSHA-256は`326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`、TokenizerのSHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。会話SFT trainは64,423例・response 770,990 TokenでSHA-256は`645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、validationは49,045例・response 738,660 TokenでSHA-256は`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`、rehearsal Token列のSHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。今回のColab bundleには元の会話JSONLや医師国家試験の原本を含めず、取得済みの加工済みNPZ・Token列と実行に必要なコードだけを含めます。元の`/Users/koseki/projects/medilink_analysis`と医師国家試験データは変更しません。

モデルはRoPE・LayerNorm・SwiGLU、dim 384、10層、6 heads、context length 256、19,308,032 parameterです。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`、baseは実験050の`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`です。会話SFT trainは`artifacts/sft/issue1-both-balanced-v1/train.npz`、validationは`artifacts/sft/issue1-both-full-v1/validation.npz`、rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`です。

学習はbatch size 8、3,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 42、EOS loss weight 0.50、学習率schedule終点3,000 stepで実行します。Colab GPUでは`--device auto`とし、割当が失敗した場合はその事実を記録して同じ条件のMPSへ切り替えます。学習中はconversation形式の固定promptを使い、step 0から3,000まで100 step間隔の生成本文を保存します。

実行コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-20m-sft-source-rehearsal020-colab-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal020-colab-3k \
  --samples-dir artifacts/samples/issue1-both-20m-sft-source-rehearsal020-colab-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device auto
```

Colabへ送るbundleは`/tmp/exp067_bundle.tar.gz`で、作成日時は2026年9月5日、サイズは約117 MiB、SHA-256は`d15d5e45d887b2cc1a4d056898158ac88db0d20dd77415fff4028ac264ce63d0`です。bundle内には067のconfig、学習script、`src` package、base checkpointとmetadata、Tokenizer、balanced SFTのtrain・validation NPZ、rehearsal Token列だけを含め、原文データは含めていません。

Colab割り当て失敗後のMPS実行では、次のコマンドを使います。Colab用configを読み込みますが、成果物の混同を避けるためcheckpointとsampleの出力先をMPS専用ディレクトリへ上書きします。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-20m-sft-source-rehearsal020-colab-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal020-mps-3k \
  --samples-dir artifacts/samples/issue1-both-20m-sft-source-rehearsal020-mps-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

## 成功条件

3,000 stepをNaN、OOM、shape errorなく完走し、best checkpoint metadata、metrics、summary、step 0〜3,000の生成本文、共通5領域評価、固定chat-test 48例を保存することです。Colabを使った場合はGPU情報と軽量成果物のmanifestを回収し、重いcheckpoint本体はGit管理外のままSHA-256を記録します。失敗した場合も、失敗理由と次の切り替えを削除せずに残します。

## 実験中の記録

準備時点ではColab sessionは存在せず、`colab sessions`でactive sessionがないことを確認しました。22:06:44 JSTにsession `exp067-both-rehearsal020`のT4割り当てを試みましたが、Colab APIがHTTP 503 `Service Unavailable`を返し、sessionは作成されませんでした。bundle uploadとColab上の学習は発生していません。これまでのT4割り当て失敗と同じため、予定どおり同じbundleを用いたMPSへ切り替えます。学習中は1,000 step以内の間隔でloss、PPL、learning rate、経過時間、生成本文、警告、途中停止を追記します。

Colab失敗を記録したcommitは`a74c7ef`、MPS fallback commandを記録したcommitは`6f8f2bf`です。MPS学習は同日22時台に開始し、step 1ではtrain loss 4.539920、SFT loss 4.371544、rehearsal loss 5.213422、validation loss 4.723595、PPL 112.5722、経過時間2.31秒でした。step 100ではvalidation loss 4.105101、PPL 60.6489、step 200では4.092865、PPL 59.9113、step 300では4.030184、PPL 56.2713、step 400では4.014373、PPL 55.3886となりました。step 500ではtrain loss 4.315105、SFT loss 4.265006、rehearsal loss 4.515504、validation loss 4.000267、PPL 54.6128、learning rate 4.7931e-5、経過時間209.07秒でした。step 600ではvalidation loss 3.967186、PPL 52.8356、step 700では3.936236、PPL 51.2254、step 800では3.924641、PPL 50.6349、step 900では3.913485、PPL 50.0731となりました。step 1,000ではtrain loss 3.506879、SFT loss 3.376545、rehearsal loss 4.028215、validation loss 3.889084、PPL 48.8661、learning rate 4.0147e-5、経過時間468.65秒でした。step 1,000までNaN、OOM、shape errorは発生せず、step 600から1,000までのconversation形式生成サンプルも保存されています。学習は継続中です。
step 1,100ではvalidation loss 3.880987、PPL 48.4720、step 1,200では3.881512、PPL 48.4975となり、1,200 step付近では一時的に横ばいでした。その後、step 1,300ではvalidation loss 3.852098、PPL 47.0918、step 1,400では3.836734、PPL 46.3738、step 1,500ではtrain loss 3.767353、SFT loss 4.053902、rehearsal loss 2.621154、validation loss 3.824488、PPL 45.8094、learning rate 2.8742e-5、経過時間750.49秒となりました。step 1,100から1,500までの生成サンプルも保存され、NaN、OOM、shape errorは発生していません。学習は継続中です。
step 1,600ではvalidation loss 3.810489、PPL 45.1725、step 1,700では3.802518、PPL 44.8139、step 1,800では3.788425、PPL 44.1867、step 1,900では3.775668、PPL 43.6266となりました。step 2,000ではtrain loss 4.446081、SFT loss 4.422170、rehearsal loss 4.541728、validation loss 3.765957、PPL 43.2051、learning rate 1.6982e-5、経過時間1,079.96秒でした。step 1,600から2,000までの生成サンプルも保存され、NaN、OOM、shape errorは発生していません。学習は継続中です。
step 2,100ではvalidation loss 3.757586、PPL 42.8449、step 2,200では3.753502、PPL 42.6703、step 2,300では3.750029、PPL 42.5223、step 2,400では3.740185、PPL 42.1058となりました。step 2,500ではtrain loss 3.527466、SFT loss 3.285070、rehearsal loss 4.497049、validation loss 3.733409、PPL 41.8214、learning rate 8.2333e-6、経過時間1,410.93秒でした。step 2,100から2,500までの生成サンプルも保存され、NaN、OOM、shape errorは発生していません。学習は継続中です。
step 2,600ではvalidation loss 3.728683、PPL 41.6243、step 2,700では3.726805、PPL 41.5462、step 2,800では3.722201、PPL 41.3553、step 2,900ではtrain loss 4.085155、SFT loss 4.047915、rehearsal loss 4.234115、validation loss 3.721274、PPL 41.3170となりました。最終step 3,000ではtrain loss 3.450162、SFT loss 3.414000、rehearsal loss 3.594808、validation loss 3.722987、PPL 41.3878、learning rate 5.0000e-6、経過時間1,744.54秒でした。最良checkpointはstep 2,900の`best.pt`で、SHA-256は`0e08765f841abd26445c9fc1160cc5cb7d7ff9c9d5f20f8aa16c9024d5ebeb2b`です。summary上の経過時間は1,744.84秒でした。step 3,000までNaN、OOM、shape errorは発生せず、step 2,600から3,000までの生成サンプルも保存されました。学習は完了し、これから評価へ移ります。

## 実験終了後の結果と解釈

学習は完了しました。共通5領域評価と固定chat-testを実行した後、runtime、学習時間、best step、SFT/rehearsal/validation loss、064〜066との差、代表生成、ratio 0.20を次の標準条件として採用するかどうかを追記します。生成本文は品質に関係なくGitHubへ保存します。

## 次に試すこと

ratio 0.10・0.20・0.25の結果から会話適合と一般性能のPareto点を選びます。次に比率の連続性が疑わしい場合は、batch内の行数を固定したままloss weightだけを変える条件、またはgradient accumulationでSFTとrehearsalのtoken予算を独立に制御する条件を実装します。その後、20Mで選んだ条件を50Mへ拡大し、Issue #1の会話効果がモデル容量でも再現するかを確認します。
