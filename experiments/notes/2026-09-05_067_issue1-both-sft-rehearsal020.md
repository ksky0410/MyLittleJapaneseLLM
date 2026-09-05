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

## 成功条件

3,000 stepをNaN、OOM、shape errorなく完走し、best checkpoint metadata、metrics、summary、step 0〜3,000の生成本文、共通5領域評価、固定chat-test 48例を保存することです。Colabを使った場合はGPU情報と軽量成果物のmanifestを回収し、重いcheckpoint本体はGit管理外のままSHA-256を記録します。失敗した場合も、失敗理由と次の切り替えを削除せずに残します。

## 実験中の記録

準備時点ではColab sessionは存在せず、`colab sessions`でactive sessionがないことを確認しました。22:06:44 JSTにsession `exp067-both-rehearsal020`のT4割り当てを試みましたが、Colab APIがHTTP 503 `Service Unavailable`を返し、sessionは作成されませんでした。bundle uploadとColab上の学習は発生していません。これまでのT4割り当て失敗と同じため、予定どおり同じbundleを用いたMPSへ切り替えます。学習中は1,000 step以内の間隔でloss、PPL、learning rate、経過時間、生成本文、警告、途中停止を追記します。

## 実験終了後の結果と解釈

ここへ実際のruntime、学習時間、best step、SFT/rehearsal/validation loss、5領域loss、chat-test結果、064〜066との差、代表生成、ratio 0.20を次の標準条件として採用するかどうかを追記します。生成本文は品質に関係なくGitHubへ保存します。

## 次に試すこと

ratio 0.10・0.20・0.25の結果から会話適合と一般性能のPareto点を選びます。次に比率の連続性が疑わしい場合は、batch内の行数を固定したままloss weightだけを変える条件、またはgradient accumulationでSFTとrehearsalのtoken予算を独立に制御する条件を実装します。その後、20Mで選んだ条件を50Mへ拡大し、Issue #1の会話効果がモデル容量でも再現するかを確認します。
