# 実験065：both-SFTへ一般日本語rehearsalを加える

## 開始前の計画

実施日は2026年9月5日、担当者はCodexです。Issue #1で進めている「一般日本語を保ちながら会話データを追加する」方針と、実験064で得たRPC・MRMP混合SFTを引き継ぎます。実験064では、両sourceをresponse Token数で均等に混ぜたSFTにより、RPCとMRMPのvalidationがそれぞれ単独条件と相手条件の中間へ入り、固定chat-testのF1も単独条件を上回りました。一方、general validationと長い文脈の生成を十分に保てたかはまだ確認が必要です。

実験065では、実験064と同じbase checkpoint、会話SFT train/validation、Tokenizer、モデル構造、seed、batch size、学習率、EOS loss weight、3,000 stepを使い、一般日本語Token列のrehearsal lossだけを追加します。rehearsal ratioは0.25とし、batch size 8のうちSFT 6例、一般日本語rehearsal 2例を使います。SFT lossに0.75、rehearsal lossに0.25を掛けて合算する実装です。比較を直接にするため、baseは064の学習後ではなく、064と同じ実験050のboth-pretraining best checkpointへ戻します。これにより、064との差はrehearsal objectiveの有無になります。

仮説は、rehearsal ratio 0.25で一般文書のvalidation lossが064より下がり、medicalやconversationの大幅な悪化を避け、RPC・MRMPのsource適合と固定chat-testのF1をおおむね維持することです。rehearsalを入れることでSFTの更新量が減り、会話F1やMRMPの短文適合が下がる可能性もあります。両方の結果を成功・失敗で単純化せず、一般文書の保持と会話適合のトレードオフとして記録します。

## 再現条件

実験064完了後の基準commitは`81aecb7`です。実験065ではrehearsal 0.25用config、Colab実行wrapper、軽量成果物の回収scriptを追加し、全テストとhash確認を行ってから学習します。元の`artifacts/corpus/conversation-v1`、医師国家試験データ、`/Users/koseki/projects/medilink_analysis`は変更しません。

base checkpointは実験050の`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`で、SHA-256は`326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`です。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。モデルはRoPE、LayerNorm、SwiGLU、dim 384、10層、6 heads、context length 256、19,308,032 parameterです。

会話trainは`artifacts/sft/issue1-both-balanced-v1/train.npz`、validationは`artifacts/sft/issue1-both-full-v1/validation.npz`です。trainはRPC 385,500、MRMP 385,490、合計770,990 response Token、64,423例です。validationは49,045例、738,660 response Tokenです。rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`で、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。rehearsalは通常のnext-token lossで、会話側はloss maskが1のresponse本文と末尾EOSを対象にします。

学習はbatch size 8、3,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 42、EOS loss weight 0.50、学習率schedule終点3,000 stepで実行します。学習中のsample promptは、raw形式によるEOS直後停止を避けるため、`<|startofconversation|><|speaker:DA|>こんにちは！<eos:3><|speaker:DC|>`のconversation形式にします。Colab T4を第一候補とし、割当失敗時は失敗を記録したうえでMPSへ切り替えます。

Colab上で実行するコマンドはwrapperから次の形になります。

```bash
python scripts/train_sft_torch.py \
  --config configs/issue1-both-20m-sft-source-rehearsal025-colab-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal025-colab-3k \
  --samples-dir artifacts/samples/issue1-both-20m-sft-source-rehearsal025-colab-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.25 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device auto
```

## 成功条件

3,000 stepをNaN、OOM、shape errorなく完走し、一般・会話・医療・RPC・MRMPの5領域、固定chat-test 48例、step 0から3,000までの生成本文、metrics、summary、checkpoint metadataを保存することです。実験064と同じbase、会話データ、seedで比較可能なことを確認します。rehearsalによりgeneralが改善しなくても、その結果を含めて記録します。

## 実験中の記録

学習開始前にconfig・wrapper・rehearsal Token列・会話NPZ・baseのhashを記録します。学習中は少なくとも1,000 step以内の間隔でstep、SFT loss、rehearsal loss、総合loss、validation loss、PPL、学習率、経過時間、runtime、生成本文を追記します。Colabの割当失敗、bundle hash不一致、引数ミス、途中停止も削除せずに残します。

準備変更はcommit `d570f4a`としてpush済みです。Colabへ送るbundleは`/tmp/exp065_bundle_v2.tar.gz`、115MB、SHA-256 `0e6d751ba4f8bcdf11f1a098e41dc96f03f719795c5d58387f0ac200961133d9`です。bundleにはbase checkpoint、Tokenizer、both-SFTのtrain/validation NPZ、rehearsal Token列、config、学習script、Colab wrapper、必要なPython packageを含め、元のJSONLや医師国家試験の原本は含めていません。

Colab session `exp065-both-rehearsal025`のT4割当を試みましたが、2026年9月5日にHTTP 503 `Service Unavailable`で失敗しました。sessionは作成されず、bundle upload、Colab上の学習、既存成果物の変更は発生していません。`colab sessions`でもactive sessionがないことを確認しました。実験063でも同じT4割当失敗が続いているため、今回はこの失敗を記録したうえで同じ条件のローカルMPSへ切り替えます。

Colab失敗後、同じ入力と条件でMPS学習を開始しました。step 1では総合loss 4.5820、SFT loss 4.3715、rehearsal loss 5.2134、validation loss 4.7236、PPL 112.57、学習率5e-7でした。step 100では総合loss 4.0665、SFT loss 4.2312、rehearsal loss 3.5727、validation loss 4.1039、PPL 60.58、step 200ではvalidation loss 4.0892、PPL 59.69、step 300ではvalidation loss 4.0258、PPL 56.03、step 400ではvalidation loss 4.0110、PPL 55.20となりました。step 500では総合loss 4.3170、SFT loss 4.2584、rehearsal loss 4.4925、validation loss 3.9958、PPL 54.37、学習率4.7931e-5、経過時間461.20秒でした。step 500までNaN、OOM、shape errorは発生しておらず、conversation形式の生成サンプルもstep 0から500まで保存されています。学習は継続中です。

step 600では総合loss 3.6279、SFT loss 3.7110、rehearsal loss 3.3788、validation loss 3.9626、PPL 52.60、step 700では総合loss 4.4477、SFT loss 4.1754、rehearsal loss 5.2646、validation loss 3.9320、PPL 51.01、step 800ではvalidation loss 3.9208、PPL 50.44、step 900ではvalidation loss 3.9082、PPL 49.81となりました。step 1,000では総合loss 3.5265、SFT loss 3.3675、rehearsal loss 4.0036、validation loss 3.8836、PPL 48.60、学習率4.0147e-5、経過時間1,124.23秒でした。064の同step validation loss 3.8755との差は0.0081で、現時点ではrehearsalによる改善はまだ見えていません。NaN、OOM、shape errorは発生しておらず、step 600から1,000までの生成サンプルも保存されています。学習は継続中です。

## 実験終了後の結果と解釈

ここへ実際のruntime、学習時間、best step、総合validation loss、SFT/rehearsal loss、5領域のloss、固定chat-testのEOS・生成長・Token overlap、064との差、代表的な生成を追記します。general lossの改善だけで会話性能が保たれたとは判断せず、source別lossと生成本文を併せて確認します。生成本文は品質に関係なくGitHubへ保存します。

## 次に試すこと

ratio 0.25で一般文書保持と会話適合のバランスが改善すれば、次は同条件でratio 0.50を比較します。会話適合が大きく落ちる場合はratio 0.10へ下げるか、rehearsal Token列のsource配分を見直します。条件が固まった後、20Mから50Mへ拡大し、現代的な正規化・位置表現・Grouped Query Attention・SwiGLU・蒸留を一つずつ比較します。
