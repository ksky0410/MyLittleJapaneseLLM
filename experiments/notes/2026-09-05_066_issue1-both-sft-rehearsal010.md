# 実験066：both-SFTへ一般日本語rehearsal 0.10を加える

## 開始前の計画

実施日は2026年9月5日、担当者はCodexです。Issue [#1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)は現在もOpenで、RealPersonaChatとMulti-Relational Multi-Party Chat Corpusを使い、一般日本語を保ちながら会話表現を学習し、pretrainingと会話SFTを分けて比較する方針を示しています。実験064ではRPCとMRMPを均等に混ぜたboth-SFT、実験065では同じboth-SFTにrehearsal ratio 0.25を加えました。065はgeneral・conversation・medical・RPC・MRMPのvalidation lossを改善した一方、固定chat-testのToken overlap F1が064の0.2297から0.1863へ低下しました。

実験066では、064・065と同じbase checkpoint、会話SFT train/validation、Tokenizer、モデル構造、seed、学習率、EOS loss weight、3,000 stepを使い、rehearsal ratioだけを0.10へ変更します。batch size 8のうちSFT 7例、一般日本語rehearsal 1例を使い、SFT lossに0.90、rehearsal lossに0.10を掛けて合算します。065の0.25より会話側の更新を強く残し、general lossを064より改善できるかを確認します。

仮説は、ratio 0.10なら065ほどgeneral lossは改善しなくても、064のchat-test F1低下を小さく抑え、RPC・MRMPのsource適合を維持しやすいことです。rehearsalが少な過ぎてgeneral lossが064とほぼ同じになる可能性、またSFT-onlyと同様に一般文書性能が崩れる可能性もあります。結果は単一指標で選ばず、5領域loss、EOS、生成長、Token overlap、生成本文を併せて比較します。

## 再現条件

実験開始時のHEADは`42de97a`です。これは実験065の成果物を保存したcommitです。作業ツリーには別途変更されたColab補助scriptが残っていますが、第066ではそれらを使用せず、`scripts/train_sft_torch.py`、Tokenizer、モデル実装はHEADと一致した状態で実行します。第065でColab T4の割当がHTTP 503となったため、今回は割当待ちを繰り返さず、ローカルMPSで実行します。元の`artifacts/corpus/conversation-v1`、医師国家試験データ、`/Users/koseki/projects/medilink_analysis`は変更しません。

base checkpointは実験050の`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`で、SHA-256は`326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`です。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。モデルはRoPE、LayerNorm、SwiGLU、dim 384、10層、6 heads、context length 256、19,308,032 parameterです。

会話trainは`artifacts/sft/issue1-both-balanced-v1/train.npz`、validationは`artifacts/sft/issue1-both-full-v1/validation.npz`です。trainはRPC 385,500、MRMP 385,490、合計770,990 response Token、64,423例です。validationは49,045例、738,660 response Tokenです。rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`で、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。

学習はbatch size 8、3,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 42、EOS loss weight 0.50、学習率schedule終点3,000 stepで実行します。sample promptは、raw形式によるEOS直後停止を避けるため、`<|startofconversation|><|speaker:DA|>こんにちは！<eos:3><|speaker:DC|>`のconversation形式にします。

実行コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-20m-sft-source-rehearsal010-mps-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal010-mps-3k \
  --samples-dir artifacts/samples/issue1-both-20m-sft-source-rehearsal010-mps-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.10 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

## 成功条件

3,000 stepをNaN、OOM、shape errorなく完走し、5領域の評価、固定chat-test 48例、step 0から3,000までの生成本文、metrics、summary、checkpoint metadataを保存することです。065と同じbase、会話データ、seed、学習率で比較できることも確認します。ratio 0.10が最適でない場合も、その結果を一般性能と会話適合の曲線上の一測定点として残します。

## 実験中の記録

開始前にconfig、base、会話NPZ、rehearsal Token列、Tokenizerのhashを確認します。学習中は1,000 step以内の間隔で総合loss、SFT loss、rehearsal loss、validation loss、PPL、学習率、経過時間、生成本文を追記します。MPSの異常、途中停止、生成の空出力、文脈に合わない出力も削除せずに記録します。

準備変更はcommit `94b84f1`としてpush済みです。`uv run pytest -q`は85件が通過しました。MPS学習は2026年9月5日に開始し、step 1では総合loss 4.3071、SFT loss 4.2788、rehearsal loss 4.5618、validation loss 4.7235、PPL 112.56、学習率5e-7でした。step 100では総合loss 4.0017、SFT loss 4.1391、rehearsal loss 2.7650、validation loss 4.0899、PPL 59.73、step 200ではvalidation loss 4.0793、PPL 59.10、step 300ではvalidation loss 4.0233、PPL 55.89、step 400ではvalidation loss 4.0057、PPL 54.91となりました。step 500では総合loss 4.3503、SFT loss 4.2190、rehearsal loss 5.5322、validation loss 3.9868、PPL 53.88、学習率4.7931e-5、経過時間578.25秒でした。第065の同step validation loss 3.9958より0.0090低い値です。step 500までNaN、OOM、shape errorは発生しておらず、conversation形式の生成サンプルもstep 0から500まで保存されています。学習は継続中です。

step 600では総合loss 3.5601、SFT loss 3.4857、rehearsal loss 4.2295、validation loss 3.9568、PPL 52.29、step 700ではvalidation loss 3.9386、PPL 51.35、step 800では3.9252、PPL 50.66、step 900では3.9144、PPL 50.12となりました。step 1,000では総合loss 3.3239、SFT loss 3.1979、rehearsal loss 4.4573、validation loss 3.8831、PPL 48.57、学習率4.0147e-5、経過時間1,086.86秒でした。064の同step validation loss 3.8755、065の3.8836と近い値で、現時点ではratio 0.10の明確な優位は確認できません。step 600から1,000までの生成サンプルも保存されています。NaN、OOM、shape errorは発生しておらず、学習は継続中です。

step 1,100では総合loss 3.5773、SFT loss 3.7043、rehearsal loss 2.4339、validation loss 3.8784、PPL 48.35、step 1,200では3.8651、PPL 47.71、step 1,300では3.8445、PPL 46.74、step 1,400では3.8277、PPL 45.96となりました。step 1,500では総合loss 3.8229、SFT loss 3.9315、rehearsal loss 2.8450、validation loss 3.8159、PPL 45.42、学習率2.8742e-5、経過時間1,378.91秒でした。064の同step validation loss 3.8088、065の3.8200の間に位置しており、step 1,500までのvalidation推移は事前の予想と整合します。step 1,100から1,500までの生成サンプルも保存されています。NaN、OOM、shape errorは発生しておらず、学習は継続中です。

step 1,600では総合loss 3.6393、SFT loss 3.4193、rehearsal loss 5.6194、validation loss 3.8068、PPL 45.01、step 1,700ではvalidation loss 3.7946、PPL 44.46、step 1,800では3.7857、PPL 44.07、step 1,900では3.7772、PPL 43.69となりました。step 2,000では総合loss 4.2741、SFT loss 4.3951、rehearsal loss 3.1844、validation loss 3.7682、PPL 43.30、学習率1.6982e-5、経過時間1,675.32秒でした。064の同step validation loss 3.7572、065の3.7605より高いものの、step 1,500から改善は継続しています。step 1,600から2,000までの生成サンプルも保存されています。NaN、OOM、shape errorは発生しておらず、学習は継続中です。

step 2,100では総合loss 3.5137、SFT loss 3.5630、rehearsal loss 3.0702、validation loss 3.7571、PPL 42.82、step 2,200では3.7524、PPL 42.62、step 2,300では3.7485、PPL 42.46、step 2,400では3.7395、PPL 42.08となりました。step 2,500では総合loss 3.3925、SFT loss 3.2799、rehearsal loss 4.4065、validation loss 3.7280、PPL 41.60、学習率8.2333e-6、経過時間1,962.41秒でした。064のbest validation loss 3.7129、065の3.7166にはまだ届いていませんが、step 2,000以降も改善しています。step 2,100から2,500までの生成サンプルも保存されています。NaN、OOM、shape errorは発生しておらず、学習は継続中です。

## 実験終了後の結果と解釈

ここへ実際のruntime、学習時間、best step、総合validation loss、SFT/rehearsal loss、5領域のloss、固定chat-testのEOS・生成長・Token overlap、064・065との差、代表的な生成を追記します。general lossだけで成功とせず、chat-test F1の変化と長い文脈の生成を確認します。生成本文は品質に関係なくGitHubへ保存します。

## 次に試すこと

ratio 0.10、0.25、rehearsalなしの差を比較し、会話性能と一般性能のPareto点を選びます。その後、必要ならratio 0.50を測り、条件が固まった段階で20Mから50Mへ拡大し、Grouped Query Attention、RoPEの設定、SwiGLU、学習率schedule、蒸留を一つずつ比較します。
