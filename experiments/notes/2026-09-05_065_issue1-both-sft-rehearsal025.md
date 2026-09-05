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

step 1,100では総合loss 3.7308、SFT loss 3.7338、rehearsal loss 3.7217、validation loss 3.8756、PPL 48.21、step 1,200ではvalidation loss 3.8758、PPL 48.22となり、一時的に横ばいでした。step 1,300ではvalidation loss 3.8472、PPL 46.86、step 1,400では3.8319、PPL 46.15へ改善し、step 1,500では総合loss 3.6781、SFT loss 4.0355、rehearsal loss 2.6058、validation loss 3.8200、PPL 45.61、学習率2.8742e-5、経過時間1,711.39秒でした。step 1,100から1,500までの生成サンプルも保存されています。NaN、OOM、shape errorは発生しておらず、学習は継続中です。

step 1,600では総合loss 3.6617、SFT loss 3.4760、rehearsal loss 4.2189、validation loss 3.8056、PPL 44.95、step 1,700ではvalidation loss 3.7974、PPL 44.58、step 1,800では3.7833、PPL 43.96、step 1,900では3.7702、PPL 43.39となりました。step 2,000では総合loss 4.4364、SFT loss 4.4198、rehearsal loss 4.4863、validation loss 3.7605、PPL 42.97、学習率1.6982e-5、経過時間2,304.31秒でした。064の同step validation loss 3.7572との差は0.0033で、現時点では明確な差ではありません。step 1,600から2,000までの生成サンプルも保存されています。NaN、OOM、shape errorは発生しておらず、学習は継続中です。

step 2,100では総合loss 3.4993、SFT loss 3.6193、rehearsal loss 3.1393、validation loss 3.7527、PPL 42.64、step 2,200では3.7484、PPL 42.45、step 2,300では3.7453、PPL 42.32、step 2,400では3.7353、PPL 41.90となりました。step 2,500では総合loss 3.5675、SFT loss 3.2803、rehearsal loss 4.4290、validation loss 3.7285、PPL 41.62、学習率8.2333e-6、経過時間2,892.35秒でした。064のbest validation loss 3.7129との差は0.0156です。step 2,100から2,500までの生成サンプルも保存されています。NaN、OOM、shape errorは発生しておらず、学習は継続中です。

step 2,600では総合loss 3.6398、SFT loss 2.9995、rehearsal loss 5.5606、validation loss 3.7239、PPL 41.42、step 2,700では3.7225、PPL 41.37、step 2,800では3.7175、PPL 41.16、step 2,900では総合loss 4.0852、SFT loss 4.0473、rehearsal loss 4.1988、validation loss 3.7166、PPL 41.12となりました。step 3,000では総合loss 3.4468、SFT loss 3.4160、rehearsal loss 3.5392、validation loss 3.7183、PPL 41.20、学習率5.0000e-6、経過時間3,469.65秒でした。最良checkpointはstep 2,900で、best.ptのSHA-256は`2e02ba55b2b1f9d83fdc0e16841a9c1a135720b6f355c6e4de536a4f3f6993c8`です。step 3,000までNaN、OOM、shape errorはなく、step 2,600から3,000までの生成サンプルも保存されました。

## 実験終了後の結果と解釈

学習はMPSで完走しました。PyTorchは2.14.0、AMPは無効、parameter数は19,308,032、経過時間は3,469.99秒でした。best stepは2,900、best validation lossは3.7166076660、PPLは41.1246486678、最終step 3,000のvalidation lossは3.7183266044、PPLは41.1954001948でした。best.ptのSHA-256は`2e02ba55b2b1f9d83fdc0e16841a9c1a135720b6f355c6e4de536a4f3f6993c8`です。summaryは`artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal025-colab-3k/summary.json`、metricsは同ディレクトリの`metrics.jsonl`、step 500間隔のcheckpoint metadataとstep 0〜3,000の生成本文は同じcheckpoint・samplesディレクトリに保存しました。重い`.pt`本体はGit管理外ですが、metadataとhashは管理します。

共通5領域をCPUで20 batchずつ評価した結果は、generalがvalidation loss 5.4092 / PPL 223.44、conversationが2.8855 / 17.91、medicalが3.1909 / 24.31、RPCが2.8890 / 17.98、MRMPが2.3281 / 10.26でした。064のboth-SFTと比べると、generalは0.8341、conversationは0.4448、medicalは0.3779、RPCは0.4592、MRMPは0.4539だけlossが下がりました。単純なrehearsalによるgeneral保持だけでなく、今回の5領域すべてで改善しています。これはrehearsalが会話SFTの更新を抑えたことに加え、064と065で同じbaseから別々に3,000 step学習したこと、評価が固定20 batchのサンプルであることの影響も含みます。したがって、rehearsalが全領域を必ず改善すると一般化せず、次の比率比較で再確認します。評価JSONのSHA-256は`2d4e07dd0ec90b569a36d3204ad70676e42e5c0a59413b1fef23f6f9a2539200`です。

固定chat-test v1の48例では、EOS到達が48/48、平均生成長が10.5625 Token、Token overlapのprecisionが0.2321、recallが0.1943、F1が0.1863でした。stratum別F1はshort 0.2952、medium 0.1329、long 0.1310です。064のF1 0.2297から0.0433下がり、平均生成長は同じ10.5625 Tokenでした。source別ではMRMP 24例が平均9.17 Token・F1 0.2079、RPC 24例が平均11.96 Token・F1 0.1648でした。つまり、rehearsal ratio 0.25は一般・会話validation lossを大きく改善しましたが、今回の固定chat-testのToken overlap F1は低下しました。短い定型応答の一致と長い文脈の自然さを分離する必要があるため、F1だけで065を失敗とは判定しません。評価JSONのSHA-256は`567ae89a6efa2d3fd10f17641587679b10929fc6587bafee85ff9135790ad926`、TXTのSHA-256は`f7a6c7e105f418d63c3634a3fc09e2d30438b7f731f91e9a562907b0850f9ab3`です。

代表例では、MRMPの`こんにちは`に`こんにちは`と返せましたが、`えー`への返答が`笑笑笑`、長いRPC文脈への返答が`お願いします! 私も一緒にお願いします、、そちらはどうですか?`や`スマホは、マスクを作っちゃうんですよね。でも、最近スパイスを使ったりするんですよね。`となるなど、文法的には日本語でも文脈適合が弱い出力が残っています。学習中のconversation形式サンプルは、064のraw形式より入力分布に近い条件で記録できました。全31個のstep別生成文は`artifacts/samples/issue1-both-20m-sft-source-rehearsal025-colab-3k/`に保存し、これらの崩れた出力も削除していません。

事前の「rehearsal ratio 0.25でgeneralを改善し、会話性能をおおむね維持する」という仮説は、generalだけでなく全5領域のvalidation loss改善という点では支持されましたが、chat-test F1を維持するという点では外れました。今回の差は、rehearsalでモデル全体の次Token性能を回復できる一方、応答の表面的な一致やsource固有の短文適合を犠牲にする可能性を示しています。単一seed、20M級、3,000 step、固定48例という制約があるため、ratio 0.25を最適値とはせず、一般文書と会話の両方を確認できる基準条件として扱います。

## 次に試すこと

ratio 0.25で一般文書保持と会話適合のバランスが改善すれば、次は同条件でratio 0.50を比較します。会話適合が大きく落ちる場合はratio 0.10へ下げるか、rehearsal Token列のsource配分を見直します。条件が固まった後、20Mから50Mへ拡大し、現代的な正規化・位置表現・Grouped Query Attention・SwiGLU・蒸留を一つずつ比較します。
