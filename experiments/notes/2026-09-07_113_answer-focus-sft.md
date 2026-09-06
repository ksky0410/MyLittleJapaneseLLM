# 実験113：FineWeb2追加事前学習後のanswer-focus医療SFT

## 実験前の計画

### 目的

実験111でFineWeb2 Edu Japaneseを追加して得た50M日本語モデルを初期値にし、実験112と同じ一般会話・通常医療SFTへ、医師国家試験の正解記号だけを短く返すanswer-focus例を追加する。一般会話の自然さを保ちながら、医療QAで長い誤説明を生成する傾向と正答率を改善できるかを調べる。

### 仮説

実験112では医療回答の形式抽出は162/162だった一方、正解は19/162に留まり、誤った選択肢と理由を長く生成する例が多かった。通常医療回答に加えて「正解はaです。」の短い正解例を同じ問題について学習すれば、選択肢記号を先に正しく選ぶ分布が強まり、医療の完全一致率が改善すると予想する。ただし、answer-focusの重複が医療学習へ偏りすぎると、理由説明の長さや一般会話のlossが悪化する可能性がある。answer-focusが一般会話の自然さを直接改善するとは予想しない。

### 比較条件

実験112と次の条件を揃える。

- 初期checkpoint：実験111 raw best、step 10,000
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- Tokenizer：`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`
- 一般データ：`artifacts/sft/issue1-quality-aware-770k-each-v1/train.npz`
- 通常医療データ：`artifacts/sft/medical-qb-sft-v1/train.npz`
- 追加医療データ：`artifacts/sft/issue1-medical-answer-focus-v1/train.npz`
- SFT学習：batch size 8、8,000 step、AdamW、learning rate 2e-5から2e-6、warmup 200、weight decay 0.01、seed 113
- rehearsal：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、ratio 0.2
- validation：実験112と同じ通常一般49,045例・通常医療162例の結合validation。answer-focus validationは学習に使わず、別条件の評価に用いる
- 差分：実験112の通常医療2,945例に、同じ問題のanswer-focus 2,945例を追加する

### 入力のハッシュ

学習開始前に次のSHA-256を計算し、実行結果にも残す。

- 設定：`configs/issue1-fineweb-new-pretrain-answer-focus-sft-runpod-8k.toml`、SHA-256 `45228695fec211b576c93a3c46b688e0a49615dc3e645a2b7c789fab3204780c`
- 学習コード：`scripts/train_sft_torch.py`、SHA-256 `bc78ec94a7f74399d049ce4d1f6a22b446437a90b8e855bf64233b935267974e`
- 初期checkpoint：実験111 raw best、SHA-256 `6957aaab539af1d6924d5c43a0c44a057a356c35dbac79c49fbe2279962468b9`
- 通常医療を含む追加SFT train：`artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz`、SHA-256 `99fc5e82cefc7efd7e4eb69bb5250d794526313c4ae6e54eeee862673100b262`
- 通常医療を含む追加SFT train manifest：`artifacts/sft/issue1-general-medical-answer-focus-v1/train.manifest.json`、SHA-256 `b26112a5f297b4d2c3cda9ca0106e3d980c8438d1270cef7fa18dedf4c79efb1`
- 通常validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal Token列：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`

### 使用コマンド

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_sft_torch.py \
  --config configs/issue1-fineweb-new-pretrain-answer-focus-sft-runpod-8k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-50m-pretrain-fineweb-new-shards-runpod-10k/best.pt \
  --train-data artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz \
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-fineweb-new-pretrain-answer-focus-sft-runpod-8k \
  --samples-dir artifacts/samples/issue1-fineweb-new-pretrain-answer-focus-sft-runpod-8k \
  --device cuda --sample-template conversation --sample-speaker-a DA --sample-speaker-b DC \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin --rehearsal-ratio 0.2
```

### 成功・失敗の判定

NaN、OOM、shape errorなく8,000 stepを完走し、250 stepごとのvalidation lossと生成文を保存する。実験112と同じ評価コードでFineWeb2、general、conversation、medicalのloss、一般会話48例、医療162例を測定する。医療の完全一致数、正解抽出数、誤答理由の反復を確認する。一般会話のEOS到達とF1が実験112から大きく悪化した場合、answer-focus条件は主線へ採用しない。

## 学習中の記録

学習開始後は、少なくとも1,000 step以内の間隔でvalidation loss、learning rate、経過時間、異常、固定promptの生成文を追記する。stepごとの生成ファイルとmetricsは削除せず保存する。

### 2026-09-07：step 1〜250

Runpod A40上で実験111 raw bestから学習を開始した。step 1のvalidation lossは2.913260、perplexityは18.4167、learning rateは1.0e-7、経過時間は0.93秒だった。step 250ではvalidation loss 2.803733、perplexity 16.5061、learning rate 1.9998e-5、経過時間21.51秒となった。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 500〜1,500

step 500のvalidation lossは2.810042、step 750は2.821706、step 1,000は2.822412、step 1,250は2.814637、step 1,500は2.805546だった。step 250の2.803733をまだ更新しておらず、学習率はstep 1,500で1.8796e-5、経過時間は124.18秒だった。answer-focus追加による異常なloss発散はなく、NaN、OOM、shape errorも発生していない。

### 2026-09-07：step 1,750〜2,250

step 1,750のvalidation lossは2.805113、step 2,000は2.812463、step 2,250は2.806176だった。最良値は依然としてstep 250の2.803733で、step 2,250時点のlearning rateは1.7105e-5、経過時間は185.74秒である。実験112の同じ時点よりvalidation lossは高めだが、answer-focus追加が学習を壊す異常は見られない。

### 2026-09-07：step 2,500〜2,750

step 2,500のvalidation lossは2.805033、step 2,750は2.816896だった。step 2,500でstep 250に近い値まで戻ったが、最良値の更新はない。step 2,750時点のlearning rateは1.5659e-5、経過時間は226.18秒で、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 3,000〜3,250

step 3,000のvalidation lossは2.810933、step 3,250は2.800252だった。step 3,250で最良値を更新し、実験112のstep 3,250付近の2.786897にはまだ届いていない。step 3,250時点のlearning rateは1.4025e-5、経過時間は266.63秒である。学習は安定して継続している。

### 2026-09-07：step 3,500〜3,750

step 3,500のvalidation lossは2.780917、step 3,750は2.782681だった。step 3,500で最良値を2.780917まで更新し、実験112のbest 2.751857との差は0.029060である。step 3,750時点のlearning rateは1.2268e-5、経過時間は307.80秒で、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 4,000〜4,250

step 4,000のvalidation lossは2.782547、step 4,250は2.784560だった。最良値はstep 3,500の2.780917で変わらず、step 4,250時点のlearning rateは1.0460e-5、経過時間は348.04秒だった。answer-focus追加による異常停止はなく、学習を継続する。

### 2026-09-07：step 4,500〜5,000

step 4,500のvalidation lossは2.781147、step 4,750は2.772061、step 5,000は2.772423だった。step 4,750で最良値を2.772061まで更新し、実験112の最良2.751857との差は0.020204へ縮まった。step 5,000時点のlearning rateは7.8119e-6、経過時間は408.65秒であり、学習は安定している。

### 2026-09-07：step 5,250〜5,500

step 5,250のvalidation lossは2.763669、step 5,500は2.767797だった。step 5,250で最良値を2.763669まで更新し、実験112の最良2.751857との差は0.011812となった。step 5,500時点のlearning rateは6.1929e-6、経過時間は449.42秒だった。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 5,750〜6,000

step 5,750のvalidation lossは2.763231、step 6,000は2.763040だった。最良値はstep 6,000の2.763040で、実験112の最良2.751857との差は0.011183である。step 6,000時点のlearning rateは4.7681e-6、経過時間は490.08秒だった。学習は異常なく継続している。

### 2026-09-07：step 6,250〜6,500

step 6,250のvalidation lossは2.763616、step 6,500は2.761822だった。step 6,500で最良値を2.761822へ更新し、実験112の最良との差は0.009965である。step 6,500時点のlearning rateは3.5952e-6、経過時間は530.76秒で、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 6,750〜7,000

step 6,750のvalidation lossは2.762432、step 7,000は2.759352だった。step 7,000で最良値を2.759352へ更新し、実験112の最良との差は0.007495となった。step 7,000時点のlearning rateは2.7216e-6、経過時間は571.36秒であり、学習は安定している。

### 2026-09-07：step 7,250〜7,750

step 7,250のvalidation lossは2.756807、step 7,500は2.758019、step 7,750は2.759397だった。step 7,250で最良値を2.756807まで更新し、実験112の最良との差は0.004950となった。step 7,750時点のlearning rateは2.0460e-6、経過時間は632.81秒で、学習は完走直前まで安定している。

## 実験終了後の記録

学習完了後に、最良checkpointのstepとSHA-256、最終loss、学習時間、最大GPUメモリ、評価JSON/TXTのSHA-256、実験112との比較、仮説の判定、次に試す変更を追記する。checkpoint本体はGitへ追加せず、metadataとハッシュだけを記録する。

実験113はNaN、OOM、shape errorなく8,000 stepを完走した。最良checkpointはstep 7,250で、validation lossは2.756807、perplexityは15.7495だった。step 8,000のvalidation lossは2.757623で、最良値は後半のstep 7,000〜7,250に現れた。最良checkpointの重みSHA-256は`7cc2c9df032f0d2a19094a80aebb4b5da700f9a3f0d2500475b66295d1a1911c`、学習時間は653.49秒、peak allocated memoryは1,490,586,112 bytesだった。

領域別validation lossはFineWeb2 2.926402、general 4.052437、conversation 2.007372、medical 1.875708だった。実験112と比べると、FineWeb2は0.001484悪化、generalは0.007616改善、conversationは0.000772悪化、medicalは0.001004改善した。answer-focus例を追加しても通常validation全体を大きく壊さず、generalとmedicalにはわずかな改善が見られた。

一般会話48例はEOS到達48/48、平均生成9.10 Token、Token overlap F1 0.254858だった。実験112のEOS 48/48、平均8.42 Token、F1 0.222845と比べ、生成長は0.69 Token伸び、F1は0.032012改善した。ただし、生成本文は「わかります」「そうですね」「そうなんですね」のような短い応答が中心で、直前の話題に具体的に答える能力はまだ弱い。自動F1の改善を、そのまま自然な会話の獲得とは解釈しない。

医療QA162例はEOS到達162/162、平均生成19.65 Token、Token overlap precision 0.737016、recall 0.192238、F1 0.240837だった。回答形式は162/162で抽出でき、正解記号の完全一致は33/162（20.37%）となった。実験112の完全一致19/162（11.73%）から14例増え、EOS未到達も7例減った。一方、平均生成長は52.42 Tokenから19.65 Tokenへ大きく短くなり、recallとF1は実験112の0.372191より低下した。つまりanswer-focusは、長い誤説明を抑えて選択肢記号を当てる能力を改善したが、理由を十分に説明する能力や医学知識そのものを改善したとは言えない。正解した33例についても、理由部分が誤っている例が残っている。

生成サンプルと評価結果は、[checkpoint metadata](../../artifacts/checkpoints/issue1-fineweb-new-pretrain-answer-focus-sft-runpod-8k/best.json)、[metrics](../../artifacts/checkpoints/issue1-fineweb-new-pretrain-answer-focus-sft-runpod-8k/metrics.jsonl)、[step別生成](../../artifacts/samples/issue1-fineweb-new-pretrain-answer-focus-sft-runpod-8k/)、[領域評価](../../artifacts/evaluations/exp113/domains.json)、[一般会話評価](../../artifacts/evaluations/exp113/general-chat.json)、[一般会話全文](../../artifacts/evaluations/exp113/general-chat.txt)、[医療評価](../../artifacts/evaluations/exp113/medical-chat.json)、[医療全文](../../artifacts/evaluations/exp113/medical-chat.txt)に保存した。metadataのbest JSONのSHA-256は`0e009c3d03a7f074957a9c2aa5b573796c4b78463c49c8a1267221d8a282968a`、metricsは`97df2ba31b72853f15c26df963875a6cdefcc10b8ace0bc7202ef26ba910fee7`、領域評価は`c4d475308ad882c1c110b0972d75b06e658420363a8a09a0705f6e9249c85614`、一般会話JSONは`1ae4682199902ac7fa6854c9b5e30df9d45ac766a095a2b0181f59f19e67c46a`、一般会話TXTは`9201817929b16498adcf72daf310e009b0110098222922c64a81fa4205182c0b`、医療JSONは`f2edf22a72c4438c909f330be0ec7f7943cdd3529eb4d81152a9c2940caa19ad`、医療TXTは`acced9952a554161de6ededc4f4b96921dd216984b338c9bb5eb509f530e7134`である。

仮説は部分的に支持された。answer-focusを加えると医療の完全一致が改善し、一般会話F1も悪化しなかったため、現在の主線へ「短い正解形式を少量混ぜる」方針は採用できる。ただし、医療F1と理由の正確さは低下したため、answer-focusだけを増やすことは採用しない。今後は、正解記号だけの例と、正しい理由を短く保った例を分け、回答の長さを抑えながら根拠の品質を守る必要がある。

次の候補は、一般会話のSFTデータで短い相づちだけを学ぶ比率を減らし、質問・話題継続・具体的な内容応答を一定量確保した構成を作ることである。同時に、FineWeb2の追加日本語Tokenをさらに取得してraw基盤を延長し、データ量の効果と会話SFTの効果を混同しないよう別実験として比較する。
