# 実験115：shard 3・4追加事前学習後のanswer-focus SFT

## 実施前の計画

### 目的

実験114でFineWeb2 Edu Japaneseの未使用shard 3・4を約20M tokens追加したraw checkpointへ、実験113と同じ一般会話・通常医療・answer-focus医療のSFTを適用する。実験113との比較によって、追加20M tokensの日本語事前学習が、SFT後の一般会話と医療QAの正答率・説明品質へ残るかを測定する。

### 仮説

実験114 rawはFineWeb validation lossを2.777802まで下げた一方、会話・医療の形式を忘れていた。実験113と同じSFTを再適用すれば、EOSと回答形式は回復すると予想する。さらに、追加FineWeb文書で増えた語彙・文体が残るなら、実験113よりgeneral・conversationのlossまたは一般会話F1が改善し、医療の完全一致33/162も維持または上回る可能性がある。ただし、raw pretrainingで知識の配置が変わったため、answer-focus SFTのvalidationが不安定になったり、医療理由の誤生成が残ったりする可能性もある。

### 比較条件

実験113から初期checkpointだけを変更し、その他を揃える。

- 初期checkpoint：実験114 raw best、step 10,000、SHA-256 `4f6f9f4ddad1f717dbf170de8b1d5e704e1ef3975c1b4ab4e5e905abc5c3eca6`
- 学習データ：`artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz`、一般会話約12.8万例、通常医療2,945例、answer-focus医療2,945例
- validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、実験113と同一
- rehearsal：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、ratio 0.2
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- SFT：batch size 8、8,000 step、AdamW、learning rate 2e-5から2e-6、warmup 200、weight decay 0.01、seed 115
- 評価：FineWeb2、general、conversation、medicalのdomain loss、一般会話48例、医療162例

### 入力のSHA-256

学習開始前に次のハッシュをRunpod上で照合する。

- 設定：`configs/issue1-fineweb-shards34-answer-focus-sft-runpod-8k.toml`、SHA-256 `9c43cedeb66436bed8b807fbd9186263a1fa55ae26d2d0e70b977affb169c508`
- 学習コード：`scripts/train_sft_torch.py`、SHA-256 `bc78ec94a7f74399d049ce4d1f6a22b446437a90b8e855bf64233b935267974e`
- answer-focus train：`artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz`、SHA-256 `99fc5e82cefc7efd7e4eb69bb5250d794526313c4ae6e54eeee862673100b262`
- validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal Token列：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`

### 実行コマンド

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_sft_torch.py \
  --config configs/issue1-fineweb-shards34-answer-focus-sft-runpod-8k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-50m-pretrain-fineweb-new-shards34-runpod-10k/best.pt \
  --train-data artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz \
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-fineweb-shards34-answer-focus-sft-runpod-8k \
  --samples-dir artifacts/samples/issue1-fineweb-shards34-answer-focus-sft-runpod-8k \
  --device cuda --sample-template conversation --sample-speaker-a DA --sample-speaker-b DC \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin --rehearsal-ratio 0.2
```

### 成功・失敗の判定

NaN、OOM、shape errorなく8,000 stepを完走し、250 stepごとのvalidation lossと固定prompt生成を保存する。実験113と同じ評価器で比較し、一般会話F1、EOS、平均生成長、医療の回答抽出・完全一致・F1を確認する。追加pretraining後のSFTが実験113より全般に悪化しても、データ量を増やすだけではSFT性能が単調に伸びない反証として記録する。

## 学習中の記録

ここに1,000 stepを超えない間隔でvalidation loss、perplexity、learning rate、経過時間、GPUメモリ、固定prompt生成、警告、設定変更を追記する。悪い生成も削除しない。

### 2026-09-07：step 1〜250

Runpod A40上で実験114 raw bestからanswer-focus SFTを開始した。step 1のvalidation lossは2.937145、perplexityは18.8619、learning rateは1.0e-7、経過時間は0.89秒だった。step 250ではvalidation loss 2.829083、perplexity 16.9299、learning rate 1.9998e-5、経過時間21.47秒となった。実験113のstep 250よりvalidation lossは高いが、学習は安定しており、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 500〜1,000

step 500のvalidation lossは2.811753、step 750は2.823874、step 1,000は2.813150だった。最良値はstep 500の2.811753で、step 1,000時点のlearning rateは1.9538e-5、経過時間は84.18秒だった。実験113の同時点より高いものの、学習は安定している。

### 2026-09-07：step 1,250〜1,500

step 1,250のvalidation lossは2.828422、step 1,500は2.827187だった。step 500の2.811753を更新しておらず、step 1,500時点のlearning rateは1.8796e-5、経過時間は125.16秒だった。validation lossは実験113より高い状態で推移しているが、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 1,750〜2,250

step 1,750のvalidation lossは2.818044、step 2,000は2.808153、step 2,250は2.807672だった。step 2,250で最良値を更新したが、実験113の同時期よりまだ高い。step 2,250時点のlearning rateは1.7105e-5、経過時間は186.84秒で、学習は安定している。

### 2026-09-07：step 2,500〜3,000

step 2,500のvalidation lossは2.809942、step 2,750は2.810437、step 3,000は2.802531だった。step 3,000で最良値を更新し、step 3,000時点のlearning rateは1.4862e-5、経過時間は247.61秒となった。実験113との差は縮まっているが、まだ追加pretraining後の条件が高い。

### 2026-09-07：step 3,250〜3,750

step 3,250のvalidation lossは2.802477、step 3,500は2.803479、step 3,750は2.794012だった。step 3,750で最良値を更新し、実験113のstep 3,750の2.782681との差は0.011331となった。step 3,750時点のlearning rateは1.2268e-5、経過時間は309.47秒で、学習は安定している。

### 2026-09-07：step 4,000〜4,500

step 4,000のvalidation lossは2.795366、step 4,250は2.781239、step 4,500は2.783774だった。step 4,250で最良値を更新し、実験113のstep 4,250の2.784560を0.003321下回った。step 4,500時点のlearning rateは9.5599e-6、経過時間は370.56秒だった。

### 2026-09-07：step 4,750〜5,000

step 4,750のvalidation lossは2.779978、step 5,000は2.784362だった。step 4,750で最良値を更新し、実験113の最良2.756807との差は0.023171である。step 5,000時点のlearning rateは7.8119e-6、経過時間は411.85秒だった。後半に向けてlossは改善しているが、まだ実験113の最終水準には届いていない。

### 2026-09-07：step 5,250〜5,750

step 5,250のvalidation lossは2.777582、step 5,500は2.776341、step 5,750は2.769971だった。step 5,750で最良値を更新し、実験113の最良2.756807との差は0.013164へ縮まった。step 5,750時点のlearning rateは5.4524e-6、経過時間は473.59秒だった。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 6,000〜6,500

step 6,000のvalidation lossは2.766638、step 6,250は2.762123、step 6,500は2.758995だった。step 6,500で最良値を更新し、実験113の最良2.756807との差は0.002188まで縮まった。step 6,500時点のlearning rateは3.5952e-6、経過時間は534.93秒となった。

### 2026-09-07：step 6,750〜7,250

step 6,750のvalidation lossは2.757589、step 7,000は2.756483、step 7,250は2.754919だった。step 7,250で最良値を更新し、実験113の最良2.756807を0.001888下回った。step 7,250時点のlearning rateは2.4086e-6、経過時間は596.77秒だった。

### 2026-09-07：step 7,500〜8,000

step 7,500、7,750を経て、最終step 8,000のvalidation lossは2.754026、perplexityは15.7057となった。最終的な最良checkpointはstep 8,000であり、実験113の最良validation loss 2.756807を0.002781下回った。学習時間は657.43秒、summaryに記録された全体時間は658.16秒、A40でのpeak allocated memoryは1,490,586,112 bytesだった。NaN、OOM、shape errorは発生せず、8,000 stepを完走した。

固定prompt `こんにちは！` の途中生成はstep 0、4,000、8,000のいずれでも、入力をほぼそのまま短く返す状態だった。これは学習が停止していたという意味ではなく、固定promptが単純すぎて改善を観察する診断として弱いことを示す。途中生成の全33ファイルは `artifacts/samples/issue1-fineweb-shards34-answer-focus-sft-runpod-8k/` に保存した。

## 実験終了後の記録

### 評価結果

学習完了後、最初の評価投入はSSH timeoutで失敗したが、checkpointや学習プロセスには影響しなかった。失敗を記録したうえで同じコマンドを再試行し、Runpod A40上でdomain loss、一般会話48例、医療162例を最後まで評価した。評価結果は `artifacts/evaluations/exp115/`、評価ログは `experiments/results/exp115/evaluation.log` に保存した。

FineWeb2のvalidation lossは2.903494、generalは4.047548、conversationは2.007362、medicalは1.876458だった。実験113と比べるとgeneralは4.052437からわずかに改善し、conversationは2.007372からほぼ同じ、medicalは1.875708からわずかに悪化した。追加したshard 3・4でrawのFineWeb2 lossは大きく下がったが、SFT後の日本語ドメインloss全体が同じ割合で改善したわけではない。

一般会話48例では48例すべてがEOSに到達し、平均生成長は9.458 tokens、token overlap F1は0.265719だった。実験113の平均9.104 tokens、F1 0.254858から、平均長とF1がともに改善した。短い相槌だけでなく、「そうなんですね。私は2週間待機です。」のように文脈に沿おうとする出力もあった一方、「そうなんですね。それか、私みたいに見ました。」のように、文法は成立していても会話として不自然な付け足しが残った。

医療QA162例では159例がEOSに到達し、平均生成長は28.580 tokens、token overlap F1は0.264921だった。正解選択肢を抽出できた例は162例中162例だったが、正解と一致したのは26例、16.05%にとどまった。実験113の33例、20.37%を下回った。たとえば正解の理由を一文だけ返せる例がある一方、選択肢を誤ったまま「家族歴 家族歴」と繰り返したり、医学用語らしい断片をつないだ長い無関係な説明を生成したりする例も残っている。したがって、validation lossが改善したことだけから、医療知識や説明の正確性が改善したとは判断できない。

今回の仮説は部分的に支持された。追加20M tokens後に同じSFTを行うと、SFT validation lossと一般会話F1は小さく改善したため、追加事前学習が会話能力に全く無効だったわけではない。しかし、医療QAの完全一致は悪化し、FineWeb2の追加だけでは知識の正確性や説明品質が保証されないことも確認した。特に、raw checkpointの評価で会話・医療の形式が崩れ、SFTによって形式は回復するものの、追加データが医療QAへ直接効いた証拠は得られなかった。

今回使用した最良checkpointの重みはGitHubへ追加せず、次のSHA-256で再現対象を固定した。

- 最良checkpoint `best.pt`：`e0c317ff57d8199a04c05f4751742367183701c1472b2c297a156f40e19beb5a`
- `best.json`：`34264ef26d0695c061439f2bfbe596852dee3418d1523e1bee1ddceb8b6ae80d`
- `summary.json`：`7a24c3b7171ffce6a8e6fe23b69a56ef4467cbc769bb354c8474922913a861fa`
- `metrics.jsonl`：`7bb5499a6e522c571fd880e96a21a6e97d1dc247653bb36035277af11f39eb0f`
- domain評価JSON：`293f165c38626070524e499fbb836b6996d54f2c054cff7c98b283ba76f792c4`
- 一般会話評価JSON：`ab8c917712416c758d77f3a63a70c40c96695c4b8b884485efd2179eda0b7bbb`
- 医療評価JSON：`63cf8202988cddc938f3c21a76dfb3bf0d3f684253d6f3b78c0ba8016d848747`
- 学習ログ：`89f1811d68add691371b2669c4eb5f69249373f21914c09a9da497f2378ebbfc`
- 評価ログ：`a24a864e8004e4ef765ec4a5c6b01e224884a6991669da084b3686329bdb425a`

次は、さらにFineWeb2を一括で足すのではなく、追加事前学習の効果とSFTデータの効果を分離できる実験を行う。具体的には、exp115のbest checkpointを起点に、一般会話データを増やしつつ、医療QAについては正解選択肢だけでなく、正解理由を短く正確に説明する例を一定割合で含めたSFT条件を作る。同時に、回答を短く切るanswer-focus条件と、理由を十分に生成させる条件を同じcheckpointから比較し、自然な会話と正確な回答の両方を測る。

### 評価開始時の接続失敗

学習完了後の評価をRunpodへ投入する最初のSSH接続がtimeoutとなり、`artifacts/evaluations/exp115`は作成されなかった。checkpointや学習結果には影響せず、Runpod上に評価プロセスが起動していないことを確認した。接続を再試行して同じ評価コマンドを実行する。

その後、SSHを再試行して評価を開始し、domain・一般会話・医療QAの全評価が完了した。最初の接続失敗は再現性やcheckpointの内容に影響していない。
