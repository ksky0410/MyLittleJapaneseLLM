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

## 実験終了後の記録

学習完了後に、最良checkpointのstepとSHA-256、最終loss、学習時間、最大GPUメモリ、評価JSON/TXTのSHA-256、実験112との比較、仮説の判定、次に試す変更を追記する。checkpoint本体はGitへ追加せず、metadataとハッシュだけを記録する。
