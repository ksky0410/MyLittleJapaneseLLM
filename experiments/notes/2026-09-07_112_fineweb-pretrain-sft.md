# 実験112：新規FineWeb追加事前学習checkpointへの一般・医療SFT

## 実施前の計画

実験111では、未使用FineWeb2日本語shard 1・2から19,999,123 tokensを抽出し、実験110のSFT済みbest checkpointへ10,000 stepの追加事前学習を行った。FineWeb validation lossは2.941318から2.796272へ大きく改善したが、raw checkpointの固定promptには「お客様の声」の反復が現れ、medical回答形式も崩れた。これはraw事前学習の知識側効果と会話形式の維持が別問題であることを示している。

今回は実験111 raw bestへ、実験110と同じ一般・医療SFTを8,000 step再適用する。目的は、新しいFineWeb文書で得た一般日本語の改善を残しながら、一般会話のEOS・話者形式と医療問題の回答形式を回復できるかを確認することである。教師LLMによる蒸留、reasoningデータ、医療データの追加倍率は使わず、pretraining後に元のSFTを戻す順序だけを検証する。

### 仮説

実験111 raw checkpointはFineWeb lossが大きく改善しているため、SFT後にもgeneral validation lossと一般会話の語彙適合へ一部の効果が残る可能性がある。実験110と同じSFTなら、一般会話のEOS 48/48と医療回答の抽出162/162を回復し、医療F1もrawの0.2467から0.36以上へ戻ると予想する。一方、SFTがFineWeb lossを押し戻すため、実験111 rawの2.796272をそのまま維持することは期待しない。

### 開始前の条件

- 実験番号：112
- 実施日：2026-09-07
- 担当：Codex
- 実行環境：Runpod Pod `j9c46julmtbcb4`、NVIDIA A40、PyTorch CUDA
- 初期checkpoint：実験111 raw best、step 10,000
- 初期checkpoint SHA-256：`6957aaab539af1d6924d5c43a0c44a057a356c35dbac79c49fbe2279962468b9`
- SFT train：`artifacts/sft/issue1-general-medical-concat-v1/train.npz`、SHA-256 `598c464b03cd94a9c5579552df5f78059410f8ce5721da6cc93acb8251382cf4`
- SFT validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal token列：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`
- Tokenizer：`mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- SFT：batch size 8、8,000 step、AdamW、learning rate 2e-5から2e-6、warmup 200、weight decay 0.01、seed 112、rehearsal ratio 0.2
- 設定ファイル：`configs/issue1-fineweb-new-pretrain-general-medical-sft-runpod-8k.toml`
- 設定ファイルSHA-256：`2f0e0f0e5d24eaebda8340f4da34487eb6ab6420573f587713ec240447a3a1d0`
- 学習コード：`scripts/train_sft_torch.py`
- 学習コードSHA-256：`bc78ec94a7f74399d049ce4d1f6a22b446437a90b8e855bf64233b935267974e`

### 実行コマンド

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_sft_torch.py \\
  --config configs/issue1-fineweb-new-pretrain-general-medical-sft-runpod-8k.toml \\
  --base-checkpoint artifacts/checkpoints/issue1-50m-pretrain-fineweb-new-shards-runpod-10k/best.pt \\
  --train-data artifacts/sft/issue1-general-medical-concat-v1/train.npz \\
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \\
  --output-dir artifacts/checkpoints/issue1-fineweb-new-pretrain-general-medical-sft-runpod-8k \\
  --samples-dir artifacts/samples/issue1-fineweb-new-pretrain-general-medical-sft-runpod-8k \\
  --device cuda --sample-template conversation --sample-speaker-a DA --sample-speaker-b DC \\
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin --rehearsal-ratio 0.2
```

### 成功判定

NaN、OOM、shape errorなく8,000 stepを完走し、250 stepごとのvalidation lossと生成サンプルを保存する。実験111 rawで崩れた会話形式が回復し、一般会話48例のEOS 48/48、医療162例の回答形式抽出162/162を目標とする。実験110と比較してFineWeb、general、conversation、medicalのloss、一般会話F1、医療F1、医療正解率をすべて記録し、どの能力が追加FineWebから残ったかを切り分ける。

## 学習中の記録

ここに250 stepごとのvalidation loss、learning rate、経過時間、GPUメモリ、固定prompt生成、警告、設定変更を追記する。悪い生成や短すぎる応答も削除せず保存する。

### 2026-09-07：step 1〜250

Runpod A40上で実験111 raw bestからSFTを開始した。step 1のvalidation lossは2.913144、step 250は2.804113、step 250のlearning rateは1.9998e-5、経過時間は21.39秒だった。step 250の固定会話生成は「こんにちは!」に対して「こんにちは!」となり、raw checkpointで見られた長い反復はこの時点では現れていない。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 1,000〜1,250

step 1,000のvalidation lossは2.811358、perplexityは16.6325、learning rateは1.9538e-5、経過時間は82.00秒だった。step 1,250ではloss 2.808899、perplexity 16.5916、learning rate 1.9209e-5、経過時間102.14秒となった。現時点の最良はstep 250の2.804113で、学習は安定している。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 2,000〜2,250

step 2,000のvalidation lossは2.803952、perplexityは16.5098、learning rateは1.7739e-5、経過時間は161.93秒だった。step 2,000でvalidation lossの最良値をわずかに更新したが、step 2,250では2.811817へ戻った。学習率はまだ減衰前半であり、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 3,000

step 3,000のvalidation lossは2.789558、perplexityは16.2738、learning rateは1.4862e-5、経過時間は242.28秒だった。step 2,750の2.799209からさらに改善し、ここまでの最良値を更新した。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 4,000

step 4,000のvalidation lossは2.787631、perplexityは16.2425、learning rateは1.1366e-5、経過時間は322.61秒だった。最良値はstep 3,250の2.786897で、step 3,500以降はこの近辺で推移している。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 5,000〜5,250

step 5,000のvalidation lossは2.769219、perplexityは15.9462、learning rateは7.8119e-6、経過時間は402.41秒だった。step 5,250ではloss 2.763094、perplexity 15.8488、learning rate 6.9821e-6、経過時間422.81秒となり、実験110のbest validation loss 2.773049を更新した。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 6,000

step 6,000のvalidation lossは2.761517、perplexityは15.8238、learning rateは4.7681e-6、経過時間は482.98秒だった。最良値はstep 5,500の2.758811で、実験110のbest 2.773049より0.014238低い。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 6,250〜7,500

step 6,250のvalidation lossは2.763860、step 6,500は2.757301、step 6,750は2.754414、step 7,000は2.756882、step 7,250は2.752892、step 7,500は2.754526だった。step 7,250で最良値を2.752892まで更新し、実験110のbest 2.773049を0.020157下回っている。step 7,500時点のlearning rateは2.1826e-6、経過時間は604.56秒であり、NaN、OOM、shape errorは発生していない。

### 2026-09-07：評価準備時の失敗

学習完了後の領域評価を開始したところ、FineWeb2 validation Token列の指定を誤り、`issue1-fineweb2-edu-ja-small-20m-val.bin` がRunpod上に存在しないため評価プログラムが停止した。学習結果やcheckpointには影響しない。実際に存在する `artifacts/tokens/fineweb2-edu-japanese-v1-test.bin` を使って同じ評価を再実行する。

## 実験終了後の記録

実験112はNaN、OOM、shape errorなく8,000 stepを完走した。最良checkpointはstep 7,750で、validation lossは2.751857、perplexityは15.6717だった。step 8,000のvalidation lossは2.753525であり、後半はstep 7,250〜7,750付近で安定した。最良checkpointの重みSHA-256は`4979355786b4faef5787e366067d856ade06a5bc62170d2a9a30325eb3abda15`、学習時間は645.63秒、peak allocated memoryは1,490,586,112 bytesだった。

領域別validation lossはFineWeb2 2.924918、general 4.060053、conversation 2.006600、medical 1.876712だった。実験110と比べると、FineWeb2は2.941318から0.016400改善し、generalは4.073868から0.013815改善、conversationは2.029899から0.023299改善した。一方、medicalは1.874220から0.002492悪化しており、追加FineWeb2をSFTへ戻しただけでは医療の改善は確認できなかった。実験111 rawと比べるとFineWeb2は2.796272から悪化したが、generalは4.229550から、conversationは2.097783から、medicalは1.923146からそれぞれ改善し、SFTによって応答形式が回復した。

一般会話48例はEOS到達48/48、平均生成8.42 Token、Token overlap F1 0.222845だった。実験110のEOS 48/48、平均7.08 Token、F1 0.232590と比べると、生成長は少し伸びたがF1は0.009745低下した。実際の生成には「こんにちは」「そうですよね!」「そうなんですね」「すごいですね」のような短い相づちが多く、文法としては自然でも、直前の話題へ具体的に応じない例が残った。したがって、general lossの改善は自然な会話能力の改善と同一ではない。

医療QA162例はEOS到達155/162、平均生成52.42 Token、Token overlap F1 0.372191だった。回答形式の抽出は162/162だったが、正解記号の完全一致は19/162（11.73%）に留まった。実験110のEOS 158/162、平均54.64 Token、F1 0.381138、完全一致31/162（19.14%）と比べ、今回は医療の正答率が下がった。出力は「正解はcです。理由は…」という形式を守る一方、選択肢と無関係な医学用語をつなげたり、同じ語句を反復したりする例が多かった。これは回答形式の学習と、問題文から正しい知識を取り出す能力が別であることを示している。

固定promptの学習中サンプルと評価全文は、[checkpoint metadata](../../artifacts/checkpoints/issue1-fineweb-new-pretrain-general-medical-sft-runpod-8k/best.json)、[metrics](../../artifacts/checkpoints/issue1-fineweb-new-pretrain-general-medical-sft-runpod-8k/metrics.jsonl)、[step別生成](../../artifacts/samples/issue1-fineweb-new-pretrain-general-medical-sft-runpod-8k/)、[領域評価](../../artifacts/evaluations/exp112/domains.json)、[一般会話評価](../../artifacts/evaluations/exp112/general-chat.json)、[一般会話全文](../../artifacts/evaluations/exp112/general-chat.txt)、[医療評価](../../artifacts/evaluations/exp112/medical-chat.json)、[医療全文](../../artifacts/evaluations/exp112/medical-chat.txt)に保存した。領域評価のSHA-256は`bd75da863dab1f2f35c70586b90e30ba8cca61e1bc6cf47727c6e7c5f8e253ef`、一般会話JSONは`70ead40470c999c0244733a9714d3969ddae0d558739f4787f171d35ea063979`、一般会話TXTは`5b71df4da2cafac6b8dbf12611b067f070855a37574391dc250740956ab989e8`、医療JSONは`c1b6c1bcb86703849064f64af260bf67714cf89e5cb3cbdef26efe8478952b2e`、医療TXTは`c59b4ef12ec575475eacfa3be4e9e5fd619ac44de03564f4822f9246bbcb8492`である。

事前の予想のうち、「追加FineWeb2で得た改善がSFT後にも一部残る」は、generalとconversationのloss、およびFineWeb2のlossが実験110をわずかに上回ったため部分的に支持された。一方、「一般会話F1と医療F1が実験110以上へ戻る」は支持されなかった。さらに、評価準備でFineWeb2検証列のファイル名を一度誤ったが、失敗を削除せず記録し、既存の`fineweb2-edu-japanese-v1-test.bin`へ修正して同じ評価を完了した。

次は、長い誤説明をそのまま学習することが医療QAの正答率を下げている可能性を切り分ける。実験113では通常医療回答に加えて、同じ問題へ「正解はaです。」だけを返すanswer-focusデータを追加し、一般会話と通常医療を残したまま学習する。
