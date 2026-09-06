# 実験110：追加事前学習checkpointへの一般・医療SFT再適用

## 実施前の計画

実験109では、SFT済みの50Mモデルへ、会話と医師国家試験を含む約20M tokensの追加事前学習を行った。FineWeb、会話、医療のvalidation lossは明確に改善した一方、一般会話と医療問題の固定評価ではEOS形式、返答の長さ、回答形式を大きく忘れた。これは追加データが無効だったという意味ではなく、raw next-token学習だけではSFTで獲得した応答形式を維持できないことを示している。

そこで今回は、実験109の最良checkpointへ、実験105〜108で使用した一般・医療の応答SFTを改めて8,000 step適用する。目的は、追加事前学習で改善した会話・医療の内部表現を残したまま、自然な返答形式と質問応答能力を回復できるかを確認することである。追加事前学習の後に十分なSFTを置く構成を、今後の標準候補にできるかを判断する。

### 仮説

実験109のraw validation loss改善は、会話と医療の語彙や文脈を追加データから学習できたことを示している。そのcheckpointに元の一般・医療SFTを重ねれば、実験105のSFT後checkpointよりも会話・医療の応答品質が高くなり、少なくともFineWeb lossの改善を一部維持できると予想する。ただし、追加事前学習で一般Web領域のlossが悪化しているため、一般会話の自然さが完全には戻らない可能性もある。

### 開始前の条件

- 実験番号：110
- 実施日：2026-09-07
- 担当：Codex
- 使用ブランチ：`main`
- 実行環境：Runpod Pod `j9c46julmtbcb4`、A40、PyTorch CUDA
- 初期checkpoint：実験109の最良checkpoint、step 9,500
- 初期checkpoint SHA-256：`0ceaeedef9d8ab8039861078bce673977a78a8bf7d329ff57096184b9f531cea`
- SFT train：`artifacts/sft/issue1-general-medical-concat-v1/train.npz`
- SFT train SHA-256：`598c464b03cd94a9c5579552df5f78059410f8ce5721da6cc93acb8251382cf4`
- SFT validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`
- SFT validation SHA-256：`95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal token列：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`
- Tokenizer：`mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- SFT設定：batch size 8、8,000 step、AdamW、learning rate 2e-5から2e-6、warmup 200、weight decay 0.01、seed 110、rehearsal ratio 0.2
- 設定ファイル：`configs/issue1-new-pretrain-general-medical-sft-runpod-8k.toml`
- 設定ファイルSHA-256：`dd75a98850863699d2befd162b279da874d59d46d3d55df601d27a3664f84c92`
- 学習コード：`scripts/train_sft_torch.py`
- 学習コードSHA-256：`bc78ec94a7f74399d049ce4d1f6a22b446437a90b8e855bf64233b935267974e`

周期checkpointは最新の1個だけ残すが、stepごとのmetrics、生成サンプル、評価結果、失敗ログは削除しない。大きな`.pt`本体はGitHubへ追加せず、最良checkpointのSHA-256と保存場所を記録する。生成された日本語は、良い結果だけでなく崩れた結果もすべてGitHubへ公開する。

### 実行コマンド

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_sft_torch.py \\
  --config configs/issue1-new-pretrain-general-medical-sft-runpod-8k.toml \\
  --base-checkpoint artifacts/checkpoints/issue1-50m-pretrain-new-japanese-20m-runpod-10k/best.pt \\
  --train-data artifacts/sft/issue1-general-medical-concat-v1/train.npz \\
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \\
  --output-dir artifacts/checkpoints/issue1-new-pretrain-general-medical-sft-runpod-8k \\
  --samples-dir artifacts/samples/issue1-new-pretrain-general-medical-sft-runpod-8k \\
  --device cuda --sample-template conversation --sample-speaker-a DA --sample-speaker-b DC \\
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin --rehearsal-ratio 0.2
```

### 成功判定

NaN、OOM、shape errorなく8,000 stepを完走し、250 stepごとのvalidation lossと生成サンプルを保存することを最低条件とする。性能面では、実験109で崩れた一般会話のEOSと返答長が回復し、医療評価で「正解は○です。」を抽出できる状態へ戻ることを重視する。実験105〜108の最良値と比べて、一般会話F1、医療正解率、領域別validation lossを比較する。すべての指標が同時に改善しなくても、raw追加事前学習の効果をSFT後まで残せたかを明確に切り分ける。

## 学習中の記録

ここに250 stepごとのvalidation loss、learning rate、elapsed time、GPUメモリ、固定prompt生成、警告、設定変更を追記する。実験が失敗した場合も、失敗した時点と原因を削除せずに残す。

### 2026-09-07：step 250

RunpodのA40上で、実験109のbest checkpointを初期値として学習を開始した。step 1のvalidation lossは3.183849、step 250は2.876897、step 250のlearning rateは1.9998e-5、経過時間は21.14秒だった。rehearsal ratioは0.2で、NaN、OOM、shape errorは発生していない。学習は継続中であり、step 250時点では生成品質の判定を保留する。

### 2026-09-07：step 500

step 500のvalidation lossは2.870976、perplexityは17.6542、learning rateは1.9935e-5、経過時間は41.47秒だった。学習中サンプルでは、`こんにちは！`に対して「こんにちは。今日はどこか行かれました?」と返しており、実験109のraw事前学習直後に失われた会話形式が早い段階で回復している。これは固定prompt一例に基づく暫定的な観察であり、一般48例と医療162例の最終評価で確認する。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 1,000

step 1,000のvalidation lossは2.875087、perplexityは17.7270、learning rateは1.9538e-5、経過時間は81.89秒だった。固定promptへの生成は「こんにちは!お願いします!」となり、会話の開始形式は維持されているが、step 500のサンプルより短い。学習中生成はtemperatureを含むため、この差だけで品質の悪化とは判断しない。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 2,000

step 2,000のvalidation lossは2.846258、perplexityは17.2232、learning rateは1.7739e-5、経過時間は162.38秒だった。この時点でvalidation lossの最良値を更新した。固定promptへの生成は「こんにちは!宜しくお願いいたします。」となり、短いながらも話者markerの後に自然な挨拶を返している。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 3,000

step 3,000のvalidation lossは2.832890、perplexityは16.9945、learning rateは1.4862e-5、経過時間は243.16秒だった。step 2,750の2.835384を経て最良値を更新しており、step 2,000以降もvalidation lossは緩やかに改善している。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 4,000

step 4,000のvalidation lossは2.808299、perplexityは16.5817、learning rateは1.1366e-5、経過時間は324.20秒だった。ここまでの最良値を更新している。固定promptへの生成は「こんにちは!」に対して「こんにちは!」と短く返した。形式は崩れていないが、返答の長さはサンプリングにより揺れているため、最終の48例評価で平均長とEOS率を確認する。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 5,000

step 5,000のvalidation lossは2.795988、perplexityは16.3788、learning rateは7.8119e-6、経過時間は405.23秒だった。最良値はstep 4,750の2.791013で、step 5,000では少し戻ったが、学習は安定して継続している。固定promptへの生成は「こんにちは!」に対して「こんにちは!」となった。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 6,000

step 6,000のvalidation lossは2.784527、perplexityは16.1922、learning rateは4.7681e-6、経過時間は484.93秒だった。最良値はstep 5,750の2.782389で、この時点ではわずかに戻っている。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 7,000

step 7,000のvalidation lossは2.774740、perplexityは16.0345、learning rateは2.7216e-6、経過時間は564.53秒だった。最良値を更新し、追加事前学習後のSFTがvalidation上で継続的に効いている。NaN、OOM、shape errorは発生していない。

## 実験終了後の記録

学習はNaN、OOM、shape error、途中停止なく8,000 stepを完走した。実際のbackendはPyTorch CUDA、Torch 2.9.1+cu128、CUDA 12.8、GPUはNVIDIA A40である。パラメータ数は50,207,616、学習時間は645.176秒、最大割当メモリは1,490,586,112 bytesだった。最良checkpointはstep 7,500で、SFT validation lossは2.773049、perplexityは16.0074、最終step 8,000はloss 2.774077だった。したがって、最終stepではなくstep 7,500の`best.pt`を比較対象にする。

最良checkpointの重みSHA-256は `cbf18b9cbd39ec9256811e945ecc3e7fc99fe39d12d6987df7198027fd8ae492` である。best metadataは `ad9ee747b4465b829a1319fed3beea2d776aefce93f473b0ca0afc82532f6f88`、metricsは `acea34bf6beeea8ecbcb1dce3b531501a37fe5dbb6811494433381732b000b71`、summaryは `38c293983e795a878d057232e09d03daa5ab70b8baa2ddf7d37a4d5e4dfcb999`、step 8,000 metadataは `5636af87485372d9ed976bd2ca303127731a04fd5766c4ce713476082504c886` である。重い重み本体はGitへ追加せず、Runpodの `/workspace/exp100/artifacts/checkpoints/issue1-new-pretrain-general-medical-sft-runpod-8k/best.pt` に保存している。

同じbest checkpointを、FineWeb・general・conversation・medicalの各20 evaluation batches、一般会話48例、医療162例で評価した。領域別validation lossは、FineWeb 2.941318、general 4.073868、conversation 2.029899、medical 1.874220だった。実験105のFineWeb 2.921111、general 4.084512、conversation 2.070219、medical 1.977347と比較すると、FineWebだけは0.020207悪化したが、general・conversation・medicalはそれぞれ0.010644、0.040320、0.103127改善した。実験108と比較しても、FineWebは0.018211悪化した一方、generalは0.008777、conversationは0.038569、medicalは0.101402改善した。追加事前学習直後の実験109と比べると、SFTによってgeneralは4.126995から4.073868へ回復したが、FineWeb・conversation・medicalはraw追加事前学習直後より少し戻った。これは、SFTが知識側lossを一部犠牲にして応答形式を回復するトレードオフを示している。

一般会話48例では、EOS到達48/48、平均生成長7.083 tokens、token-overlap F1 0.232590だった。実験105の11.333 tokens・F1 0.208588よりF1は高く、実験108の9.583 tokens・F1 0.235895とはほぼ同水準だったが、生成長は短くなった。実験109直後のF1 0.125177からは明確に回復した。固定評価の出力には「はい」「そうですね」「わかります」「いいですね」のような自然な短い相づちが含まれる一方、空の応答や一語で終わる例も残っているため、自然な会話能力が十分に得られたとはまだ判定しない。

医療162例では、EOS到達158/162、平均生成長54.636 tokens、token-overlap F1 0.381138だった。「正解は○です」という形式を162/162例から抽出でき、正解選択肢と一致したのは31例、正解率は31/162＝19.14%だった。実験105の29/162＝17.90%、F1 0.374503からは改善し、実験108のF1 0.368071も上回った。実験109直後は抽出12/162、正解2/162＝1.23%、F1 0.090522だったため、SFTによる形式回復は明確である。ただし正解率は実験106の31/162＝19.14%と同じで、5択問題の偶然水準に近い。医師国家試験への医学的な実用性や正確性を得たとは扱わない。

この結果は、今回の仮説を部分的に支持する。追加事前学習後に元の一般・医療SFTを十分に戻すことで、実験109で失われた会話形式・医療回答形式を回復できた。また、実験105・108と比べてgeneral、conversation、medicalのvalidation lossと医療F1は改善した。一方、FineWeb lossは悪化し、一般会話は自然な長い返答ではなく短い相づちへ寄った。医療正答率も大きく伸びなかったため、「追加データを増やしてSFTを重ねれば、そのまま知識と自然さが同時に伸びる」とは結論できない。

現時点では、実験110のstep 7,500 checkpointを、実験105・108に代わる総合候補として保存する。ただし、自然な日本語を優先する本来の目的に対しては、一般会話の意味適合性と返答長がまだ不足している。次は、追加事前学習の量をさらに増やす前に、SFT中の返答長・空応答・話題適合性を評価できる形へ整え、一般会話データの質と反復を変数にした比較を行う。医療QAについては、正解率が低いままなので、単に医療例を増やすのではなく、選択肢識別を直接学ぶデータ形式を別実験として切り分ける。

学習中のmetrics、summary、checkpoint metadata、step別生成は [`artifacts/checkpoints/issue1-new-pretrain-general-medical-sft-runpod-8k/`](../../artifacts/checkpoints/issue1-new-pretrain-general-medical-sft-runpod-8k/) と [`artifacts/samples/issue1-new-pretrain-general-medical-sft-runpod-8k/`](../../artifacts/samples/issue1-new-pretrain-general-medical-sft-runpod-8k/) に保存した。領域評価は [`domains.json`](../../artifacts/evaluations/exp110/domains.json)（SHA-256 `82e4f70eb3a6a299b283fd535ac85685bc9b14ff6da8291cb0c518701f9842bf`）、一般会話のJSON/TXTは [`general-chat.json`](../../artifacts/evaluations/exp110/general-chat.json)（`b71a4e8364a104b35eaf9c51f32a197d86fb56d58309d32924410a34cc184cd0`）と [`general-chat.txt`](../../artifacts/evaluations/exp110/general-chat.txt)（`207f06651679dc166e1ed06993a61d942087026b5f368f2041d53f3a20290f20`）、医療のJSON/TXTは [`medical-chat.json`](../../artifacts/evaluations/exp110/medical-chat.json)（`4cf429baee8534b1c87c92ba5413b442960a40e9ddd02f355dff843bb37a3863`）と [`medical-chat.txt`](../../artifacts/evaluations/exp110/medical-chat.txt)（`b5f83bd62426a9bbac8767e8e4d969fbe18ecec048acc384488b0c72d7c00f24`）で確認できる。学習ログと評価ログは [`experiments/results/exp110/`](../../experiments/results/exp110/) に保存した。
