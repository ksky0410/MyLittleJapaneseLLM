# 実験106：balanced SFT後の短期医療比重化SFT

## 実施前の計画

- 実施日：2026-09-07
- 担当：Codex
- 状態：実行前
- 使用ブランチ：`main`
- 目的：実験105で改善した自然な一般会話を保ちながら、医師国家試験形式の正答率だけを短い追加SFTで改善できるかを確かめる。

実験105は4領域のvalidation lossと一般会話の生成を改善したが、医療162問の正解率は17.90%にとどまった。実験103では医療QAを4倍にした最初の1,000 stepで24.07%まで上がった一方、累積3,000 stepでは17.28%に戻った。そこで今回は、実験105の最終checkpointから医療4倍データで1,000 stepだけ学習し、長期学習による過適応を避ける。

### 仮説

短期の医療比重化SFTなら、医療の回答形式と選択肢識別を補強し、実験105の17.90%を上回る可能性がある。一般会話48例のEOS 48/48と、実験105の平均生成11.33 tokensは大きく崩れないと予想する。ただし、正解率が上がらず語句一致だけが変わる場合は、データ比率だけでは医学的正答を改善できないと判断する。

### 開始前の条件

- 初期checkpoint：実験105の最良 `artifacts/checkpoints/issue1-balanced-pretrain-general-medical-sft-runpod-8k/best.pt`
- 初期checkpoint SHA-256：`1652603515b24e0538abeba01a63c53da1af4de87b51738b90acebe7326b9149`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 学習データ：実験103と同じ一般127,731例・医療QA11,780例の医療4倍データ、`artifacts/sft/issue1-general-medical-heavy-v1/train.npz`、SHA-256 `0756ec956f98f67dd0b712bbf9b94a217bbb92c42dffe4e72c989f117421f493`
- validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal：`artifacts/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`、比率20%
- 学習設定：`configs/issue1-balanced-pretrain-medical-heavy-sft-runpod-1k.toml`
- 乱数seed：106
- 学習率：3e-6から3e-7までのcosine decay、warmup 100 steps
- 予定step：1,000 steps

### 成功判定

医療162問の正解率が17.90%を上回り、一般会話48例のEOS 48/48を維持することを第一条件とする。医療回答のF1だけが上がって正解率が変わらない場合は、医療正答能力の改善とはみなさない。失敗してもデータ比率をさらに上げる根拠にはせず、問題形式の統一や評価方法の改善へ進む。

## 実験中の記録

未実施。250 stepごとのmetricsと生成文を保存する。

### 2026-09-07：step 500

実験105の最良checkpointから短期医療比重化SFTを開始した。step 1のvalidation lossは2.919164、step 250は2.918554、step 500は2.916562だった。step 500の経過時間は41.18秒、学習率は1.889e-6だった。NaNやOOMは発生していない。

## 実験終了後の記録

### 2026-09-07：累積1,000 stepの本走と最終評価

1,000 stepまで学習し、validation lossは2.916562、学習時間は82.89秒、ピークGPUメモリは約1.49GBだった。最良checkpointの重みSHA-256は `11000731966707ccbe97a20a436c7a5253697ebcb74f4014851c86479bde1068` である。NaN、OOM、shape errorは発生しなかった。

領域別lossはFineWeb 2.921637、一般4.083526、会話2.067973、医療1.976129だった。実験105から医療lossは1.977347からわずかに改善し、会話lossも2.070219から改善した。

一般会話48例ではEOS 48/48、平均生成12.15 tokens、token overlap F1 0.1826だった。EOSは維持したが、実験105のF1 0.2086から下がった。医療162例ではEOS 153/162、平均生成56.48 tokens、F1 0.3560となった。回答形式は162/162例で抽出でき、正解は31例、正解率19.14%だった。実験105の29/162、17.90%からは改善したが、実験102の22.98%には届かなかった。

短期医療比重化には小さな改善があったが、一般会話の語句一致を犠牲にしており、医療正答の改善幅も十分ではない。このcheckpointは医療寄りの比較対象として保存するが、総合モデルの採用候補は実験105のままとする。次はデータ量や反復倍率ではなく、医療回答の学習目標を「理由を含む長い回答」から「最初に正解選択肢を確定し、その後に理由を続ける形式」へ変えるanswer-first実験を行う。
