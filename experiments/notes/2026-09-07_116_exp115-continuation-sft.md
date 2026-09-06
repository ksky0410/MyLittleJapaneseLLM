# 実験116：exp115 checkpointの低学習率SFT継続

## 実施前の計画

### 目的

実験115の最良checkpointを初期値として、同じ一般会話・通常医療・answer-focus医療のSFTを低い学習率で追加8,000 step行う。実験115では8,000 stepの最後までvalidation lossが改善し続け、一般会話の固定評価も過去条件より良かったため、学習stepを増やすことが自然な日本語と質問応答能力の改善につながるかを検証する。

### 仮説

exp115のbest checkpointから急に大きな学習率へ戻さず、2e-6から2e-7へゆっくり減衰させれば、既に得た会話形式を壊さず、語彙選択や文脈への適合が少し改善すると予想する。反対に、同じデータをさらに8,000 step見せることで訓練例の定型句を過剰に繰り返し、一般会話F1や医療正答率が下がる可能性もある。validation lossだけでなく、固定一般会話48例、医療162例、生成全文を用いて判断する。

### 比較条件

exp115とデータ・モデル・評価器を揃え、初期checkpointと継続学習の学習率・seed・sample promptだけを変更する。

- 初期checkpoint：exp115 best、step 8,000、重みSHA-256 `e0c317ff57d8199a04c05f4751742367183701c1472b2c297a156f40e19beb5a`
- 学習データ：`artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz`、SHA-256 `99fc5e82cefc7efd7e4eb69bb5250d794526313c4ae6e54eeee862673100b262`
- validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、ratio 0.2、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 継続後の累積step：16,000。追加stepは8,000
- batch size：8。学習率：2e-6から2e-7。warmup：0。weight decay：0.01。seed：116
- 設定：`configs/issue1-exp115-continuation-sft-runpod-16k.toml`

### 成功・失敗の判定

NaN、OOM、shape errorなく16,000 stepまで完走し、250 stepごとのvalidation lossと生成文を保存する。exp115に対して一般会話F1が維持または改善し、EOS 48/48を維持できれば、継続学習を自然な会話改善の候補とする。医療の完全一致が改善すれば追加の副次的成果とする。validation lossだけが改善して生成文が定型句や無関係な付け足しへ崩れた場合は、stepを増やすだけでは不十分と判断する。

### 実行コマンド

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_sft_torch.py \
  --config configs/issue1-exp115-continuation-sft-runpod-16k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-fineweb-shards34-answer-focus-sft-runpod-8k/best.pt \
  --train-data artifacts/sft/issue1-general-medical-answer-focus-v1/train.npz \
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-exp115-continuation-sft-runpod-16k \
  --samples-dir artifacts/samples/issue1-exp115-continuation-sft-runpod-16k \
  --device cuda --start-step 8000 --max-steps 16000 --lr-schedule-steps 16000 \
  --sample-template conversation --sample-speaker-a DA --sample-speaker-b DC \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin --rehearsal-ratio 0.2
```

学習開始前に、設定・学習コード・入力データ・exp115 checkpointのハッシュをRunpod上で照合する。学習中は少なくとも1,000 step以内ごとにこのノートへvalidation loss、学習率、経過時間、生成文、警告を追記する。学習終了後はexp115と同じ評価を行い、checkpoint本体をGitHubへ追加せず、メタデータ・ハッシュ・全生成文・評価全文・ログを保存する。

## 学習中の記録

学習開始前。ここへstep 8,250以降のvalidation lossと生成結果を追記する。

### 2026-09-07：初回起動の設定検証失敗

最初の起動は、設定の`training.warmup_steps`へ0を指定したため、設定検証で停止した。`warmup_steps`は正の整数だけを受け付ける実装であり、モデルのforward、checkpoint読み込み、学習データの読み込みより前に終了している。Runpod上の出力ディレクトリにはcheckpointや生成文は作成されず、exp115のcheckpointと入力データは変更されていない。設定を1へ修正し、継続学習開始時点ではwarmupの影響がほぼない条件として再実行する。

### 2026-09-07：step 8,250〜8,750

設定修正後の再実行は正常に開始した。step 8,250のvalidation lossは2.754303、step 8,500は2.754326、step 8,750は2.753209だった。step 8,750でexp115のbest 2.754026を更新し、learning rateは9.682e-7、経過時間は62.64秒だった。学習率を抑えた継続でも初期には改善が見られている。NaN、OOM、shape errorは発生していない。各stepの生成文は `artifacts/samples/issue1-exp115-continuation-sft-runpod-16k/` に保存されている。

### 2026-09-07：step 9,000〜9,250

step 9,000のvalidation lossは2.752965、step 9,250は2.751872だった。exp115のbestから合計0.002154改善しており、learning rateはstep 9,250で8.816e-7、経過時間は104.43秒だった。低学習率でvalidation lossは単調ではないものの、現時点では改善傾向を保っている。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 9,500〜10,000

step 9,500のvalidation lossは2.751341、step 9,750は2.750261、step 10,000は2.750426だった。step 9,750で現時点の最良値を更新し、exp115のbestから0.003765改善した。step 10,000のlearning rateは7.558e-7、経過時間は166.36秒だった。step 10,000でわずかな反発はあるが、継続学習によるvalidation lossの改善は続いている。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 10,250〜10,750

step 10,250のvalidation lossは2.750998、step 10,500は2.750023、step 10,750は2.748146だった。step 10,750で現時点の最良値を更新し、exp115のbestから0.005880改善した。step 10,750のlearning rateは6.375e-7、経過時間は227.13秒だった。学習率が下がる中でもvalidation lossは改善しており、同じSFTデータを追加で見せる効果がまだ残っている可能性がある。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 11,000〜11,500

step 11,000のvalidation lossは2.748025、step 11,250は2.747518、step 11,500は2.748593だった。step 11,250で現時点の最良値を更新し、exp115のbestから0.006507改善した。step 11,500では小さな反発が見られたが、直前までの改善幅は維持されている。step 11,500のlearning rateは5.292e-7、経過時間は288.44秒だった。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 11,750〜12,250

step 11,750のvalidation lossは2.747498、step 12,000は2.748418、step 12,250は2.748064だった。step 11,750で現時点の最良値をわずかに更新し、exp115のbestから0.006528改善した。step 12,250のlearning rateは4.333e-7、経過時間は349.26秒だった。step 11,250以降は改善がほぼ横ばいであり、継続学習の効果が飽和し始めた可能性がある。学習は安定している。

### 2026-09-07：step 12,500〜13,000

step 12,500のvalidation lossは2.748631、step 12,750は2.749006、step 13,000は2.748297だった。step 11,750のbest 2.747498は更新できず、step 13,000のlearning rateは3.518e-7、経過時間は409.08秒だった。低学習率の継続による改善はstep 11,750付近で頭打ちとなり、その後は小さな揺らぎの範囲に入った可能性が高い。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 13,250〜13,750

step 13,250のvalidation lossは2.748865、step 13,500は2.748030、step 13,750は2.748123だった。最良値はstep 11,750の2.747498のままで、step 13,750のlearning rateは2.865e-7、経過時間は469.24秒だった。step 13,000以降も大きな悪化はないが、追加学習によるvalidation改善は確認できない。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 14,000〜14,500

step 14,000のvalidation lossは2.748182、step 14,250は2.748196、step 14,500は2.748038だった。最良値はstep 11,750の2.747498から更新されず、step 14,500のlearning rateは2.388e-7、経過時間は529.04秒だった。後半はvalidation lossが2.748前後で安定しているが、exp115をさらに下回る新しい改善は確認できていない。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 14,750〜15,000

step 14,750のvalidation lossは2.747659、step 15,000は2.747452だった。step 15,000で最良値をわずかに更新し、exp115のbestから0.006573改善した。step 15,000のlearning rateは2.173e-7、経過時間は569.18秒だった。終盤に小さいながら改善が戻っているため、step 16,000まで記録を継続する。

### 2026-09-07：step 15,250〜15,750

step 15,250のvalidation lossは2.747539、step 15,500は2.747322、step 15,750は2.747276だった。step 15,750で現時点の最良値を更新し、exp115のbestから0.006750改善した。step 15,750のlearning rateは2.011e-7、経過時間は630.08秒だった。終盤に小さな改善が続いているが、改善幅は非常に小さい。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 16,000、学習完了

step 16,000のvalidation lossは2.747612、perplexityは15.6053だった。最良checkpointはstep 15,750で、validation lossは2.747276、perplexityは15.6001、重みSHA-256は `eaa3b779778be238ba5bbdfaae28bdabceb7e3c996971b5b20d1326c08870406` である。追加8,000 stepによって、exp115のbest 2.754026から0.006750改善した。学習時間は650.33秒、summaryに記録された全体時間は650.72秒、A40のpeak allocated memoryは1,489,078,784 bytesだった。NaN、OOM、shape errorなく完走した。

step 16,000までの全33本の生成文はRunpod上に保存されている。初期promptを`今日は天気がいいですね。`へ変更したため、exp115の固定promptとは直接比較せず、評価セットの生成結果を主比較に使う。

学習完了後、exp115と同じ評価器でbest checkpointを評価する。評価完了後に、step 15,750のbestとstep 16,000の周期checkpointの生成差も確認する。

### step 16,000 checkpoint評価時の指定ミス

step 16,000の周期checkpointを比較する評価で、domainと一般会話は完了したが、医療評価のcheckpoint引数へ誤って`step_016000.json`を指定したため、評価器がメタデータ形式エラーで停止した。学習checkpoint本体やbest評価には影響していない。原因は評価コマンドのファイル名指定ミスであり、正しい`step_016000.pt`を指定して医療評価を再実行する。

## 実験終了後の記録

### 評価結果

exp115と同じ評価器・評価データを使い、best checkpoint（step 15,750）と最終step 16,000の周期checkpointを比較した。bestのdomain lossはFineWeb2 2.904958、general 4.046684、conversation 2.002839、medical 1.873826だった。exp115の2.903494、4.047548、2.007362、1.876458と比べると、FineWeb2はわずかに悪化したが、general・conversation・medicalは改善した。SFT validation lossだけでなく、会話・医療のdomain lossにも追加stepの効果が残った。

bestの一般会話48例はEOS 48/48、平均生成長9.708 tokens、token overlap F1 0.249142だった。EOSはexp115と同じだが、F1はexp115の0.265719から低下した。出力には「そうなんですね。最近はどんな映画を?」のように会話を続けようとするものがある一方、「わけわかりません」のように文脈を壊す応答も残った。したがって、validation lossやdomain lossの改善だけで自然な会話が改善したとは判断できない。

bestの医療162例はEOS 158/162、平均生成長23.975 tokens、token overlap F1 0.242326だった。正解形式は162/162例で抽出でき、完全一致は33/162、20.37%だった。exp115の26/162、16.05%から7例改善し、実験113の33/162と同じ水準へ戻った。ただしF1はexp115の0.264921より低く、理由の説明が短くなった。正解を当てた例でも、理由が医学的に正しいとは限らず、「正解はeです。理由は○e 聴覚過敏 聴覚過敏は聴覚過敏であり…」のような反復が残っている。

最終step 16,000ではFineWeb2 2.904876、general 4.046810、conversation 2.002643、medical 1.873839だった。一般会話はEOS 48/48、平均10.167 tokens、F1 0.243196、医療はEOS 159/162、平均23.821 tokens、F1 0.249731、完全一致33/162だった。bestと比べて医療EOSとF1はわずかに改善したが、一般会話F1はさらに低かった。総合比較の基準checkpointは、validation lossが最小であるbest step 15,750とする。

今回の仮説は部分的に支持された。同じSFTデータを低学習率で追加8,000 step見せると、validation loss、conversation loss、medical loss、医療の完全一致は改善したため、exp115時点で学習が完全に飽和していたわけではない。しかし、一般会話F1は悪化し、自然な応答の文脈適合も明確には向上しなかった。追加stepは「医療の選択肢を正解形式で返す能力」と「domain loss」には有効だが、「自然な一般会話」を同時に改善する方法ではないと判断する。

この結果から、同じデータをさらに反復する実験は一旦止める。次は、一般会話の学習データについて、短い相槌・質問の繰り返し・文脈から外れた発話を減らし、応答が十分に長く、直前の話題へ具体的に反応する例を選ぶ。応答長だけを人工的に増やすのではなく、質問と回答の関係、話者交代、重複、定型句の割合をmanifestへ記録し、自然会話データの質を変えた効果をexp116 bestから短いSFTで比較する。医療QAは同じ選択肢正答データを過度に増やさず、一般会話の自然さを主指標とする。

### 保存した成果物

最良checkpointの重み本体はGitHubへ追加せず、重みSHA-256 `eaa3b779778be238ba5bbdfaae28bdabceb7e3c996971b5b20d1326c08870406` を記録した。metadata、metrics、評価全文、学習中の33本の生成文、学習ログ、評価ログはGit管理へ追加する。

- `best.json`：`be327e34c2dfad980e370b1e699ccf79d909853c11d8b4701bc9927cd5ee0b17`
- `summary.json`：`b4436e94481326a7355773e9d26bd2936cdfd6a99f0c9b54645ad9222d914492`
- `metrics.jsonl`：`05e446439b75ae920ec82a9b527d78871f18c26f54686ab0966b576d3512b4a2`
- best domain評価：`1451d6043f0738d1684af0b8a4d57413dd45619bff26123b1ce6df96f411584f`
- best一般会話評価：`e017f9fd1e66023881897691d10627065a9d387261e26435fac0ba1f778d0347`
- best医療評価：`9218c326f91310eee50ec1ce41172ded27941cb8aa0654dfe50b54bf47efdaff`
- final domain評価：`aebcd7b434ff04450f7c0fbe6a4b9c343859676610e9bdda49bbb1153aa5fd35`
- final一般会話評価：`254753fc3caa1644503822021d2fd4e7032c3a85df27d10877eadb05ff9603be`
- final医療評価：`ed94a33913950bfa1ae5717202b94a744d0c889ccbd60abaf527a6ead628e90d`
- 学習ログ：`e9a42b4529e2e56d46d3dfe2f7b60152f7d0c229f98b9e086a7f6b6d29645917`
- best評価ログ：`bc9484c02ff96b3ec7080fcc692c1693a0ded268b76ac3c1d48e3942129b49c8`
- final評価ログ：`d10a0981d780ba8a3328e6ff40bff4d01eb26a38b161c8c53fcddef6902c84e2`
- final医療再評価ログ：`ed94a33913950bfa1ae5717202b94a744d0c889ccbd60abaf527a6ead628e90d`
