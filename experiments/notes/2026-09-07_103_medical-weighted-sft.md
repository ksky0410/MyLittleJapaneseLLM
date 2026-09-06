# 実験103：医療QAを4倍にした一般・医療混合SFT

## 実施前の計画

- 実施日：2026-09-07
- 担当：Codex
- 状態：準備完了、パイロット実行前
- 使用ブランチ：`main`
- 開始時点のコード：`c2ab7fc`（`exp: prepare medical weighted sft`）
- 使用GPU：Runpod A40 Pod `j9c46julmtbcb4`
- 目的：教師LLMによる蒸留を使わず、医師国家試験形式の問題に対する回答の正確さを高めつつ、一般的な日本語会話能力を維持できるかを確かめる。

実験102では一般会話と医療QAを混ぜたSFTにより、文章の終了、回答形式、問題文との語句の一致が大きく改善した。一方、医療問題の正解率は、抽出できた161例中37例、22.98%にとどまった。そこで今回はデータの内容を変えず、医療QAの提示回数だけを4倍にする。

### 仮説

医療QAを4倍にすると、医療問題の選択肢や「正解は…」という回答形式をより安定して学習し、実験102の22.98%を上回る可能性がある。ただし、一般会話の学習例を相対的に減らすため、一般会話の自然さや終了の安定性が悪化する可能性もある。医療回答の改善だけでなく、一般会話48例、医療問題162例、4領域のvalidation lossを同じ条件で比較し、採用可否を判断する。

### 開始前の条件

- 初期checkpoint：実験102のフルSFT最良checkpoint `artifacts/checkpoints/issue1-general-medical-50m-sft-runpod-8k/best.pt`
- 初期checkpoint SHA-256：`a3404756796e5ecdaa33b09ebf11fb5fafa4660007ef6ae48d31f078ca647acc`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 学習データ：一般会話127,731例に、医療QA 2,945例を同じデータのまま4回反復した11,780例を連結
- 一般会話データ：`artifacts/sft/issue1-quality-aware-770k-each-v1/train.npz`、SHA-256 `fc2a9367a16c0cf963fbc2e96530a9944d41e4d2d8e4029874dface2d71b5d57`
- 医療QA原データ：`artifacts/sft/medical-qb-sft-v1/train.npz`、SHA-256 `fe1db09546363145845e219f71e05dfa6d8ceb006dfa2093fca302da2d2b8b07`
- 反復後医療QA：11,780例、応答トークン690,180、SHA-256 `ef574b78e1991fd52f6ab5f1dfa5c6a5098a5f80258aca023ffcc4d352a61678`
- 連結後学習データ：139,511例、応答トークン2,232,155、`artifacts/sft/issue1-general-medical-heavy-v1/train.npz`、SHA-256 `0756ec956f98f67dd0b712bbf9b94a217bbb92c42dffe4e72c989f117421f493`
- 連結データのmanifest：`artifacts/sft/issue1-general-medical-heavy-v1/train.manifest.json`
- validation：実験102と同じ一般・医療混合validation、`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- 事前学習リハーサル：`artifacts/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`、混合比率20%
- tokenizer：`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- 学習設定：`configs/issue1-general-medical-heavy-50m-sft-runpod-3k.toml`
- 乱数seed：303
- バッチサイズ：8
- 学習率：5e-6から5e-7までのcosine decay、warmup 200 steps
- 評価・生成間隔：250 steps
- checkpoint保存間隔：500 steps
- 予定する累積step：3,000 steps

### 実行手順

まず1,000 stepsのパイロットを実行する。lossが有限で、医療・一般の短い評価で明らかな崩壊がない場合、パイロット最良checkpointから累積3,000 stepsまで継続する。学習中は各checkpointのlossと固定promptの生成文を保存し、少なくとも500 stepsごとにこのノートへ追記する。

### 成功判定

主指標は、実験102と同じ162問で計算する「正解は…」形式の抽出後の正解率であり、22.98%を上回ることを第一条件とする。併せて、医療生成のEOS割合と一般会話48例のEOS割合・語句一致率を比較する。医療正解率だけが上がっても一般会話の終了が大きく崩れる場合は、そのまま採用せず、トレードオフとして記録する。

## 実験中の記録

### 2026-09-07：データ準備とテスト

医療QAの反復処理を実装した。最初の実装では入力、教師、損失マスクをそれぞれ別の乱数順で並べ替える問題があったため、同じ行対応を保つ共通の並べ替え順を使うよう修正した。`PYTHONPATH=scripts uv run pytest -q` は113件すべて成功した。

反復後のデータは、元の2,945例から11,780例になり、応答トークンは172,545から690,180になった。一般会話を合わせた学習データ全体では、一般会話の例数127,731に対して医療例11,780となる。例数では医療が約8.4%、応答トークンでは約30.9%であり、実験102の約10.1%から医療応答の比率を大きく上げた。

### 2026-09-07：1,000 stepパイロット

実験102の最良checkpointから開始し、同じRunpod A40で1,000 stepまで学習した。学習時間は81.95秒、ピークGPUメモリは約1.49GBだった。NaN、OOM、shape errorは発生しなかった。最良checkpointはstep 1,000で、validation lossは2.892228、perplexityは18.0334だった。実験102の最良値2.894461よりわずかに低く、少なくとも短いパイロットではvalidation全体を悪化させていない。

固定promptの生成は、step 0からstep 1,000にかけて「こんにちは!」から「こんにちはー。」へ変化し、会話の終了記号も維持された。生成ファイルは `artifacts/samples/exp103-pilot/` に保存した。

同じ評価条件で比較すると、一般会話48例はEOS 48/48、平均生成7.83 tokens、token overlap F1 0.1880だった。実験102のEOS 48/48、平均7.90 tokens、F1 0.1892とほぼ同じである。医療162例はEOS 154/162、平均生成54.02 tokens、F1 0.3799となった。回答形式は162/162例で抽出でき、正解は39例、正解率は24.07%だった。実験102の37/161、22.98%からは改善したが、差は小さいため、効果が確定したとは扱わない。

領域別lossはFineWeb 2.955133、一般 4.089521、会話 2.096207、医療 1.986751だった。実験102の会話2.100814、医療1.987503に対して少し改善しており、パイロット時点では一般会話の自然な終了を損なわずに医療比重化できている。

パイロットは成功判定を満たしたため、step 1,000の最良checkpoint `d427cf35922d91c5a64a7437edb3d019492b15c8c4016493c68ec9e72836e1bd` から、同じ学習率スケジュールのまま累積3,000 stepまで継続する。再開時のstepを失わないようSFTスクリプトに `--start-step` を追加し、1,001から3,000までを実行する。一般会話・医療会話・領域別lossのパイロット結果は `experiments/results/exp103/` に保存した。3,000 stepで医療正解率が伸びない場合は、医療QAの反復倍率をさらに増やすのではなく、まず学習率とvalidation指標の関係を確認する。

## 実験終了後の記録

未実施。パイロットと本実験の終了後に、実際のstep、loss、checkpoint、生成結果、評価値、停止理由、次の変更を追記する。
