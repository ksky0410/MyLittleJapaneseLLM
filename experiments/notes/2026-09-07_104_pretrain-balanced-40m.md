# 実験104：初回20Mと追加20Mを混ぜた継続事前学習

## 実施前の計画

- 実施日：2026-09-07
- 担当：Codex
- 状態：準備中、実行前
- 使用ブランチ：`main`
- 目的：教師LLMによる蒸留を使わず、日本語の知識と文章生成能力を増やす。前回の追加20M tokenだけの継続事前学習で会話・医療形式を忘れた原因が、データの偏りにあるかを確認する。

実験101では追加FineWeb2・Wikipediaの約20M tokenだけを40,000 step学習し、FineWeb validationは改善したが、会話lossが2.18から3.86へ、医療lossが2.02から2.27へ悪化した。今回は、実験098で使った初回約20M tokenと追加約20M tokenを約262k tokenずつ交互に並べ、片方のコーパスへ連続して偏らない40M token列を作る。

### 仮説

初回コーパスと追加コーパスを交互に混ぜると、追加コーパスだけを使うよりFineWeb validationの改善は小さくなるかもしれない。しかし、データ分布の急変を抑え、初回学習で得た日本語の分布を保ちやすくなると予想する。継続事前学習後にFineWeb、一般、会話、医療のlossを測り、会話形式が壊れていないかを確認する。

### 開始前の条件

- 初期checkpoint：実験101の継続事前学習最良checkpoint `artifacts/checkpoints/issue1-both-50m-pretrain-continuation-20m-40k-runpod-cuda-lr3e-5/best.pt`
- 初期checkpoint SHA-256：`6057172a5a2b3b420c5c751388eead0b17e0dfaa41585259265859ab9bf016b4`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 初回token列：`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-20m-v1-train.bin`、19,987,750 tokens、SHA-256 `707561109835bed5bccd8126527debeb86e86f0e6daaa9abea4ea78ba63e54e6`
- 追加token列：`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-continuation-20m-v1-train.bin`、19,993,334 tokens、SHA-256 `f19878618870a487ce5b0aab6970d6d72b2ef71ab76ee79520e7c3fe3341dec1`
- 連結後token列：約39,981,084 tokens。生成後のmanifestへ正確なtoken数とSHA-256を記録する。
- validation：`artifacts/tokens/fineweb2-edu-japanese-v1-test.bin`、2,061,459 tokens、SHA-256 `36d8d5c8bc92de1e168b8c3de9dd4ee975dec66f6b644b83bfbf9b239877161c`
- tokenizer：`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- token列の交互chunk長：262,144 tokens
- 学習設定：`configs/issue1-both-50m-pretrain-balanced-40m-runpod-5k.toml`
- 乱数seed：104
- バッチサイズ：8、context 256
- 学習率：1e-5から1e-6までのcosine decay、warmup 250 steps
- 予定step：5,000 steps

### 成功判定

FineWeb validation lossが実験101の2.835023を維持または改善し、会話・医療lossが実験101の3.860727・2.266281より明確に低くなることを第一の成功条件とする。一般会話生成のEOS率と固定prompt生成も保存する。raw継続事前学習だけで会話能力が壊れる場合は失敗として記録し、SFT前提の二段階手順へ進む。

## 実験中の記録

データ生成、学習開始、500 step以上の節目ごとに、このノートへ実測値を追記する。

## 実験終了後の記録

未実施。学習終了後に最良checkpoint、全領域loss、生成結果、実行時間、次のSFT方針を追記する。
