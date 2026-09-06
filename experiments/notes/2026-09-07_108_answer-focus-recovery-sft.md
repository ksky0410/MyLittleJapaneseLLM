# 実験108：answer-focus後の理由付き応答回復SFT

## 実施前の計画

- 実施日：2026-09-07
- 担当：Codex
- 状態：準備中
- 使用ブランチ：`main`

実験107では、実験105の最良checkpointへ医療問題ごとの短い回答「正解は○です。」を追加した。一般会話の固定48例F1は0.2086から0.2331へ改善したが、医療162問の正解率は17.90%から14.20%へ下がり、平均生成長も56.10から25.54 tokensへ急減した。短い回答例が医療の説明付き回答形式を壊した可能性がある。

今回は実験107の最良checkpointから、answer-focus例を含めない元の一般・医療SFTデータへ1,000 stepだけ戻す。これにより、実験107で得られた一般会話の状態をできるだけ引き継ぎながら、理由付き回答と適切なEOS位置を回復できるかを検証する。

### 仮説

元の理由付きSFTを短期で再学習すれば、医療回答の平均生成長とtoken overlap F1は実験107より回復し、正解率も14.20%を上回ると予想する。一般会話F1は実験107の0.2331から少し下がる可能性があるが、実験105の0.2086を大きく下回らず、EOS 48/48を維持すると予想する。医療の正答率が回復しない場合、問題は応答形式ではなく、選択肢を正しく識別する知識・推論不足であり、SFTの短期反復を続ける根拠にはしない。

### 開始前の条件

- 初期checkpoint：実験107の最良 `artifacts/checkpoints/issue1-balanced-pretrain-answer-focus-sft-runpod-1k/best.pt`
- 初期checkpoint SHA-256：`080d6e30f33b3464fdf470d674edee48f47e59c9117cafba0c5120532f8cee44`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 学習データ：実験105と同じ一般127,731例・医療2,945例、`artifacts/sft/issue1-general-medical-concat-v1/train.npz`、SHA-256 `598c464b03cd94a9c5579552df5f78059410f8ce5721da6cc93acb8251382cf4`
- validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`、比率20%
- 学習設定：`configs/issue1-balanced-pretrain-answer-focus-recovery-sft-runpod-1k.toml`
- 学習設定SHA-256：`3b791b4656c28e372981fcb48abbb3ead2a973bff1e92763b95c5f9a26e4f5d9`
- 学習コード：`scripts/train_sft_torch.py`、SHA-256 `bc78ec94a7f74399d049ce4d1f6a22b446437a90b8e855bf64233b935267974e`
- 乱数seed：108
- 学習率：3e-6から3e-7までのcosine decay、warmup 100 steps
- 予定step：1,000 steps

### 実行コマンド

Runpod Pod `j9c46julmtbcb4` のA40上で、実験107と同じPyTorch CUDA版を使う。実験107のcheckpoint、元のtrain/validation NPZ、rehearsal token列をSHA-256照合してから開始する。

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_sft_torch.py \
  --config configs/issue1-balanced-pretrain-answer-focus-recovery-sft-runpod-1k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-balanced-pretrain-answer-focus-sft-runpod-1k/best.pt \
  --train-data artifacts/sft/issue1-general-medical-concat-v1/train.npz \
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-balanced-pretrain-answer-focus-recovery-sft-runpod-1k \
  --samples-dir artifacts/samples/issue1-balanced-pretrain-answer-focus-recovery-sft-runpod-1k \
  --device cuda \
  --sample-template conversation \
  --sample-speaker-a DA \
  --sample-speaker-b DC \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.2
```

### 成功判定

医療162問の正解率が実験107の14.20%を上回り、平均生成長が25.54 tokensから回復することを第一条件とする。一般会話48例のEOS 48/48を維持することも必須とする。validation lossだけが下がり、医療正答率が改善しない場合は、応答形式の回復には成功したが知識・識別能力の改善には失敗したと判断する。

## 実験中の記録

学習中は250 stepごとにvalidation loss、学習率、生成例、警告・停止の有無を追記する。短答に戻る出力や、理由の反復・崩壊も削除せず保存する。

### 本走：step 1,000まで完了

step 1のvalidation lossは2.911438、step 250は2.911840、step 500は2.908120、step 750は2.908841、step 1,000は2.908092だった。step 500付近で改善し、その後も大きな発散はなかった。NaN、OOM、shape errorは発生せず、学習時間は84.05秒、ピークGPU allocated memoryは1,490,586,112 bytesだった。混合validation lossは実験107の2.911412を下回ったが、固定生成の評価前である。

## 実験終了後の記録

### 2026-09-07：学習・固定評価完了

実験107の最良checkpointから、元の一般127,731例・医療2,945例を連結した130,676例で1,000 step学習した。rehearsal比率は0.20、seedは108、学習率は3e-6から3e-7、batch sizeは8である。最良checkpointはstep 1,000、SHA-256は `14e6a3715c8f2dc91beb2256b2cc15f60c3a26d7260973405da65d64e71a5d8a` である。学習時間は84.05秒、ピークGPU allocated memoryは1,490,586,112 bytes、NaN・OOM・shape errorはなかった。

混合SFT validation lossは2.908092、perplexityは18.3218だった。領域別lossはFineWeb 2.923107、general 4.082645、conversation 2.068468、medical 1.975622である。実験107との比較では、FineWeb 2.922944からわずかに悪化、general 4.083455から改善、conversation 2.068766から改善、medical 1.976274から改善した。

固定一般会話48例ではEOS 48/48、平均生成9.58 tokens、token overlap F1 0.2359だった。実験107のF1 0.2331をわずかに上回り、実験105の0.2086も上回った。医療162例では回答形式の抽出が162/162、EOS 157/162、平均生成51.44 tokens、token overlap F1 0.3681、正解率28/162、17.28%だった。実験107の14.20%からは回復し、平均長とF1も大きく戻ったが、実験105の17.90%、実験106の19.14%には届かなかった。

したがって、answer-focus追加SFTの後に元の理由付きSFTへ戻すことで、短すぎる医療回答と説明形式は回復できることが分かった。一方、医療の正解選択肢を当てる能力は回復しても実験105・106を超えず、今回の二段階SFTを総合モデルの改善策として採用する根拠はない。医療正答率はvalidation loss、回答長、語句一致とは別に動いており、次はSFT反復ではなく、事前学習データ量・問題の選択肢識別形式・評価時の生成安定性を分けて検証する。

生成JSON/TXTは `artifacts/evaluations/exp108/`、学習中のstep別サンプルは `artifacts/samples/issue1-balanced-pretrain-answer-focus-recovery-sft-runpod-1k/`、metricsとsummaryは `artifacts/checkpoints/issue1-balanced-pretrain-answer-focus-recovery-sft-runpod-1k/` に保存する。
