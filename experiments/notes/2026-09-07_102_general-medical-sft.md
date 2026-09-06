# 実験102：50Mモデルの一般会話・医師国家試験混合SFT

## 目的

実験101の継続事前学習で得る50M日本語モデルに対して、一般会話の自然さと質問回答能力を同時に改善できるかを調べる。一般会話だけでSFTすると会話の型は身につきやすい一方、質問に対して内容のある回答を返す能力が不足する可能性がある。そこで、既存の品質管理済み会話データへ医師国家試験の質問回答データを加え、応答部分だけをSFTの損失対象にする。

## 事前仮説

一般会話データに医師国家試験データを少量加えることで、会話の自然さを大きく損なわず、明示的な質問に対して「正解と理由」を返す能力が改善すると予想する。ただし、医療問題は一般会話と文体・語彙が異なるため、医療データを過度に繰り返すと雑談の自然さが悪化する可能性がある。本実験では医療データを重複複製せず一度だけ連結し、事前学習トークンを20%のrehearsalとして混ぜて過学習と知識忘却を抑える。

## 実験前の条件

- 実施日：2026年9月7日予定
- 担当：Codex
- 初期checkpoint：実験101の低学習率継続事前学習のbest checkpoint
- モデル：dim 576、12層、9 heads、RoPE、LayerNorm、SwiGLU、context 256、約50.2M parameters
- Tokenizer：`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`
- tokenizer SHA-256：`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- 学習データ：一般会話 `issue1-quality-aware-770k-each-v1/train.npz` と医師国家試験 `medical-qb-sft-v1/train.npz` の連結
- 学習データの例数：一般会話127,731例、医療2,945例、合計130,676例
- 応答対象Token数：一般会話1,541,975、医療172,545、合計1,714,520
- 検証データ：一般会話validationと医療validationの連結。合計49,207例、応答対象747,937 tokens
- 学習配列SHA-256：`598c464b03cd94a9c5579552df5f78059410f8ce5721da6cc93acb8251382cf4`
- 検証配列SHA-256：`95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- 学習配列manifest SHA-256：`4c5e4f458a281598fd2c57726064be4164c26a2032bc7d5030f968c642ad94cd`
- 検証配列manifest SHA-256：`bd1b1a097f988be0dca4fcc84b929be032f0deb29afb69daf9c7abf495eebc47`
- 設定ファイルSHA-256：`af7953a1c5dfec5bbcd06772c4d07dacdfba3427db2968a845cc9dac5161d756`
- 学習：batch size 8、最大8,000 step、eval/sample 250 stepごと、checkpoint 500 stepごと
- optimizer：AdamW、learning rate `2e-5`から`2e-6`へのcosine decay、warmup 200、weight decay 0.01、seed 202
- rehearsal：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、ratio 0.20
- GPU：実験101と同じRunpod A40 Secureを予定

## 成功基準

一般会話・医療のvalidation loss、固定会話プロンプト、新規会話評価、医療質問評価を実験101のbest checkpointと比較する。一般会話の自然さを保ったまま医療質問の正解率または回答形式が改善し、NaN・OOM・shape errorなく完走することを有望な結果とする。validation lossだけで採用checkpointを決めず、生成結果を必ず人手確認用に保存する。

## 実行前の準備

`scripts/concat_sft_npz.py`で一般会話と医療SFT配列を連結した。元の医師国家試験データは読み取り専用で扱い、元ディレクトリを変更していない。大きなNPZ本体はGitへ追加せず、入力hash・出力hash・例数は`artifacts/sft/issue1-general-medical-concat-v1/*.manifest.json`へ保存した。

学習開始前に、実験101のbest checkpointのSHA-256、Runpod Pod ID、設定ファイルSHA-256、実際の実行コマンドをこのノートへ追記する。実験101が完了するまでは本実験を開始しない。

## 開始前の実行記録

実験101が累積step 40,000まで完走し、step 38,000のbest checkpointを回収したため、本実験を開始する。初期checkpointのSHA-256は`6057172a5a2b3b420c5c751388eead0b17e0dfaa41585259265859ab9bf016b4`、設定ファイルのSHA-256は`af7953a1c5dfec5bbcd06772c4d07dacdfba3427db2968a845cc9dac5161d756`である。Runpod Podは実験101と同じ`j9c46julmtbcb4`（A40 Secure、CA-MTL-1）を使う。

まず同じ条件のpilotを最大1,000 stepで実行する。pilotの出力は本番と分離して`artifacts/checkpoints/exp102-pilot`と`artifacts/samples/exp102-pilot`へ保存し、validation loss、EOS到達率、生成文を確認する。pilotでNaN、OOM、loss急上昇、出力形式の明らかな崩壊がなければ、同じ初期checkpointからではなくpilotの重みを引き継いで本番8,000 stepへ進む。本番では、SFTの学習率scheduleをpilot後も連続させる。

### pilot実行コマンド

```bash
PYTHONPATH=scripts python3 scripts/train_sft_torch.py \
  --config configs/issue1-general-medical-50m-sft-runpod-8k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-continuation-20m-40k-runpod-cuda-lr3e-5/best.pt \
  --train-data artifacts/sft/issue1-general-medical-concat-v1/train.npz \
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \
  --output-dir artifacts/checkpoints/exp102-pilot \
  --samples-dir artifacts/samples/exp102-pilot \
  --max-steps 1000 \
  --lr-schedule-steps 8000 \
  --rehearsal-tokens artifacts/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 \
  --sample-template conversation \
  --sample-speaker-a A \
  --sample-speaker-b B \
  --device cuda
```

### pilot途中経過

Runpod上のpilotはPID 2223で正常に実行中である。step 250の混合validation lossは3.133442、step 500では3.094915まで下がった。step 500のSFT train lossは3.257999、rehearsal train lossは1.999201、学習率は`1.9935e-5`である。NaN、OOM、shape errorは発生していないため、1,000 stepまで継続する。なお、このvalidation lossは一般会話と医療を連結したSFT validationの値であり、実験101のraw domain lossとは直接比較しない。

### pilot完了結果

pilotはstep 1,000まで完走し、best checkpointもstep 1,000だった。validation lossは3.067110、Perplexityは21.479729、SFT train lossは2.641896、rehearsal train lossは3.052814である。A40での経過時間は83.35秒、ピークGPU allocated memoryは約1.49GBだった。初期値からNaN、OOM、shape errorは発生していない。

pilot bestを用いた小規模生成評価では、一般会話12例のEOS到達が12例、平均生成Token数が12.0、token overlap F1が0.1742だった。医療validation 20例ではEOS到達が19例、平均生成Token数が52.4、token overlap F1が0.3212だった。実験101 continuationの一般48例EOS 16例、医療162例EOS 58例と比べ、SFTによって終了形式が明確に回復したため、本番8,000 stepへ進む。

pilotのsummary、metrics、生成文、評価結果をRunpod上の`artifacts/checkpoints/exp102-pilot`、`artifacts/samples/exp102-pilot`、`evaluations/exp102-pilot-*`へ保存した。本番はpilot bestを初期checkpointに使い、学習率scheduleの終点を8,000 stepとして連続させる。

### 本番途中経過（step 2,000）

本番プロセスはPID 2494で継続中である。step 250、500、750、1,000、1,250、1,500、1,750、2,000のvalidation lossはそれぞれ3.083431、3.092302、3.111439、3.103841、3.072422、3.055892、3.048445、3.034527となった。step 2,000で本番開始後のbestを更新し、Perplexityは20.791146、SFT train lossは2.282514、rehearsal train lossは2.297451、学習率は`1.7739e-5`である。

step 750付近の一時的な悪化から回復し、pilot bestのvalidation loss 3.067110も下回ったため、現時点では早期停止せず8,000 stepまで継続する。後半で再びvalidationが悪化した場合はbest stepのcheckpointを採用する。

step 2,250、2,500、2,750のvalidation lossは3.016256、3.011978、3.004206となった。step 2,750のPerplexityは20.170203、SFT train lossは2.366128、rehearsal train lossは2.244631、学習率は`1.5659e-5`である。step 2,750でbestを更新しており、validationは安定して改善している。

step 3,000、3,250、3,500、3,750のvalidation lossは3.007122、2.998693、2.989079、2.974693となった。step 3,750のPerplexityは19.583609、SFT train lossは1.622862、rehearsal train lossは2.367937、学習率は`1.2268e-5`である。step 3,750でbestを更新し、pilot終了時よりvalidation lossが約3.0%下がった。step 3,000の小さな反発後も改善へ戻っている。

step 4,000、4,250、4,500のvalidation lossは2.972452、2.957586、2.953993となった。step 4,500のPerplexityは19.182388、SFT train lossは2.792804、rehearsal train lossは2.818789、学習率は`9.5599e-6`である。step 4,500でbestを更新し、SFT開始時のvalidation loss 3.067089から約3.7%低下した。

### 本番実行コマンド

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=scripts python3 scripts/train_sft_torch.py \
  --config configs/issue1-general-medical-50m-sft-runpod-8k.toml \
  --base-checkpoint artifacts/checkpoints/exp102-pilot/best.pt \
  --train-data artifacts/sft/issue1-general-medical-concat-v1/train.npz \
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-general-medical-50m-sft-runpod-8k \
  --samples-dir artifacts/samples/issue1-general-medical-50m-sft-runpod-8k \
  --max-steps 8000 \
  --lr-schedule-steps 8000 \
  --rehearsal-tokens artifacts/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 \
  --sample-template conversation \
  --sample-speaker-a A \
  --sample-speaker-b B \
  --device cuda
```

## 実行コマンド（予定）

```bash
PYTHONPATH=scripts uv run python scripts/train_sft_torch.py \
  --config configs/issue1-general-medical-50m-sft-runpod-8k.toml \
  --base-checkpoint <experiment-101-best.pt> \
  --train-data artifacts/sft/issue1-general-medical-concat-v1/train.npz \
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-general-medical-50m-sft-runpod-8k \
  --samples-dir artifacts/samples/issue1-general-medical-50m-sft-runpod-8k \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 \
  --sample-template conversation \
  --sample-speaker-a A \
  --sample-speaker-b B \
  --device cuda
```

## 結果

実験101の終了後、実行条件、途中metrics、best checkpoint、生成サンプル、一般会話評価、医療質問評価、停止理由を追記する。失敗した場合も出力を削除せず、そのまま記録する。
