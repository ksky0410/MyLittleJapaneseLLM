# 実験101：Runpod 50M継続事前学習の低学習率比較

## 目的

実験100では、Runpod 40,000 step事前学習モデルへ未使用20M-tokenコーパスを追加する継続事前学習を開始したが、最大学習率`1e-4`へ到達した後に既存FineWeb validation lossが悪化した。実験101では初期checkpointと新規token列を固定し、最大学習率だけを`3e-5`へ下げる。これにより、既存の日本語能力を壊さずに新しい文書を取り込めるかを確認する。

## 事前仮説

継続事前学習ではランダム初期値からのpretrainingより小さい学習率が適切であり、`3e-5`ならstep 500以降のvalidation悪化が弱まる、またはvalidation lossが2.9未満へ下がると予想する。もし低学習率でも改善せず、新規データの分布とvalidationのずれが主因なら、学習率を下げてもlossの改善は限定的になる。

この実験も会話SFTではないため、質問への直接回答や自然な雑談の改善は別途評価する。事前学習終了後、良いcheckpointを会話SFTの基盤へ渡す。

## 条件

- 実施日：2026年9月7日
- 担当：Codex
- 初期checkpoint：実験098の`artifacts/checkpoints/issue1-both-50m-pretrain-20m-40k-runpod-cuda/best.pt`
- 初期checkpoint SHA-256：`83e8be941b645823efd1ae0a358d2c4521faa49b58de7696229298973bd25ac7`
- 追加train binary：`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-continuation-20m-v1-train.bin`
- 追加train binary SHA-256：`f19878618870a487ce5b0aab6970d6d72b2ef71ab76ee79520e7c3fe3341dec1`
- 追加tokens：19,993,334
- モデル：dim 576、12層、9 heads、RoPE、LayerNorm、SwiGLU、context 256、50,207,616 parameters
- 学習：batch size 8、40,000 step、約81.92M提示tokens、seed 101
- optimizer：AdamW。重みだけを初期checkpointから読み込み、optimizer stateは初期化
- 学習率：`3e-5`から`3e-6`までcosine decay、warmup 1,000 step、weight decay 0.1
- 評価：FineWeb2 Japanese testを500 stepごとに20 evaluation batchesで測定し、同じpromptの生成文を保存
- GPU：Runpod A40 Secure。前回実測速度は約14.5 step/秒、約$0.49/時
- 設定SHA-256：`a719b97d30c602076ace58c238d647b19b3160dd81f2a0f2ce9fc3455d62463b`
- Runpod Pod：`j9c46julmtbcb4`（CA-MTL-1）
- 転送bundle SHA-256：`beb3beb7339d2e81946d1fca61f1234ef3cb3bb1e97db5a2a750c8a22c3efac4`

## 実行前の成功基準

step 500以降のFineWeb validation lossが実験100のstep 500 `2.927500`から悪化せず、40,000 stepまでに実験098のbest `2.973267`を安定して下回ることを有望な結果とする。validationが改善しても、固定chat-testの生成全文と新しい会話testで自然さを確認する。NaN、OOM、shape errorなく完走し、500 stepごとのmetrics・生成文・checkpoint metadataを保存する。

## 開始前の実行記録

このノートは実験100の高学習率条件を停止した直後に作成する。実験100の失敗結果、データ準備、入力hashは上書きせず、実験101の出力先を分離する。開始前に設定のSHA-256、Runpod Pod ID、bundle SHA-256、GPUを追記し、学習開始後は500 stepを超えて記録を空けない。

### 開始前の追加準備

医師国家試験データを後段SFTで使える質問回答形式へ変換する`prepare_medical_sft.py`とテストを追加した。元の`artifacts/corpus/medical-qb-v2`は読み取り専用で扱い、正解欄が空の問題と、context 256で質問または回答が切れる問題を学習用から除外する。変換後のSFT候補はtrain 2,945例、response 172,545 tokens、validation 162例、response 9,277 tokensとなり、SFT配列の`truncated_example_count`はtrain・validationとも0である。除外件数と問題番号は`artifacts/corpus/medical-qb-sft-v1/manifest.json`へ記録した。

Runpodの同じA40 Pod `j9c46julmtbcb4`上で、実験101の2,000 step pilotを本番と別の`exp101-pilot`へ実行した。step 500、1,000、1,500、2,000のFineWeb validation lossはそれぞれ2.931313、2.914403、2.902358、2.896122となり、step 1の2.973276から一貫して改善した。step 2,000のbest checkpoint SHA-256は`e2e23d652fd365716c5f97b68f8da8144332aa1ddb3d9eba465b3f4fe229759f`である。高学習率の実験100とは異なり、step 500以降にvalidation lossが悪化しなかったため、このpilotを有望と判断する。metrics、best metadata、生成サンプルは`artifacts/checkpoints/issue1-both-50m-pretrain-continuation-20m-40k-runpod-cuda-lr3e-5-pilot/`と対応するsamplesディレクトリへ回収した。

本番はpilotのstep 2,000 best重みから`--start-step 2000`で継続し、累積40,000 stepまで学習する。これによりpilotの2,000 stepを捨てず、学習率scheduleと乱数の進行も連続させる。

### 本番開始の追記

pilotの入力と結果を確認後、同じRunpod Pod `j9c46julmtbcb4`で本番プロセスをPID 808として起動した。学習ログはPod上の`/workspace/exp100/exp101.log`、軽量metricsと生成文は設定どおりの`issue1-both-50m-pretrain-continuation-20m-40k-runpod-cuda-lr3e-5`ディレクトリへ保存する。pilotのstep 2,000 best重みを初期値に使い、`--start-step 2000`で累積stepを継続している。終了までPodを保持し、途中で500 step以上記録を空けない。

## 実行コマンド

```bash
PYTHONPATH=scripts uv run python scripts/train_torch.py \
  --config configs/issue1-both-50m-pretrain-continuation-20m-40k-runpod-cuda-lr3e-5.toml \
  --initial-checkpoint artifacts/checkpoints/exp101-pilot/best.pt \
  --start-step 2000 \
  --device cuda
```

## 結果

学習中にmetrics、生成、checkpoint、GPU速度、料金、停止理由を追記する。終了後にbest step、FineWeb・各domain validation、固定chat-test、新規会話test、人手レビュー用サンプルを記録し、会話SFTへ渡すcheckpointを明記する。

### 途中経過（2026年9月7日、累積step 5,500）

Runpod上の本番プロセスは継続中で、NaN、OOM、shape errorは発生していない。pilotのstep 2,000 best重みから再開した後、FineWeb validation lossはstep 2,500の2.894975からstep 5,500の2.876570まで緩やかに改善した。step 5,000では2.878091、step 5,500ではPerplexity 17.753284である。実験098のbest validation loss 2.973267と比べ、現時点でlossは約3.3%低く、Perplexityは約9%低い。ただし、これは事前学習評価の改善であり、自然な会話応答の改善を意味しないため、後段の会話SFTと生成比較を省略しない。

step 5,500時点のtrain lossは2.395376、学習率は`2.9123e-5`である。step 5,000から5,500までの経過時間は約33秒で、A40は約1.9GBのGPUメモリを使用し、異常なメモリ増加は見られない。metricsと生成サンプルはPod上の設定出力先に保存されており、学習完了後に回収する。

step 6,000、6,500、7,000のvalidation lossはそれぞれ2.874430、2.874084、2.870859となった。step 7,000のPerplexityは17.652169、train lossは2.843273、学習率は`2.8454e-5`である。step 5,500から7,000までの間も学習プロセスは正常に動作し、validation lossは悪化していない。step 7,000時点では、実験098のbest loss 2.973267に対して約3.4%低い。

その後もstep 7,500、8,000、8,500、9,000、9,500のvalidation lossはそれぞれ2.868662、2.867681、2.864839、2.863642、2.859826となった。step 9,500のPerplexityは17.458489、train lossは2.910493、学習率は`2.6958e-5`である。step 7,000以降もlossは一度も悪化せず、step 9,500時点では実験098のbest lossより約3.8%低い。Runpod A40上のプロセスは正常に継続している。

step 10,000、10,500、11,000、11,500、12,000のvalidation lossはそれぞれ2.863249、2.861999、2.860042、2.856804、2.855655となった。step 12,000のPerplexityは17.385823、train lossは2.655377、学習率は`2.5039e-5`である。validation lossは引き続き改善し、step 12,000時点では実験098のbest lossより約4.0%低い。SFT用データのRunpod転送も完了し、事前学習終了後に同じPodで続けられる状態にした。

step 12,500ではvalidation lossが2.856284へ一時的に上がったが、step 13,000では2.855608へ戻った。step 13,000のPerplexityは17.385009、train lossは2.550091、学習率は`2.4170e-5`である。悪化幅は小さく、現在のbestはstep 13,000である。学習は停止せず、以後も500 stepごとのmetricsと生成文を保存する。

step 13,500と14,000のvalidation lossはそれぞれ2.852940、2.851455となった。step 14,000のPerplexityは17.312960、train lossは2.405879、学習率は`2.3251e-5`である。step 12,500の小さな揺らぎの後も再び改善しており、step 14,000時点では実験098のbest lossより約4.1%低い。

step 14,500ではvalidation lossが2.850629まで下がり、現時点のbestとなった。step 15,000、15,500ではそれぞれ2.852036、2.851940へわずかに上がった。step 15,500のPerplexityは17.321354、train lossは2.917514、学習率は`2.1793e-5`である。validationの揺らぎが見え始めたため、終了後は最終stepではなくbest checkpointをSFTへ渡す。

step 16,000から20,000までのvalidation lossは順に2.851190、2.850689、2.850443、2.849562、2.849151、2.846352、2.845883、2.847966、2.846438である。step 20,500から24,000までは2.845631、2.843498、2.844260、2.844370、2.842431、2.841459、2.843888、2.842477である。step 24,000時点のPerplexityは17.158220、train lossは2.362274、学習率は`1.2745e-5`で、現在のbestはstep 23,000のvalidation loss 2.841459である。途中で小さな揺らぎはあるものの、全体として改善傾向を維持している。

なお、この区間のmetrics.jsonlはRunpodからローカルへ中間回収した。学習中の生成サンプルは学習完了後にまとめて回収し、GitHubで追跡できる形に整理する。

続くstep 24,500、25,000、25,500、26,000、26,500のvalidation lossは2.841545、2.841187、2.841557、2.839787、2.840211となった。step 26,000でvalidation lossのbestを更新したが、step 26,500では0.0004程度の小さな反発があった。step 26,500のPerplexityは17.119381、学習率は`1.0227e-5`であり、Runpod上の学習プロセスは継続中である。

step 27,000、27,500のvalidation lossは2.839855、2.839585となった。step 27,500のPerplexityは17.108666、train lossは2.480381、学習率は`9.2856e-6`である。step 27,500でbest validation lossを2.839585へ更新した。学習率はすでに1e-5を下回っているが、lossはまだ緩やかに改善している。

step 28,000、28,500のvalidation lossは2.839717、2.838744となった。step 28,500のPerplexityは17.094283、train lossは2.389570、学習率は`8.3908e-6`である。step 28,500でbest validation lossを更新した。低学習率の終盤でもvalidation lossが急落せず、緩やかな改善を保っている。

step 29,000、29,500、30,000のvalidation lossは2.839523、2.838294、2.838723となった。step 29,500でbest validation lossを2.838294へ更新し、step 30,000では0.0004程度反発した。step 30,000のPerplexityは17.093922、train lossは2.523805、学習率は`7.1490e-6`である。学習は正常に継続している。

step 30,500、31,000のvalidation lossは2.839558、2.838394となった。step 31,000のPerplexityは17.088293、train lossは2.191333、学習率は`6.3958e-6`である。step 30,500で一時的に悪化したものの、step 31,000でほぼ戻っており、現時点のbestは引き続きstep 29,500である。

step 31,500、32,000のvalidation lossは2.838054、2.837723となった。step 32,000のPerplexityは17.076836、train lossは2.588188、学習率は`5.7082e-6`である。step 32,000でbest validation lossを更新し、実験098のbest loss 2.973267より約4.6%低くなった。

step 32,500、33,000のvalidation lossは2.838222、2.837074となった。step 33,000のPerplexityは17.065758、train lossは2.459014、学習率は`5.0905e-6`である。step 33,000でbest validation lossを更新した。学習率は最終値`3e-6`へ近づいているが、なお改善が続いている。

step 33,500、34,000のvalidation lossは2.836096、2.835807となった。step 34,000のPerplexityは17.044152、train lossは2.597677、学習率は`4.5468e-6`である。step 34,000でbest validation lossを更新し、実験098のbest lossから約4.6%低い状態を保っている。

step 34,500ではvalidation lossが2.836688へ一時的に上がったが、step 35,000では2.835155へ下がり、bestを更新した。step 35,000のPerplexityは17.033047、train lossは2.376689、学習率は`4.0807e-6`である。学習率の終盤でも改善が残っているため、40,000 stepまで継続する。

step 35,500、36,000のvalidation lossは2.836233、2.836324となった。step 36,000のPerplexityは17.052971、train lossは2.485464、学習率は`3.6951e-6`である。終盤の2回はbestを更新していないが、step 35,000からの悪化は0.0012未満であり、停止条件には該当しない。
