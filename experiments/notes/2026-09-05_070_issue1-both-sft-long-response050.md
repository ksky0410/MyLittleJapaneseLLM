# 実験070：SFT部分の6行中3行を長文にする層化sampling

## 開始前の計画

実施日は2026年9月5日、担当者はCodexです。GitHubの[Issue #1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)を候補に、一般日本語の保持とRPC・MRMPの会話応答を同じモデルで両立する実験を続けます。医療専用化は行わず、通常の日本語データ、Issue #1に関連する会話データ、医師国家試験由来のデータを目的別に組み合わせます。元の`/Users/koseki/projects/medilink_analysis`と医師国家試験データは保全し、変更・削除しません。

実験067はSFT例を一様samplingし、実験068はSFT部分6行中2行を長文へ層化しました。実験069では6行中1行を長文に固定しました。long F1は067の0.133749、069の0.153085、068の0.162121となり、長文例を増やすほど改善する傾向が見えました。ただしmedical lossとshort F1には悪化があり、長文の増加がどこで飽和または逆効果になるかは未確認です。

今回はrehearsal ratio 0.20、SFT部分6行、応答24 Token以上という定義を固定し、長文3行・通常3行を抽出します。`--long-response-ratio 0.50`により、実装の丸めで`round(6 * 0.50)=3`となり、指定値と実効比率が一致します。067の0/6、069の1/6、068の2/6、今回の3/6を比較し、long oversamplingの効果と副作用を段階的に確認します。

仮説は、3/6条件ではlong F1と平均生成長がさらに伸びる一方、short F1とmedical lossの悪化が明確になるというものです。もしlong F1が068から伸びず、short・medicalだけが悪化すれば、2/6を会話性能と基盤保持の暫定上限とします。評価は全体F1だけで決めず、5領域loss、EOS、長さ別F1、source別集計、生成本文を比較します。Token overlapは意味的な自然さの代替ではありませんので、話題逸脱や反復も目視確認します。

## 再現条件

実験069の評価まで完了した基準commitは`12c2090`で、`origin/main`へpush済みです。使用する設定は[`configs/issue1-both-20m-sft-source-rehearsal020-long050-mps-3k.toml`](../../configs/issue1-both-20m-sft-source-rehearsal020-long050-mps-3k.toml)です。設定ファイルのSHA-256は`7b728499282637e37364f06441c01bd4e115745e8d3ccaa5c3a85a1f7037000d`です。モデルはRoPE・LayerNorm・SwiGLU、dim 384、10層、6 heads、context length 256、19,308,032 parameterです。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`です。

実験069と同じbase checkpoint、Tokenizer、会話SFT train・validation、rehearsal Token列、学習率、EOS loss weight、seed、3,000 stepを使います。base checkpointは`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`、会話SFT trainは`artifacts/sft/issue1-both-balanced-v1/train.npz`、validationは`artifacts/sft/issue1-both-full-v1/validation.npz`、rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`です。SFT trainは64,423例・response 770,990 Tokenで、24 Token以上の長文は4,286例です。validationは49,045例・response 738,660 Tokenで、24 Token以上の長文は6,890例です。

学習条件はbatch size 8、3,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 42、EOS loss weight 0.50、schedule終点3,000 stepです。SFTとrehearsalを0.80対0.20で合算し、SFT部分6行から長文3行・通常3行を抽出します。MPSではAMPを使いません。生成はconversation形式、話者DAとDC、固定promptは`こんにちは！`、最大160 Token、temperature 0.8、top-k 40です。

再現に使うコマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-20m-sft-source-rehearsal020-long050-mps-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal020-long050-mps-3k \
  --samples-dir artifacts/samples/issue1-both-20m-sft-source-rehearsal020-long050-mps-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --long-response-ratio 0.50 \
  --long-response-min-tokens 24 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

入力ファイルのSHA-256は、実験069と同じです。base checkpointは`326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`、Tokenizerは`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`、SFT trainは`645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、SFT validationは`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`、rehearsal Token列は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。

## 成功・失敗の判定基準

3,000 stepをNaN、OOM、shape errorなく完走し、step 0〜3,000の生成本文、metrics、summary、checkpoint metadata、5領域評価、固定chat-test 48例を保存できれば実装上の成功とします。068よりlong F1が上がり、short F1とmedical lossの悪化が許容範囲なら3/6を候補とします。long F1が飽和し副作用だけが増える場合は、068の2/6を暫定上限とします。いずれの場合も、失敗や悪い生成を削除せず記録します。

Colab CLIでT4割り当てを試します。失敗時はHTTP応答とsession状態を追記し、同一条件をMPSで実行します。raw会話JSONLや医師国家試験の原本はColab bundleへ含めません。

## 実験中の記録

この節にはColab試行、MPS切り替え、500 stepごとのloss・PPL・経過時間・生成本文、警告や途中停止を時系列で追記します。学習中の出力は省略せずGitHubへ保存します。

2026年9月5日23:55:30 JSTに`colab new -s exp070-both-long050 --gpu T4`を実行しましたが、Colab APIのassignment endpointがHTTP 503 `Service Unavailable`を返しました。直後の`colab sessions`でもactive sessionがないことを確認し、bundle uploadやColab上の学習は発生していません。実験062〜069と同様に、同一条件をMPSで実行します。

23:56台に、予定したMPSコマンドで学習を開始しました。step 1はtrain loss 5.055552、SFT loss 5.094285、rehearsal loss 4.900624、validation loss 4.723784、PPL 112.5935、経過時間2.26秒でした。step 100はvalidation loss 4.105484、PPL 60.6721、step 200は4.048785、PPL 57.3277、step 300は4.010555、PPL 55.1775、step 400は4.000342、PPL 54.6168でした。step 500ではtrain loss 3.673428、SFT loss 3.714670、rehearsal loss 3.508461、validation loss 3.978557、PPL 53.4399、learning rate 4.7931e-5、経過時間227.39秒となりました。step 0〜500の生成本文とstep 500までのmetrics・checkpoint metadataを保存し、500 step時点の成果物をコミット・pushします。固定promptへのstep 500生成は「よろしくお願いしますー!」となり、EOSへ到達しました。ここまで異常はありません。

step 600ではvalidation loss 3.933197、PPL 51.0700、step 700では3.925920、PPL 50.6997、step 800では3.954736、PPL 52.1819、step 900では3.889577、PPL 48.8902となりました。step 1,000ではtrain loss 3.772419、SFT loss 3.502555、rehearsal loss 4.851875、validation loss 3.899693、PPL 49.3873、learning rate 4.0147e-5、経過時間498.44秒となりました。step 600〜1,000の生成本文、step 1,000のcheckpoint metadata、metricsを保存し、1,000 step時点の成果物をコミット・pushします。step 800で一時的なvalidation loss悪化があり、step 900で改善しました。固定promptへのstep 1,000生成は「よろしくお願いします!」となりました。ここまで異常はありません。

## 実験終了後の結果と解釈

学習終了直後に、最終train・validation loss、PPL、最良checkpoint、学習時間、5領域loss、EOS、長さ別F1、source別F1、生成例、成果物hash、実験067〜069との差を追記します。

## 次に試すこと

0/6〜3/6の結果から長文層化の暫定比率を決め、複数seedまたは長文評価用の人手レビューで再確認します。その後、20Mで選んだ条件を50Mへ拡大し、Issue #1の会話データを含む一般日本語モデルで容量差を調べます。
