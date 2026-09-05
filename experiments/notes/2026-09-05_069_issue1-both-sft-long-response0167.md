# 実験069：SFT部分の6行中1行を長文にする層化sampling対照

## 開始前の計画

実施日は2026年9月5日、担当者はCodexです。GitHubの[Issue #1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)を今後の実験候補として扱い、一般日本語を保ちながらRPCとMRMPの会話応答を改善する研究を続けます。医療専用モデルにはせず、通常の日本語データ、Issue #1に関連する会話データ、医師国家試験由来のデータを同じ日本語モデルの中で役割を分けて利用します。元の`/Users/koseki/projects/medilink_analysis`と医師国家試験データは保全し、変更・削除しません。

実験067はrehearsal ratio 0.20でSFT例を一様にsamplingし、実験068は応答24 Token以上を長文と定義して長文例を増やしました。068では、SFT部分6行のうち長文2行・通常4行となり、long F1が067の0.133749から0.162121へ改善しました。ただし、068のCLI指定は25%でもbatchの丸めにより実効33.3%であり、指定値と実効値が一致しませんでした。

今回は同じbase、データ、seed、rehearsal ratio、学習step、評価条件を固定し、SFT部分6行のうち長文をちょうど1行、通常応答を5行にします。`--long-response-ratio 0.1666666667`は実装の丸めで`round(6 * ratio)=1`となるため、実効長文比率はSFT部分の16.7%です。068の2/6と対照することで、長文例を増やす効果が連続的か、2行以上を入れたときだけ現れるかを調べます。

仮説は、長文例を1/6へ増やすだけでも一様samplingの067よりlong F1と平均生成長が改善し、medical lossとshort F1の悪化は068より小さいというものです。反対に、1行では学習量が足りず067と同程度なら、長文層の効果には一定以上のoversamplingが必要だと解釈します。Token overlapだけでは自然さを確定できないため、5領域loss、EOS、長さ別F1、source別傾向、生成TXTを併せて確認します。

## 再現条件

実験068の評価まで完了した基準commitは`b148916`で、`origin/main`へpush済みです。使用する設定は[`configs/issue1-both-20m-sft-source-rehearsal020-long0167-mps-3k.toml`](../../configs/issue1-both-20m-sft-source-rehearsal020-long0167-mps-3k.toml)です。設定ファイルのSHA-256は`84fc7601372d4e9f9b507d55c7da73024bbe2d4a6b498b1fd42020250945474e`です。モデルはRoPE・LayerNorm・SwiGLU、dim 384、10層、6 heads、context length 256、19,308,032 parameterです。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`です。

実験068と同じbase checkpoint、Tokenizer、会話SFT train・validation、rehearsal Token列を使用します。base checkpointは`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`、会話SFT trainは`artifacts/sft/issue1-both-balanced-v1/train.npz`、validationは`artifacts/sft/issue1-both-full-v1/validation.npz`、rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`です。SFT trainは64,423例・response 770,990 Tokenで、24 Token以上の長文は4,286例です。validationは49,045例・response 738,660 Tokenで、24 Token以上の長文は6,890例です。

学習条件はbatch size 8、3,000 step、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 42、EOS loss weight 0.50、schedule終点3,000 stepです。SFTとrehearsalを0.80対0.20で合算し、SFT部分6行から長文1行・通常5行を抽出します。MPSではAMPを使いません。生成はconversation形式、話者DAとDC、固定promptは`こんにちは！`、最大160 Token、temperature 0.8、top-k 40です。

再現に使うコマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-20m-sft-source-rehearsal020-long0167-mps-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal020-long0167-mps-3k \
  --samples-dir artifacts/samples/issue1-both-20m-sft-source-rehearsal020-long0167-mps-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --long-response-ratio 0.1666666667 \
  --long-response-min-tokens 24 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

入力ファイルのSHA-256は、実験068と同じです。base checkpointは`326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`、Tokenizerは`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`、SFT trainは`645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44`、SFT validationは`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`、rehearsal Token列は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。

## 成功・失敗の判定基準

3,000 stepをNaN、OOM、shape errorなく完走し、step 0〜3,000の生成本文、metrics、summary、checkpoint metadata、5領域評価、固定chat-test 48例を保存できれば実装上の成功とします。067よりlong F1と平均生成長が改善し、068よりshort F1またはmedical lossの悪化が小さければ、長文層化の低い実効比率を次の候補にします。差が小さい場合も、丸めを統制した比較結果として記録します。

Colab CLIでT4割り当てを試します。失敗時はHTTP応答とsession状態をノートへ追記し、同一コマンドをMPSで実行します。raw会話JSONLや医師国家試験の原本はColab bundleへ含めません。

## 実験中の記録

この節には、Colab試行、MPSへの切り替え、500 stepごとのloss・PPL・経過時間・生成本文、警告や途中停止を時系列で追記します。学習中の生成本文は省略せずGitHubへ保存します。

2026年9月5日23:23:46 JSTに`colab new -s exp069-both-long0167 --gpu T4`を実行しましたが、Colab APIのassignment endpointがHTTP 503 `Service Unavailable`を返しました。`colab sessions`でもactive sessionがないことを確認し、bundle uploadやColab上の学習は発生していません。実験068までと同じ制約ですので、同一条件をMPSへ切り替えます。なお、その前のCLI呼び出しには実行前の記述エラーがありましたが、Colab APIへ到達した試行はこの記録の一回です。

## 実験終了後の結果と解釈

学習終了直後に、最終train・validation loss、PPL、最良checkpoint、学習時間、5領域loss、EOS、長さ別F1、source別F1、生成例、成果物hash、実験068および067との差を追記します。失敗した場合も削除せず、原因不明ならそのまま記録します。

## 次に試すこと

結果が068と同じ方向なら、SFT部分3/6の長文条件を追加して効果の飽和点を探します。結果が067に近ければ、長文oversamplingより会話テンプレートや話題継続評価の改善を優先します。段階比較が固まった後、20Mで選んだ条件を50Mへ拡大します。
