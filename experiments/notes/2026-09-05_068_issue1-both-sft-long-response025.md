# 実験068：長い会話応答を25%へ層化したboth-SFT

## 開始前の計画

実施日は2026年9月5日、担当者はCodexです。GitHubの[Issue #1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)を今後の実験候補として扱い、RealPersonaChat（RPC）とMulti-Relational Multi-Party Chat Corpus（MRMP）を混ぜた会話SFTを継続します。Issue #1が示している「一般日本語を保ちながら会話品質を高める」という方向を維持し、医療専用モデルにはせず、通常の日本語学習データ、Issue由来の会話データ、医師国家試験由来のデータを同じ研究計画の中で比較していきます。

実験064〜067では、RPCとMRMPを混ぜたSFTにrehearsalを加える条件を比較しました。実験067のrehearsal ratio 0.20は、5領域のvalidation lossを実験066より改善し、chat-test F1もほぼ同じでした。しかし、48例のchat-testではlong stratumのToken overlap F1が0.133749に留まり、料理や動画などの話題を長く維持できない出力が残りました。

そこで今回は、rehearsal ratio 0.20を固定したまま、SFT batch内の長い応答例を増やします。SFT trainの応答長を調べると、64,423例のうち24 loss対象Token以上は4,286例（6.65%）だけで、validationでも49,045例中6,890例（14.05%）です。通常の一様サンプリングでは、長い応答の学習機会が少なすぎる可能性があります。

今回確かめたい仮説は、「長い応答を意図的に増やせば、long stratumの会話適合性と話題継続が改善する」です。副作用として、短い定型応答への適合や一般文書保持が少し悪化する可能性もあります。したがって、全体F1だけでなくshort・medium・long別F1、5領域loss、平均生成長、EOS到達率、生成本文を実験067と比較します。

長い応答は`loss_mask.sum() >= 24`で定義し、SFT batchの少なくとも25%をその層から抽出します。rehearsal ratio 0.20では全batch 8行のうちSFT部分が6行になるため、実装の丸めにより1 batchあたり長い応答2行、通常応答4行となります。これはSFT部分に対する実効比率33.3%であり、指定値25%とのずれを結果解釈に明記します。短い応答の層化とは同時に使いません。

## 再現条件

長文層化sampling機能とテストを追加したコードの基準commitは`d7e109a`です。このcommitは`origin/main`へpush済みです。学習前の作業treeでは、065関連の既存未管理差分（`scripts/colab_bootstrap_065.py`、`scripts/colab_package_065.py`、`scripts/colab_concat_065.py`）を変更・追加・削除せず、そのまま残します。

使用する設定は[`configs/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k.toml`](../../configs/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k.toml)です。設定ファイルのSHA-256は`826042efb68ec208f45311992e9110e51da30dd89de15089ed4aa5378d1bf4cd`です。モデルはRoPE・LayerNorm・SwiGLU、dim 384、10層、6 heads、context length 256、19,308,032 parameterです。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`です。

実験067と同じbase checkpoint、Tokenizer、会話SFT train・validation、rehearsal Token列、学習率、EOS loss weight、seed、3,000 stepを使います。base checkpointは`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`、会話SFT trainは`artifacts/sft/issue1-both-balanced-v1/train.npz`、validationは`artifacts/sft/issue1-both-full-v1/validation.npz`、rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`です。

学習条件はbatch size 8、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 42、EOS loss weight 0.50、learning-rate schedule終点3,000 stepです。SFTとrehearsalを0.80対0.20で合算し、SFTの6行のうち長文2行・通常4行を抽出します。MPSではAMPを使いません。生成はconversation形式、話者はDAとDC、固定promptは`こんにちは！`、最大160 Token、temperature 0.8、top-k 40です。

再現に使うコマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k \
  --samples-dir artifacts/samples/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k \
  --lr-schedule-steps 3000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --long-response-ratio 0.25 \
  --long-response-min-tokens 24 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

入力ファイルのSHA-256は次のとおりです。大きな原文データをこの実験の成果物へ複製せず、加工済みデータとハッシュだけを再現条件に残します。元の`/Users/koseki/projects/medilink_analysis`と医師国家試験データは読み取り対象として保全し、変更・削除しません。

| 入力 | SHA-256 |
| --- | --- |
| base checkpoint | `326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751` |
| Tokenizer | `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4` |
| SFT train | `645febae8fc8d471a78822027c3b693da346cf605c5cdf435a84902dffb73a44` |
| SFT validation | `fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1` |
| rehearsal Token列 | `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090` |

## 成功・失敗の判定基準

3,000 stepをNaN、OOM、shape errorなく完走し、best checkpoint metadata、metrics、summary、step 0〜3,000の生成本文、5領域評価、固定chat-test 48例を保存できれば学習実験として成功とします。性能面では、実験067に対してlong F1が改善し、平均生成長または話題適合の悪化が許容範囲に収まることを期待します。long F1が改善しなくても、validation lossや他の層とのトレードオフを記録し、仮説の反証として扱います。

Colab CLIが利用可能ですので、学習前にT4 GPUの割り当てを一度試します。割り当てが失敗した場合は、HTTPエラーとsession状態を記録し、同じコード・データ・seed・出力仕様でMPSへ切り替えます。Colabへ送る場合もraw会話JSONLや医師国家試験の原本はbundleへ含めません。

## 実験中の記録

この節には、学習開始前のColab試行、500 stepごとの状況、異常、生成サンプル保存状況を時系列で追記します。1,000 stepを超えて記録間隔を空けません。学習終了後に予定と実際の差分、停止理由、最良checkpoint、評価結果を追記します。

2026年9月5日22:49:57 JSTに、Colab CLIで`colab new -s exp068-both-long025 --gpu T4`を実行しました。しかしColab APIのassignment endpointがHTTP 503 `Service Unavailable`を返し、sessionは作成されませんでした。直後の`colab sessions`でもactive sessionがないことを確認しました。したがってColab上のbundle uploadや学習は発生しておらず、予定どおり同じ条件をMPSで実行します。今回もColab割り当て失敗は成果の失敗ではなく、計算資源の切り替えとして扱います。

2026年9月5日22:50台に、予定したMPSコマンドで学習を開始しました。step 1はtrain loss 5.056197、SFT loss 5.095090、rehearsal loss 4.900624、validation loss 4.723757、PPL 112.5905、経過時間2.33秒でした。step 100はvalidation loss 4.107846、PPL 60.8156、step 200は4.042297、PPL 56.9570、step 300は4.009820、PPL 55.1370、step 400は3.998523、PPL 54.5176でした。step 500ではtrain loss 3.321728、SFT loss 3.270458、rehearsal loss 3.526806、validation loss 3.974348、PPL 53.2154、learning rate 4.7931e-5、経過時間294.72秒となりました。step 0〜500の生成本文とstep 500までのmetrics・checkpoint metadataを保存済みです。固定promptへのstep 500生成は「こんにちは〜。」となり、少なくともEOS直後の会話形式を壊さず応答しています。学習は継続中で、ここまで異常はありません。

step 600ではvalidation loss 3.944994、PPL 51.6760、step 700では3.926233、PPL 50.7156、step 800では3.922354、PPL 50.5192、step 900では3.900266、PPL 49.4156となりました。step 1,000ではtrain loss 3.740906、SFT loss 3.463129、rehearsal loss 4.852015、validation loss 3.891312、PPL 48.9751、learning rate 4.0147e-5、経過時間561.71秒となりました。step 600〜1,000の生成本文、step 1,000のcheckpoint metadata、metricsを保存し、1,000 step時点の成果物をコミット・pushします。step 500の生成「こんにちは〜。」に対し、step 1,000は「よろしくお願いします!」となり、短い挨拶応答の表現が変化しました。長文層化の効果はまだ評価できないため、学習を継続します。

## 実験終了後の結果と解釈

学習終了直後に、実行条件、最終train・validation loss、PPL、最良checkpoint、学習時間、評価JSONと全文生成へのリンク、実験067との比較、次に変える条件を追記します。悪い生成や失敗も削除せず、GitHubから追跡できる形で保存します。

## 次に試すこと

今回の結果で長文層化が有効なら、次は長文層の割合を15%または50%へ変え、SFTとrehearsalのToken予算を分離して比較します。有効でなければ、Issue #1の会話データを追加することよりも、会話テンプレート、応答長、話題継続評価の設計を見直します。その後、20Mで得た条件を50Mへ拡大し、モデル容量を増やしたときにも効果が再現するかを確認します。
