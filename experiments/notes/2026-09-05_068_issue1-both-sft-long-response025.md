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

step 1,100ではvalidation loss 3.859794、PPL 47.4556、step 1,200では3.843093、PPL 46.6696、step 1,300では3.840304、PPL 46.5396、step 1,400では3.829444、PPL 46.0369となりました。step 1,500ではtrain loss 3.643088、SFT loss 3.513138、rehearsal loss 4.162886、validation loss 3.820388、PPL 45.6219、learning rate 2.8742e-5、経過時間838.03秒となりました。step 1,100〜1,500の生成本文、step 1,500のcheckpoint metadata、metricsを保存し、1,500 step時点の成果物をコミット・pushします。固定promptへのstep 1,500生成は「こんばんはー!」となり、短い挨拶の出力表現は引き続き変化しています。ここまで異常はありませんが、long stratumへの効果は最終評価まで保留します。

step 1,600ではvalidation loss 3.811465、PPL 45.2166、step 1,700では3.806735、PPL 45.0033、step 1,800では3.804759、PPL 44.9144、step 1,900では3.798270、PPL 44.6239となりました。step 2,000ではtrain loss 2.876710、SFT loss 2.845041、rehearsal loss 3.003387、validation loss 3.785511、PPL 44.0582、learning rate 1.6982e-5、経過時間1,124.38秒となりました。step 1,600〜2,000の生成本文、step 2,000のcheckpoint metadata、metricsを保存し、2,000 step時点の成果物をコミット・pushします。固定promptへのstep 2,000生成は「こんにちは!」となりました。validation lossは一貫して改善していますが、長文応答への効果判定は最終評価で行います。

## 実験終了後の結果と解釈

学習は2026年9月5日にMPS上で完走しました。実際のbackendはPyTorch 2.14.0、deviceは`mps`、AMPは無効、parameter数は19,308,032、学習時間はsummary上で1,683.35秒でした。Colab T4は開始前にHTTP 503で割り当てられなかったため、MPSへ切り替えました。NaN、OOM、shape error、checkpoint reload errorは発生していません。

最終step 3,000のtrain lossは3.838269、SFT lossは3.928383、rehearsal lossは3.477811、validation lossは3.734906、PPLは41.8841でした。最良checkpointもstep 3,000で、`best.pt`のSHA-256は`953544f7be5bc35954ffd731f11cc906a0067a4067673685f40de7e0d28ccff9`です。step 2,800のvalidation loss 3.744378を下回って最終stepで更新されました。学習途中ではstep 2,200にvalidation loss 3.789005へ一時的に悪化しましたが、step 2,800以降に再び改善しましたので、異常停止ではなく通常のbatch揺らぎとして扱います。

5領域のvalidation lossは、general 5.461471（PPL 235.4436）、conversation 2.884240（17.8900）、medical 3.232369（25.3396）、RPC 2.872909（17.6884）、MRMP 2.327911（10.2565）でした。実験067（rehearsal ratio 0.20、一様SFT sampling）との差は、general -0.008409、conversation -0.013348、medical +0.014136、RPC -0.027021、MRMP -0.013503です。長文例を増やしてもgeneralはほぼ維持され、medicalだけわずかに悪化しましたが、会話およびRPC・MRMPのvalidation lossは改善しました。ただし差は単一seedの結果ですので、長文層化が一般化したと断定せず、再現実験の候補とします。

固定chat-test v1の48例では、EOS到達は48/48、平均生成Token数は11.2083、precisionは0.297795、recallは0.230276、Token overlap F1は0.220352でした。short・medium・long別F1はそれぞれ0.317120、0.181814、0.162121です。実験067との差は、平均生成長が+0.7292、precisionが+0.036389、recallが+0.014054、全体F1が+0.012598、short F1が-0.005548、medium F1が+0.014971、long F1が+0.028372となりました。特にlong F1は0.133749から0.162121へ改善し、今回の仮説を支持する方向です。一方、short F1は少し下がったため、すべての応答長に対して無条件に良くなったわけではありません。

source別では、MRMP 24例のF1が0.228068、平均生成長が8.5833、RPC 24例のF1が0.212635、平均生成長が13.8333でした。全例でEOSへ到達しました。long例には、MRMPの「ソメイヨシノの寿命が近い」という話題へ「おー!いいですね!」と返すような短いが話題に触れる出力がある一方、RPCのSpotifyの話題へ「スマホはどうしてるんですが、スポーツを作ったりもします。」と返すような話題逸脱も残りました。したがって、長文層化は長い履歴への表層一致を改善した可能性がありますが、話題継続や意味の自然さを解決したとは扱いません。Token overlapは補助指標であり、生成TXTの目視確認と併用します。

固定promptのstep 3,000生成は、`<|speaker:DA|>こんにちは！`に対し、`<|speaker:DC|>こんにちは!`でEOSへ到達しました。step 500の「こんにちは〜。」、step 1,000の「よろしくお願いします!」、step 1,500の「こんばんはー!」、step 2,000の「こんにちは!」を含め、学習途中の全31個の生成本文を保存しています。悪い生成を含む固定chat 48例の全文も削除せず保存しています。

成果物のSHA-256は、metricsが`937ba0ec9599495fcda0e13065cb47275cb5153de365265ba8ffbcde58e7b1fc`、summaryが`60e21ed7c08ff509e4eeb7fcfe18f7cf9030b87f7856778dfbae9feedb162310`、best metadataが`101d0f4749e8a57e417edf1b6896563f1f9955a033cab583285a0471e36c7f6b`、step 3,000 metadataが`dd2e17e2937ad536bff8ac6c95fb19b79d204c18009dcea82546d4223ac2782c`、step 3,000生成TXTが`fffa73d93c4b9662dc906591a36597c336a723d8f0fd04c0785c28c51bff2f1b`です。5領域評価JSONは`b1889eb8138efd66fc8b1fe8cf4569a5f478f41d89deec3562d457da46afab95`、固定chat評価JSONは`386c57b4d827c80a20e760cb52a8ce4e953dca06e48fa5defeece97f5b910593`、固定chat全文TXTは`7272224de2c531909601ea704265259cd619e9c503dfbdca7a09cb35f1923dac`です。

評価結果と生成本文は、[checkpoint metadata](../../artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k/)、[学習中の生成サンプル](../../artifacts/samples/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k/)、[5領域評価JSON](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-domains.json)、[固定chat評価JSON](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-chat-test-v1.json)、[固定chat全文TXT](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long025-mps-3k-chat-test-v1.txt)から確認できます。重い`.pt`本体はGit管理外ですが、checkpoint metadataとSHA-256をGitHubへ保存しています。

## 結論

実験068は実装上は成功し、長文応答の意図的な再サンプリングによってlong F1、medium F1、全体F1、conversation・RPC・MRMP lossが実験067より改善しました。したがって、「長い応答が少ないために長文会話の学習機会が不足している」という仮説は、今回の単一seed・48例の評価では支持する方向です。ただし、指定した長文比率25%はSFT部分のbatchサイズ6に丸められて2/6、実効33.3%になっており、厳密な25%比較ではありません。またmedical lossとshort F1は悪化し、長文出力にも話題逸脱があるため、現時点で標準条件を置き換えるのではなく、有力な候補として扱います。

## 次に試すこと

次は今回の実効33.3%を基準に、SFT部分の長文行数を固定してlong層を1/6、2/6、3/6へ分け、指定比率の丸めによる影響を除いた比較を行います。その際、medical lossの悪化とshort F1の低下がどの程度再現するかを確認します。並行して、長文応答の話題適合を人手レビューできる評価表を追加し、Token overlapだけでは捉えられない改善を記録します。長文層化の効果が再現すれば、20Mで選んだ条件を50Mへ拡大し、Issue #1の会話データを含む一般日本語モデルで容量差を検証します。
