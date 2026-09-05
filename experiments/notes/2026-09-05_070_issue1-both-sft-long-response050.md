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

step 1,100ではvalidation loss 3.865315、PPL 47.7183、step 1,200では3.858782、PPL 47.4076、step 1,300では3.850078、PPL 46.9967、step 1,400では3.850653、PPL 47.0238となりました。step 1,500ではtrain loss 3.838816、SFT loss 3.755827、rehearsal loss 4.170773、validation loss 3.847234、PPL 46.8632、learning rate 2.8742e-5、経過時間769.97秒となりました。step 1,100〜1,500の生成本文、step 1,500のcheckpoint metadata、metricsを保存し、1,500 step時点の成果物をコミット・pushします。step 1,300以降のvalidation lossはおおむね横ばいで、長文3/6によるbatchごとの揺らぎが続いています。固定promptへのstep 1,500生成は「こんにちは!」となりました。ここまで異常はありません。

step 1,600ではvalidation loss 3.829579、PPL 46.0432、step 1,700では3.816498、PPL 45.4448、step 1,800では3.819072、PPL 45.5619、step 1,900では3.826329、PPL 45.8938となりました。step 2,000ではtrain loss 3.192136、SFT loss 3.244103、rehearsal loss 2.984266、validation loss 3.826663、PPL 45.9091、learning rate 1.6982e-5、経過時間1,041.78秒となりました。step 1,600〜2,000の生成本文、step 2,000のcheckpoint metadata、metricsを保存し、2,000 step時点の成果物をコミット・pushします。step 1,700でvalidation loss 3.816498まで下がった後、step 1,800〜2,000で悪化しました。068・069の同stepより悪く、長文3/6でvalidation保持が弱くなる可能性を記録しますが、最終評価までは結論を保留します。

step 2,100ではvalidation loss 3.795547、PPL 44.5026、step 2,200では3.821363、PPL 45.6664、step 2,300では3.792368、PPL 44.3613、step 2,400では3.801078、PPL 44.7494となりました。step 2,500ではtrain loss 3.736397、SFT loss 3.699453、rehearsal loss 3.884171、validation loss 3.788308、PPL 44.1816、learning rate 8.2333e-6、経過時間1,317.38秒でした。step 2,600ではvalidation loss 3.787263、PPL 44.1354、step 2,700では3.789566、PPL 44.2372、step 2,800では3.781380、PPL 43.8765、step 2,900では3.792364、PPL 44.3612となりました。最終step 3,000ではtrain loss 3.679527、SFT loss 3.722701、rehearsal loss 3.506831、validation loss 3.778070、PPL 43.7316、learning rate 5.0000e-6、経過時間1,590.69秒となりました。step 2,100〜3,000の生成本文、step 2,500・3,000のcheckpoint metadata、metrics、summaryを保存し、学習はNaN、OOM、shape errorなく完走しました。

## 実験終了後の結果と解釈

学習はMPS上で完走しました。実際のbackendはPyTorch 2.14.0、deviceは`mps`、AMPは無効、parameter数は19,308,032、summary上の学習時間は1,591.17秒でした。Colab T4は開始前のassignment endpointでHTTP 503となったため、MPSへ切り替えました。NaN、OOM、shape error、checkpoint reload errorは発生していません。最良checkpointはstep 3,000で、`best.pt`のSHA-256は`39cf937ff9ef53c779d96fdbf4f44b1c487959ba76c9d7a95325a7f6107d9632`です。

5領域のvalidation lossは、general 5.466829（PPL 236.7083）、conversation 2.897165（18.1227）、medical 3.236975（25.4566）、RPC 2.887402（17.9466）、MRMP 2.357969（10.5695）でした。実験067との差はgeneral -0.003052、conversation -0.000423、medical +0.018742、RPC -0.012528、MRMP +0.016556です。実験068との差はgeneral +0.005357、conversation +0.012925、medical +0.004606、RPC +0.014493、MRMP +0.030059です。3/6条件は一般lossを067よりわずかに改善しましたが、medicalとMRMPは悪化し、068より5領域すべてで悪化しました。

固定chat-test v1の48例では、EOS到達は48/48、平均生成Token数は16.7917、precisionは0.229566、recallは0.243152、Token overlap F1は0.199866でした。short・medium・long別F1はそれぞれ0.274311、0.166940、0.158348です。実験067との差は、平均生成長+6.3125、precision-0.031840、recall+0.026930、全体F1-0.007887、short F1-0.048358、medium F1+0.000097、long F1+0.024599です。実験068との差は、平均生成長+5.5833、precision-0.068229、recall+0.012876、全体F1-0.020486、short F1-0.042809、medium F1-0.014874、long F1-0.003773です。

long F1は0/6の067の0.133749から1/6の069で0.153085、2/6の068で0.162121、3/6の070で0.158348となりました。1/6と2/6では改善したものの、3/6では2/6からわずかに下がり、全体F1は0.199866まで下がりました。平均生成長は16.7917へ大きく伸びましたが、precisionとshort F1が大きく落ちています。したがって、長文例の増加は一定範囲でlong層を改善する一方、3/6まで増やすと短い応答の適合や生成の精度を損なう可能性があると解釈します。

source別では、MRMP 24例のF1が0.209273、平均生成長が13.0833、RPC 24例のF1が0.190459、平均生成長が20.5000で、両sourceともEOS到達は24/24でした。長文には「カルディ」の話題へトマトやキャラクターを混ぜる出力、「Spotify」の話題へスパイスやスポーツを混ぜる出力があり、長く生成できても話題適合は改善していません。これは、長文oversamplingを続ける前に話題継続の評価とデータ形式を見直すべきことを示します。

固定promptのstep 3,000生成は、`<|speaker:DA|>こんにちは！`に対して`<|speaker:DC|>こんにちは!`でEOSへ到達しました。step 500の「よろしくお願いしますー!」、step 1,000の「よろしくお願いします!」、step 1,500の「こんにちは!」、step 2,000の「こんにちは!」を含むstep 0〜3,000の全31個の生成本文を保存しています。悪い生成を含む48例の評価全文も削除せず保存しています。

成果物のSHA-256は、metricsが`2621a4e7bbdd322a1c108229f51d0b0f708fbfb022a34088966a1e721556daae`、summaryが`20ae83ac88c360dd119d22e0e04cb2b8276b295f94d17da70dcb82314bdd4311`、best metadataが`ed5f0c2acaa4fc51b3a6442d78f38eae1747ab7b4770bc16e42f1fc120c97d0d`、step 3,000 metadataが`e636803d6173c3c799501b0559ac4d6a522d0ebfde021c0b5e9ccb7fac4fdc1a`、step 3,000生成TXTが`fffa73d93c4b9662dc906591a36597c336a723d8f0fd04c0785c28c51bff2f1b`です。5領域評価JSONは`e5372e7d8abc6069336265cf03966fc63681520a1ad74d272efdaabea302c06d`、固定chat評価JSONは`d376b2d2f22abd6f09e5d72c9b507c6a287e690aff5078c1f4182add98617a3d`、固定chat全文TXTは`4b5b55697e31569e7b571dbe586b5c8509a4a80e7924acb7c5dbf39a64183af7`です。

評価結果と生成本文は、[checkpoint metadata](../../artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal020-long050-mps-3k/)、[学習中の生成サンプル](../../artifacts/samples/issue1-both-20m-sft-source-rehearsal020-long050-mps-3k/)、[5領域評価JSON](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long050-mps-3k-domains.json)、[固定chat評価JSON](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long050-mps-3k-chat-test-v1.json)、[固定chat全文TXT](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long050-mps-3k-chat-test-v1.txt)から確認できます。重い`.pt`本体はGit管理外ですが、checkpoint metadataへSHA-256を記録しています。

## 結論

実験070は実装上成功しましたが、SFT部分3/6の長文oversamplingは採用しない判断です。long F1は067より高いものの068より低く、全体F1、short F1、medical loss、MRMP lossが悪化し、生成長だけが大きく伸びました。現時点ではSFT部分2/6の実効33.3%（実験068）が会話適合と基盤保持のバランス上限候補です。ただし、068も単一seed・48例のToken overlapに基づく候補にすぎず、複数seedと人手レビューで再確認する必要があります。

## 次に試すこと

0/6〜3/6の比較結果から、2/6条件を暫定候補として複数seedで再確認します。その前に、話題適合、応答の自然さ、話者marker維持、反復、過生成を人手レビューできる評価表を追加し、Token overlapと意味的品質を分離します。長文例を増やす以外にも、SFTとrehearsalのToken予算を独立に制御する方法を検討します。評価設計が固まった後、20Mで選んだ条件を50Mへ拡大し、Issue #1の会話データを含む一般日本語モデルで容量差を確認します。
