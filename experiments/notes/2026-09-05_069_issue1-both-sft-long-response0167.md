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

23:24台に、予定したMPSコマンドで学習を開始しました。step 1はtrain loss 4.872262、SFT loss 4.865171、rehearsal loss 4.900624、validation loss 4.723680、PPL 112.5818、経過時間2.26秒でした。step 100はvalidation loss 4.116166、PPL 61.3237、step 200は4.038470、PPL 56.7394、step 300は4.013930、PPL 55.3640、step 400は3.999679、PPL 54.5806でした。step 500ではtrain loss 3.342968、SFT loss 3.296082、rehearsal loss 3.530514、validation loss 3.975393、PPL 53.2711、learning rate 4.7931e-5、経過時間235.36秒となりました。step 0〜500の生成本文とstep 500までのmetrics・checkpoint metadataを保存し、500 step時点の成果物をコミット・pushします。固定promptへのstep 500生成は「そうですね!あなたは。」となり、応答はEOSへ到達しましたが、質問の形が不自然です。ここまで異常はありません。

step 600ではvalidation loss 3.945801、PPL 51.7177、step 700では3.927178、PPL 50.7635、step 800では3.923494、PPL 50.5769、step 900では3.888334、PPL 48.8294となりました。step 1,000ではtrain loss 3.927993、SFT loss 3.691380、rehearsal loss 4.874445、validation loss 3.880866、PPL 48.4662、learning rate 4.0147e-5、経過時間508.48秒となりました。step 600〜1,000の生成本文、step 1,000のcheckpoint metadata、metricsを保存し、1,000 step時点の成果物をコミット・pushします。固定promptへのstep 1,000生成は「こんにちは!」となり、step 500の不自然な続きから変化しました。ここまで異常はありません。

step 600ではvalidation loss 3.945801、PPL 51.7177、step 700では3.927178、PPL 50.7635、step 800では3.923494、PPL 50.5769、step 900では3.888334、PPL 48.8294となりました。step 1,000ではtrain loss 3.927993、SFT loss 3.691380、rehearsal loss 4.874445、validation loss 3.880866、PPL 48.4662、learning rate 4.0147e-5、経過時間508.48秒となりました。step 600〜1,000の生成本文、step 1,000のcheckpoint metadata、metricsを保存し、1,000 step時点の成果物をコミット・pushします。固定promptへのstep 1,000生成は「こんにちは!」となり、step 500の不自然な続きから変化しました。ここまで異常はありません。

step 1,100ではvalidation loss 3.869417、PPL 47.9144、step 1,200では3.846312、PPL 46.8201、step 1,300では3.844671、PPL 46.7433、step 1,400では3.827139、PPL 45.9309となりました。step 1,500ではtrain loss 3.929037、SFT loss 3.874792、rehearsal loss 4.146019、validation loss 3.816231、PPL 45.4327、learning rate 2.8742e-5、経過時間783.48秒となりました。step 1,100〜1,500の生成本文、step 1,500のcheckpoint metadata、metricsを保存し、1,500 step時点の成果物をコミット・pushします。固定promptへのstep 1,500生成は「よろしくお願いします。」となりました。ここまで異常はありません。

step 1,600ではvalidation loss 3.796752、PPL 44.5562、step 1,700では3.796638、PPL 44.5512、step 1,800では3.790928、PPL 44.2975、step 1,900では3.782652、PPL 43.9324となりました。step 2,000ではtrain loss 2.982955、SFT loss 2.972044、rehearsal loss 3.026596、validation loss 3.775949、PPL 43.6389、learning rate 1.6982e-5、経過時間1,061.16秒となりました。step 1,600〜2,000の生成本文、step 2,000のcheckpoint metadata、metricsを保存し、2,000 step時点の成果物をコミット・pushします。固定promptへのstep 2,000生成は「こんにちは!」となりました。068の同step validation loss 3.785511より低いものの、長文層の効果は最終評価で判断します。

step 2,100ではvalidation loss 3.767005、PPL 43.2503、step 2,200では3.764137、PPL 43.1265、step 2,300では3.747430、PPL 42.4119、step 2,400では3.747834、PPL 42.4291となりました。step 2,500ではtrain loss 3.669346、SFT loss 3.616761、rehearsal loss 3.879684、validation loss 3.741541、PPL 42.1629、learning rate 8.2333e-6、経過時間1,332.80秒でした。step 2,600ではvalidation loss 3.742669、PPL 42.2105、step 2,700では3.738292、PPL 42.0261、step 2,800では3.731720、PPL 41.7508、step 2,900では3.731568、PPL 41.7445となりました。最終step 3,000ではtrain loss 3.958955、SFT loss 4.074616、rehearsal loss 3.496309、validation loss 3.723846、PPL 41.4234、learning rate 5.0000e-6、経過時間1,605.73秒となりました。step 2,100〜3,000の生成本文、step 2,500・3,000のcheckpoint metadata、metrics、summaryを保存し、学習はNaN、OOM、shape errorなく完走しました。

## 実験終了後の結果と解釈

学習はMPS上で完走しました。実際のbackendはPyTorch 2.14.0、deviceは`mps`、AMPは無効、parameter数は19,308,032、summary上の学習時間は1,606.35秒でした。Colab T4は開始前のassignment endpointでHTTP 503となったため、予定どおりMPSへ切り替えました。NaN、OOM、shape error、checkpoint reload errorは発生していません。最良checkpointはstep 3,000で、`best.pt`のSHA-256は`e6fc24d55dad637e75eb5e4f691a7fcf3c9ebb29a6311209a5ddc2f0389c6d96`です。

5領域のvalidation lossは、general 5.456032（PPL 234.1664）、conversation 2.878196（17.7822）、medical 3.233512（25.3686）、RPC 2.868983（17.6191）、MRMP 2.311286（10.0874）でした。実験067との差はgeneral -0.013848、conversation -0.019392、medical +0.015280、RPC -0.030946、MRMP -0.030127です。068との差はgeneral -0.005439、conversation -0.006044、medical +0.001144、RPC -0.003926、MRMP -0.016625です。1/6条件でも一般・会話・RPC・MRMPのlossは一様samplingより下がりましたが、medical lossは067・068より高くなりました。validationだけではlong samplingの最適値は決められません。

固定chat-test v1の48例では、EOS到達は48/48、平均生成Token数は11.9792、precisionは0.278684、recallは0.220230、Token overlap F1は0.215316でした。short・medium・long別F1はそれぞれ0.324356、0.168507、0.153085です。実験067との差は、平均生成長+1.5000、precision+0.017277、recall+0.004008、全体F1+0.007563、short F1+0.001687、medium F1+0.001664、long F1+0.019336です。実験068との差は、平均生成長+0.7708、precision-0.019112、recall-0.010046、全体F1-0.005036、short F1+0.007235、medium F1-0.013307、long F1-0.009036です。

この結果から、long F1は0行の一様sampling（0.133749）から1行（0.153085）、2行（0.162121）へ増えるにつれて改善しました。全体F1も0.207753、0.215316、0.220352と同じ順序で上がりましたので、今回の3条件では長文応答のoversamplingが会話の表層一致と平均生成長を改善する傾向が確認できました。ただし、1行から2行への増加でshort F1は0.324356から0.317120へ下がり、medical lossも1/6の3.233512、2/6の3.232369と一様条件の3.218233より悪いため、長文を増やすほど全領域で有利になるわけではありません。なお、3条件は同じseedでもbatchごとの乱数系列が条件変更によって変わるため、差は厳密なpaired ablationではなく、単一seedの比較候補として扱います。

source別の集計では、MRMP 24例のF1が0.225914、平均生成長が8.3333、RPC 24例のF1が0.204719、平均生成長が15.6250で、両sourceともEOS到達は24/24でした。068のMRMP F1 0.228068、RPC F1 0.212635よりは低いものの、個々の長文生成には、文脈に応じた返答に近いものと、話題を外した不自然なものが混在しています。たとえば「Spotify」の話題に対してスポーツへ逸れる出力や、長い履歴に対して短い相づちだけを返す出力が残っています。そのため、long F1の上昇を話題継続能力の証明とは扱いません。

固定promptのstep 3,000生成は、`<|speaker:DA|>こんにちは！`に対して`<|speaker:DC|>こんにちは!`でEOSへ到達しました。step 500の「そうですね!あなたは。」、step 1,000の「こんにちは!」、step 1,500の「よろしくお願いします。」、step 2,000の「こんにちは!」を含むstep 0〜3,000の全31個の生成本文を保存しています。悪い生成を含む48例の評価全文も削除せず保存しています。

成果物のSHA-256は、metricsが`902ac3081beb9c1d9cb5a555c90f845e09fcdf7311c7dc335c9dcaf987430ce6`、summaryが`0602f56402bd527243784fd9e8fe067c17444f4e9c45fba2d1c793f0b1ac9b43`、best metadataが`f46ac938addd03ba5e2f2d1d8fc79e2da6ad0ba65a895db7577c096557aa5905`、step 3,000 metadataが`7b181da9d068384496637d4b6f1dbb2db19413693f968fbc5c2ae24412a59584`、step 3,000生成TXTが`fffa73d93c4b9662dc906591a36597c336a723d8f0fd04c0785c28c51bff2f1b`です。5領域評価JSONは`33efb06fdf64e90a677740191deb37f966b02e84aee5c4640f71af211996c801`、固定chat評価JSONは`fac7e0b2f1576e614864182c7d23e9464949a9e9028c96c64ce514c45a4a8a72`、固定chat全文TXTは`715379f2b7a2b1fd93a3b90eb64defea3d68909d86e3d601a00595482800bd3b`です。

評価結果と生成本文は、[checkpoint metadata](../../artifacts/checkpoints/issue1-both-20m-sft-source-rehearsal020-long0167-mps-3k/)、[学習中の生成サンプル](../../artifacts/samples/issue1-both-20m-sft-source-rehearsal020-long0167-mps-3k/)、[5領域評価JSON](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long0167-mps-3k-domains.json)、[固定chat評価JSON](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long0167-mps-3k-chat-test-v1.json)、[固定chat全文TXT](../../artifacts/evaluations/issue1-both-20m-sft-source-rehearsal020-long0167-mps-3k-chat-test-v1.txt)から確認できます。重い`.pt`本体はGit管理外ですが、checkpoint metadataへSHA-256を記録しています。

## 結論

実験069は実装上成功しました。SFT部分の長文例を1/6へ増やすと、一様samplingの067よりlong F1と全体F1が改善し、068の2/6条件との間に入る結果になりました。これは、長文応答を増やす効果が今回の範囲では段階的に現れるという仮説を支持します。一方、068より全体F1とlong F1が低く、medical lossも一様条件より悪いため、現時点の標準条件を1/6へ決めるのではなく、1/6を低コストな候補、2/6を高い会話適合候補として記録します。自然さ・話題継続については、Token overlapのみでは判断できません。

## 次に試すこと

次はSFT部分3/6の長文条件を追加し、0/6・1/6・2/6・3/6の段階比較を完成させます。その後、長文のoversampling比率だけでなく、SFTとrehearsalのToken予算を独立に制御する実験へ進みます。長文層化の有効性が再現すれば、20Mで選んだ条件を50Mへ拡大し、Issue #1の会話データを含む一般日本語モデルで容量差を確認します。並行して、人手レビュー用に話題適合、応答の自然さ、話者marker維持、反復の評価欄を追加し、Token overlapと意味的品質を分離します。
