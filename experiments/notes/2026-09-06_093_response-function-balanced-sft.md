# 実験093：応答機能を考慮した会話SFT

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。実験092では、質問履歴を含む例を増やし、定型挨拶と初回発話を抑えました。しかし、step 7,400の固定Issue #1 promptでは挨拶への縮退が続き、held-out chat-test F1は0.212413で、実験087の0.216545を超えませんでした。質問があるかどうかだけでは、自然な応答の機能を十分に表せないと考えます。

093では、RPCとMRMPの原データについて、直前発話と応答の表面特徴から、応答の仮カテゴリを付けます。候補は、質問への回答、短い相づち、同意・否定、話題継続、会話終了、挨拶、その他です。これは教師モデルによるラベルではなく、再現可能な規則による仮分類です。分類の誤りを前提に、manifestへ規則のバージョン、カテゴリの重複、候補数、選択数を残します。

今回の仮説は、質問例を単純に増やすよりも、応答機能の偏りを抑えた方が、固定promptで入力に応じた返答が出やすくなり、held-out chat-testのshort・medium・longの複数層を改善するというものです。成功条件は、092より全体F1または複数の層別F1が改善し、固定promptで挨拶以外の意味応答が増えることです。生成token数だけが増える場合、validation lossだけが改善する場合、挨拶の種類だけが変わる場合は成功としません。

## 予定条件

モデルは実験092と同じ50,207,616 parameter、dim 576、12層、9 heads、context length 256、RoPE、LayerNorm、SwiGLUを使います。base checkpoint、Tokenizer、general validation、rehearsal token列、seed 123、batch size 8、10,000 step、learning rate 5e-5から5e-6、warmup 100、rehearsal ratio 0.20、EOS loss weight 0.50も固定します。変更するのはSFTデータの選別方法だけです。

RPCとMRMPから、それぞれresponse token約770,000を選び、合計約1.54M response tokenを作ります。各カテゴリの目標比率は、まず候補数とvalidationの分布を確認してから決めます。候補が不足する場合は無理に同数へせず、実際の選択数と不足理由を記録します。原データと`medilink_analysis`内の医師国家試験データは変更しません。

学習はColab GPUを優先し、HTTP 503が続く場合はMPSへ切り替えます。MPSで実行する場合はbackendを比較表に明記します。checkpointは`best.pt`と最新の周期checkpoint一個だけを保持し、生成サンプル、metrics、metadata、評価結果はすべて残します。checkpoint管理修正のコミットは`a1c0cb4`です。

## 実験開始前に行うこと

まず、カテゴリ規則を実装し、RPC・MRMP全候補のカテゴリ分布を集計します。次に、選別データ、manifest、選択元のSHA-256を作成し、選択だけのテストを通過させます。カテゴリ分布が極端、または規則がほとんどの応答をその他へ送る場合は、学習を始めずに分類規則を見直し、新しい実験番号を発行します。

学習開始前に、使用するconfig、選別データ、validation、base checkpoint、rehearsal token列のSHA-256と、実行コマンドをこのノートへ追記します。学習中は100 stepごとの生成とmetricsを保存し、少なくとも1,000 stepごとに解釈を追記します。

## 仮分類の実装と集計

GitHubの[Issue #1](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)は2026-09-06時点でOpenのままです。本文は、現代会話を追加候補にし、標準文のみ、標準文+RPC、標準文+RPC+MRMPを比較すること、話者境界・出所・ライセンス・ハッシュ・固定promptを記録することを求めています。093はこのうちRPC+MRMPをSFTへ使う比較の一つであり、Issueの目的と矛盾しません。RPCとMRMPは2022年前後のデータを中心とするため、2026年の最新スラングの代表とは扱いません。

2026-09-06、`scripts/analyze_response_functions.py`とそのテストを追加しました。分類器は`greeting`、`closing`、`question_answer`、`backchannel`、`agreement_disagreement`、`topic_continuation`、`other`の優先順で一つだけカテゴリを付けます。質問文かどうか、定型挨拶かどうか、短い相づち、否定表現、長さ16 token以上という規則を使っており、人手ラベルではありません。

最初の集計では、長い応答の末尾に偶然「またね」が含まれる例を終了カテゴリへ入れる誤分類が見つかりました。終了カテゴリを12 token以下の短い応答に限定し、さらに「そうですね、いいですね」のような短い相づちを拾えるようにして、分類器をv3へ更新しました。テストは`PYTHONPATH=scripts uv run pytest -q tests/test_analyze_response_functions.py tests/test_prepare_quality_chat_sft.py tests/test_train_sft_torch.py tests/test_train_torch.py`で22件すべて通過しました。

v3の全候補集計では、RPCは315,584例・5,132,071 response tokenで、token比率はquestion_answer 25.51%、topic_continuation 48.46%、other 21.46%、backchannel 3.14%、greeting 0.52%、agreement_disagreement 0.88%、closing 0.04%でした。MRMPは81,382例・770,975 response tokenで、question_answer 13.08%、topic_continuation 15.51%、other 60.38%、backchannel 9.03%、greeting 0.87%、agreement_disagreement 1.12%、closing 0.01%でした。相づちを語句包含で拾うと、MRMPでも十分な候補が得られることが分かりました。

この分布から、MRMPでは応答機能の希少カテゴリをRPCと同じtoken比率へ揃えられないことが分かります。093ではカテゴリを完全に同率へするのではなく、RPCとMRMPで別の上限を持たせ、希少カテゴリを過剰複製せず、MRMPの`other`を無理に別カテゴリへ偽装しない設計にします。特にMRMPのtopic_continuationは約119,543 tokenしかないため、MRMP側で40%を目標にする設定は実行不能です。

集計結果は`artifacts/analysis/issue1-response-functions-v1.json`に保存し、SHA-256は`1d0b3cb26ccdc61bad92a7799e3362f7dc7e539af62f555082971b75763a9119`です。入力JSONLのSHA-256はRPCが`aba75dbbba72b2d1839c11cdc96e36ea5b87e4f3a8351175a1259dc21a3bb610`、MRMPが`93a85f6be0d300980f1c9bcc6cb65845ff7671cd0243390feee6df0a816e9c1e`、Tokenizerは`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

## 評価計画

step 7,400または最終学習のbest checkpointについて、Issue #1固定prompt、48例のheld-out chat-test、general・conversation・medical・RPC・MRMPの5領域validationを評価します。092との比較ではcheckpoint stepとbackendの違いを明示し、validation lossだけで採用を決めません。固定promptの全出力、held-out全文、評価JSON、SHA-256をGitHubへ保存します。

## 現時点の状態

このノート作成時点では、093のカテゴリ集計と仮分類器のテストまで完了し、SFT用NPZの作成と学習はまだ開始していません。MRMPで希少カテゴリが不足することを確認したため、次の操作はsource別の上限を持つ選別器の実装と、その選別結果の監査です。

## 093選別データの予定条件

RPCはresponse token比率を`question_answer 0.30`、`topic_continuation 0.45`、`other 0.20`、`backchannel 0.03`、`agreement_disagreement 0.015`、`greeting 0.004`、`closing 0.001`とします。MRMPは候補不足を反映し、`question_answer 0.13`、`topic_continuation 0.15`、`other 0.60`、`backchannel 0.09`、`agreement_disagreement 0.02`、`greeting 0.009`、`closing 0.001`とします。どちらも合計1.0です。カテゴリ予算を満たせない場合は、実際に存在する候補だけを使い、残りを他カテゴリから決定的に補充します。希少カテゴリの重複サンプリングは行いません。

各sourceからresponse token約770,000、context length 256、seed 9301で作成します。出力予定先は`artifacts/sft/issue1-functional-770k-each-v1/train.npz`と`manifest.json`です。選別前にこの条件、入力hash、分類器version、候補数、選択数、出力hashをmanifestと本ノートへ残します。

2026-09-06、最初の実データ選別は、MRMPの希少カテゴリが`first_turn`上限にも重なったため、補充候補を選べず`ValueError`で終了しました。NPZとmanifestは作成されていません。この失敗を受け、カテゴリ予算を満たせない候補群がturn上限で枯れた場合は、未選択の通常候補へフォールバックする処理を追加し、テストを増やします。データの重複複製は行いません。

補充処理を修正した後、同日中に再実行が成功しました。RPCは47,084例・770,004 response token、MRMPは81,281例・770,002 response tokenとなり、合計は128,365例・1,540,006 response tokenです。選択済みカテゴリのtoken比率は、RPCではquestion_answer 30.0001%、topic_continuation 45.0016%、other 20.0015%、backchannel 3.0009%、agreement_disagreement 1.5014%、greeting 0.4003%、closing 0.0943%でした。MRMPではquestion_answer 13.0730%、topic_continuation 15.5097%、other 60.3818%、backchannel 9.0300%、agreement_disagreement 1.1195%、greeting 0.8744%、closing 0.0117%でした。MRMPのagreement_disagreement・greeting・closingは候補数の上限により目標比率へ届かず、重複複製なしで他カテゴリへ補充されています。

出力は`artifacts/sft/issue1-functional-770k-each-v1/train.npz`と`manifest.json`です。NPZのSHA-256は`5688d15626eb91b923f818d6cad1348480b9182019524e5fd4d5a81cc4815526`で、manifestには入力hash、分類器version、source別のfull/selected分布、選択provenanceを保存しています。選択結果の実例数はRPC 47,084例、MRMP 81,281例、全体128,365例です。

093の学習configは`configs/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-functional-v1.toml`です。base checkpoint、Tokenizer、rehearsal、validation、seed、モデル構造、10,000 step、learning rate、EOS loss weight、rehearsal ratioは092と同じにし、SFT NPZだけを変更します。学習前にconfigとmanifestのSHA-256、実行コマンド、MPS backend、checkpoint保持数を追記します。

選別データ準備後のcommitは`3d1ba48`です。configのSHA-256は`1a8c27bafc8b170c9e762b6c6862e23d2d1292365dd692d3bf00d16b1e385f28`、NPZのSHA-256は`5688d15626eb91b923f818d6cad1348480b9182019524e5fd4d5a81cc4815526`、manifestのSHA-256は`99d2141534ad076439f37d1d428c532ea8cea6e219f12b62932f4e754520341f`です。base checkpointは`1e09a7386a630133503c12052f7701216d505cb3bca8765cade38c879ba5e8cb`、validationは`fd93655b36aafe2a823886595e7f749762800ce741087d4a39035bbe75ea63e1`、rehearsal token列は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`、Tokenizerは`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

学習開始コマンドは次のとおりです。MPSで実行し、周期checkpointは最新1個だけ保持します。

```text
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-functional-v1.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt \
  --train-data artifacts/sft/issue1-functional-770k-each-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-functional-v1-mps-10k \
  --samples-dir artifacts/samples/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-functional-v1-mps-10k \
  --lr-schedule-steps 10000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 \
  --sample-template conversation --sample-speaker-a DA --sample-speaker-b DC \
  --keep-periodic-checkpoints 1 --device mps
```

## 学習中の記録

2026-09-06 16:54 JST、上記コマンドでMPS学習を開始しました。step 1はtrain loss 3.928150、SFT loss 4.231068、rehearsal loss 2.716481、validation loss 4.042264、PPL 56.9551、経過5.02秒でした。step 100はtrain loss 3.708910、SFT loss 3.836306、rehearsal loss 3.199324、validation loss 3.489293、PPL 32.7628、経過134.27秒でした。step 200はvalidation loss 3.506221、PPL 33.3221、経過256.24秒、step 300は3.526139、PPL 33.9925、経過380.87秒、step 400は3.512475、PPL 33.5311、経過505.27秒でした。step 500はtrain loss 3.507281、SFT loss 3.741178、rehearsal loss 2.571692、validation loss 3.497676、PPL 33.0386、経過628.85秒でした。

step 100のbest checkpointは正常に`best.pt`へ保存され、step 500の周期checkpointも正常に保存されました。checkpoint保持設定により、step 500時点で`.pt`本体は`best.pt`と`step_000500.pt`の2個だけです。生成サンプルはstep 100〜500を保存しています。step 500の固定生成は会話markerの後に「こんにちは!」と返しており、まだ挨拶から広がっていません。学習は継続中です。

step 600はvalidation loss 3.504400、PPL 33.2615、経過756.65秒、step 700は3.484157、PPL 32.5949、経過879.96秒、step 800は3.467179、PPL 32.0462、経過1004.96秒でした。step 900は3.457781、PPL 31.7465、経過1130.22秒、step 1000はtrain loss 2.967423、SFT loss 3.123777、rehearsal loss 2.342007、validation loss 3.448577、PPL 31.4556、経過1256.09秒でした。step 800、900、1000でbest checkpointが順に更新されました。

step 1000時点で、周期checkpointは`step_001000.pt`一個、bestは`best.pt`一個だけで、保存失敗や一時ファイルの残留はありません。空き容量は約2.7GiBです。step 1000の固定生成は引き続き短い挨拶形式であり、会話機能の成功判定は学習完了後のIssue #1 promptとheld-out評価まで保留します。

step 1100はvalidation loss 3.473346、PPL 32.2444、経過1385.23秒、step 1200は3.456418、PPL 31.7032、経過1508.25秒でした。step 1300は3.441512、PPL 31.2341、経過1632.86秒、step 1400は3.434141、PPL 31.0048、経過1757.54秒でbest checkpointを更新しました。step 1500はtrain loss 3.037303、SFT loss 3.022823、rehearsal loss 3.095221、validation loss 3.442286、PPL 31.2583、経過1883.02秒でした。step 1500でもNaN、OOM、shape error、checkpoint保存エラーはありません。学習は継続中です。

step 1600はvalidation loss 3.417994、PPL 30.5082、経過2009.37秒、step 1700は3.411189、PPL 30.3012、経過2134.50秒、step 1800は3.408966、PPL 30.2340、経過2260.90秒でした。step 1900は3.394762、PPL 29.8076、経過2386.82秒、step 2000はtrain loss 2.765648、SFT loss 2.865922、rehearsal loss 2.364552、validation loss 3.379617、PPL 29.3595、経過2512.75秒でした。step 1600から2000までvalidation lossは連続して改善し、step 2000でbest checkpointを更新しました。学習は継続中です。
