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

2026-09-06、`scripts/analyze_response_functions.py`とそのテストを追加しました。分類器は`greeting`、`closing`、`question_answer`、`backchannel`、`agreement_disagreement`、`topic_continuation`、`other`の優先順で一つだけカテゴリを付けます。質問文かどうか、定型挨拶かどうか、短い相づち、否定表現、長さ16 token以上という規則を使っており、人手ラベルではありません。

最初の集計では、長い応答の末尾に偶然「またね」が含まれる例を終了カテゴリへ入れる誤分類が見つかりました。終了カテゴリを12 token以下の短い応答に限定し、さらに「そうですね、いいですね」のような短い相づちを拾えるようにして、分類器をv3へ更新しました。テストは`PYTHONPATH=scripts uv run pytest -q tests/test_analyze_response_functions.py tests/test_prepare_quality_chat_sft.py tests/test_train_sft_torch.py tests/test_train_torch.py`で22件すべて通過しました。

v3の全候補集計では、RPCは315,584例・5,132,071 response tokenで、token比率はquestion_answer 25.51%、topic_continuation 48.46%、other 21.46%、backchannel 3.14%、greeting 0.52%、agreement_disagreement 0.88%、closing 0.04%でした。MRMPは81,382例・770,975 response tokenで、question_answer 13.08%、topic_continuation 15.51%、other 60.38%、backchannel 9.03%、greeting 0.87%、agreement_disagreement 1.12%、closing 0.01%でした。相づちを語句包含で拾うと、MRMPでも十分な候補が得られることが分かりました。

この分布から、MRMPでは応答機能の希少カテゴリをRPCと同じtoken比率へ揃えられないことが分かります。093ではカテゴリを完全に同率へするのではなく、RPCとMRMPで別の上限を持たせ、希少カテゴリを過剰複製せず、MRMPの`other`を無理に別カテゴリへ偽装しない設計にします。特にMRMPのtopic_continuationは約119,543 tokenしかないため、MRMP側で40%を目標にする設定は実行不能です。

集計結果は`artifacts/analysis/issue1-response-functions-v1.json`に保存し、SHA-256は`1d0b3cb26ccdc61bad92a7799e3362f7dc7e539af62f555082971b75763a9119`です。入力JSONLのSHA-256はRPCが`aba75dbbba72b2d1839c11cdc96e36ea5b87e4f3a8351175a1259dc21a3bb610`、MRMPが`93a85f6be0d300980f1c9bcc6cb65845ff7671cd0243390feee6df0a816e9c1e`、Tokenizerは`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

## 評価計画

step 7,400または最終学習のbest checkpointについて、Issue #1固定prompt、48例のheld-out chat-test、general・conversation・medical・RPC・MRMPの5領域validationを評価します。092との比較ではcheckpoint stepとbackendの違いを明示し、validation lossだけで採用を決めません。固定promptの全出力、held-out全文、評価JSON、SHA-256をGitHubへ保存します。

## 現時点の状態

このノート作成時点では、093のカテゴリ集計と仮分類器のテストまで完了し、SFT用NPZの作成と学習はまだ開始していません。MRMPで希少カテゴリが不足することを確認したため、次の操作はsource別の上限を持つ選別器の実装と、その選別結果の監査です。
