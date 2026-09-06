# Issue #1短い口語prompt監査

- 評価例数：48
- checkpoint：/Users/koseki/projects/small_llm/artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-functional-v1-mps-10k/best.pt
- 実験コミット：755d6ea9fda54861da176b9caa710486d7c659e2

## held-out分布

- mrmp：24例、平均F1 0.258411、平均参照長 14.38 token、平均生成長 5.67 token
- real-persona-chat：24例、平均F1 0.228571、平均参照長 16.75 token、平均生成長 9.83 token

## Issue #1 promptの出現状況

- `まじで`：合計161件（完全一致0、部分一致161、応答あり159、SFT採用30）。応答機能：other=54, question_answer=52, topic_continuation=36, backchannel=12, agreement_disagreement=4, closing=1
- `それな`：合計711件（完全一致1、部分一致710、応答あり691、SFT採用131）。応答機能：other=247, topic_continuation=282, backchannel=58, question_answer=81, agreement_disagreement=18, greeting=4, closing=1
- `今日なにしてた？`：合計0件（完全一致0、部分一致0、応答あり0、SFT採用0）。応答機能：なし
- `やば`：合計746件（完全一致1、部分一致745、応答あり730、SFT採用237）。応答機能：other=379, question_answer=69, topic_continuation=202, backchannel=74, agreement_disagreement=4, greeting=1, closing=1
- `なんかさ`：合計5件（完全一致0、部分一致5、応答あり5、SFT採用1）。応答機能：other=3, topic_continuation=1, backchannel=1
- `いやそれは`：合計6件（完全一致0、部分一致6、応答あり6、SFT採用2）。応答機能：question_answer=1, topic_continuation=2, other=3
- `おつかれ`：合計38件（完全一致1、部分一致37、応答あり34、SFT採用5）。応答機能：other=22, topic_continuation=7, question_answer=3, backchannel=1, greeting=1
- `明日ひま？`：合計0件（完全一致0、部分一致0、応答あり0、SFT採用0）。応答機能：なし

## 解釈

この監査はモデル性能を再評価するものではない。固定promptの出現数、実際の直後応答、SFT選択への採用状況、held-outの分布を分離して確認し、次の学習変更を一つに絞るために使う。
