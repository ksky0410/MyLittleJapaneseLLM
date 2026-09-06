# 実験094：held-out分布と短い口語promptの監査

## 目的

実験093では、RPCとMRMPの応答機能をsource別に選別したSFTによって、held-out chat-testのtoken overlap F1と5領域のvalidation lossが改善しました。しかし、Issue #1で定めた8個の短い口語promptは、すべて挨拶へ崩壊しました。

この差が、モデルの会話能力不足によるものなのか、固定promptが学習・評価データの分布から外れているためなのかを切り分けます。Issue #1は2026-09-06時点でOpenであり、RPCとMRMPを現代会話の候補として比較する方針を示しています。本実験ではその方針を維持し、教師LLMによる蒸留は行いません。

## 事前仮説

1. 固定promptの8例は、実際の会話評価に比べて履歴が短すぎ、応答対象の話者や直前文脈がないため、現在のモデルでは入力依存の応答を選びにくい。
2. held-out 48例には、学習文との完全一致、source別の偏り、履歴の切り詰めが混在しているため、093のF1改善だけでは自然な会話能力を十分に説明できない。
3. Issue #1の口語表現が学習コーパスに存在しても、短い入力として独立して現れるとは限らない。表現の出現数、直後応答の機能、SFT選別への採用状況を確認する必要がある。

## 今回の操作

まず、093のheld-out評価JSONと固定評価JSON、評価manifest、RPC・MRMPのtrain/validation/test JSONL、093のSFT選択provenanceを入力にして監査します。監査では次を数えます。

- source、short/medium/long、履歴切り詰め、学習本文との重複別の評価件数
- 評価対象の履歴token長と応答token長の分布
- 8個のIssue #1 promptの完全一致・部分一致の出現箇所
- 一致した発話の直後応答、応答機能カテゴリ、split、SFT選択への採用状況

出力はJSONと可読Markdownで保存し、入力ファイルのSHA-256、分類器version、チェックポイント、実験Gitコミットを記録します。今回の監査だけでは重みを変更せず、監査結果を見て次の一変数実験を決めます。

## 使用予定の実行条件

- Python環境：`uv run`
- 監査対象checkpoint：093のstep 10,000 best checkpoint
- Tokenizer：`mixed-ja-80-10-10-v2-unigram.model`
- Issue #1候補：RPC、MRMP
- 乱数：検索・集計では使用しない。生成比較を追加する場合はseed 42を固定する

## 成功判定

少なくとも、48例の層別分布と8 promptの候補出現状況を、再実行可能なファイルとハッシュ付きで説明できることを成功とします。候補が不足する場合も、見つからなかった事実を結果として記録します。自然な日本語が改善したとは、監査結果だけからは主張しません。

## 開始前の状態

093の最終コミットは`6531cee`で、`origin/main`と同期済みです。ユーザーが進めているローカルチャットアプリ関連の未コミット変更は対象外として保持します。093の固定promptでは8例すべてがEOSへ到達した一方、入力に依存せず挨拶へ崩壊しました。held-out chat-testの全体F1は0.243491でした。

## 監査の実装

`scripts/audit_issue1_short_prompts.py`を追加し、`tests/test_audit_issue1_short_prompts.py`で、応答付きのprompt検索とheld-out metadataの結合をテストしました。分類には093と同じ`analyze_response_functions.py`のv3を使い、原文を変更せずに完全一致と単純な部分一致を別々に記録します。

監査対象は093のheld-out 48例、評価manifest、093のSFT選択provenance、RPC・MRMPそれぞれのtrain/validation/test JSONLです。評価JSONのSHA-256は`219165049d152e9ac01e35490575ede288a473f75aac0722fa7da434abe2c3e6`、評価manifestは`ab2f372d4c6d5000ab0a8ec91c8d8c22837b6ffa2005e79db3f63fdc7a8ab530`、SFT manifestは`99d2141534ad076439f37d1d428c532ea8cea6e219f12b62932f4e754520341f`です。093 best checkpointは`6bbb0ff0fac6c63ce9c3a4fc807744072390bad8f7daaa35066016901fa4180f`、Tokenizerは`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

実行コマンドは次のとおりです。

```text
PYTHONPATH=scripts uv run python scripts/audit_issue1_short_prompts.py \\
  --evaluation artifacts/evaluations/issue1-both-50m-functional-mps-best-step10000-chat-test-v1.json \\
  --selection experiments/evaluation/chat-test-v1.json \\
  --selected-manifest artifacts/sft/issue1-functional-770k-each-v1/manifest.json \\
  --checkpoint artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-functional-v1-mps-10k/best.pt \\
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \\
  --source-file rpc=train=artifacts/corpus/conversation-sft-sources-v1/rpc/train.jsonl \\
  --source-file rpc=validation=artifacts/corpus/conversation-sft-sources-v1/rpc/validation.jsonl \\
  --source-file rpc=test=artifacts/corpus/conversation-sft-sources-v1/rpc/test.jsonl \\
  --source-file mrmp=train=artifacts/corpus/conversation-sft-sources-v1/mrmp/train.jsonl \\
  --source-file mrmp=validation=artifacts/corpus/conversation-sft-sources-v1/mrmp/validation.jsonl \\
  --source-file mrmp=test=artifacts/corpus/conversation-sft-sources-v1/mrmp/test.jsonl \\
  --output artifacts/analysis/issue1-short-prompt-audit-v1.json \\
  --markdown-output artifacts/analysis/issue1-short-prompt-audit-v1.md
```

## 監査結果

held-out 48例は、RPC 24例とMRMP 24例、short/medium/long各16例を均等に含みます。しかし履歴はMRMPで24例中19例、RPCで24例中14例が256 tokenへ切り詰められています。学習本文との完全一致フラグはMRMPで6例、RPCで1例あり、重複のない例だけを独立した自然さの指標とみなすことはできません。source別の平均F1はMRMP 0.258411、RPC 0.228571でした。

8個の固定promptを原文の部分一致で検索した結果、完全一致は`それな`、`やば`、`おつかれ`が各1件だけで、いずれもMRMPのtrainにあり、093のSFT選択へ入っていました。`まじで`、`今日なにしてた？`、`なんかさ`、`いやそれは`、`明日ひま？`は完全一致がありませんでした。特に`今日なにしてた？`と`明日ひま？`は、今回の6分割のコーパスにそのままの形では存在しません。

単純な部分一致は、`それな`が711件、`やば`が746件まで増えますが、`それなら`や`やばい`も含むため、固定promptそのものの学習量とは解釈できません。`まじで`は161件、`おつかれ`は38件、`なんかさ`は5件、`いやそれは`は6件でした。この監査によって、093のSFT選別データに口語コーパスが含まれていないのではなく、固定promptの表面形と実際の学習例がずれていることが確認できました。

検索結果とheld-out分布は`artifacts/analysis/issue1-short-prompt-audit-v1.json`、可読版は`artifacts/analysis/issue1-short-prompt-audit-v1.md`に保存しました。JSONのSHA-256は`5e8956c6e9125ed1ce52d8bf4924c3bedc550e911c82b8705ce656d5779c6052`、MarkdownのSHA-256は`93906861e49564455652ba9d20603c5630dba83decaa85715e7b83aff3b4806f`です。監査時点のGitコミットは`6531ceecdae588b1b1e3a3827ef72535db699ce8`です。

## 解釈と次の変更

仮説1と2は支持されました。固定promptは実際の学習例よりも短く、held-out評価はsource、履歴切り詰め、train overlapが混在しています。一方、部分一致検索だけで入力依存の応答能力を推定することはできないため、093のF1改善を取り消す理由にはなりません。仮説3も支持され、Issue #1の8表現のうち5表現は完全一致がありませんでした。

次の実験095では、既存のRPC・MRMPのvalidation/testから、単純な部分一致ではなく、短い質問・相づち・反応の表面形を保った候補を抽出し、実会話履歴付きの評価セットを作ります。まずは学習せずに093 checkpointで評価し、単独promptと実履歴付きpromptの差を測ります。その結果を確認した後、train側では一変数だけ、短い口語応答の選択比率を増やすSFTを検討します。外部教師LLMによる蒸留は行いません。
