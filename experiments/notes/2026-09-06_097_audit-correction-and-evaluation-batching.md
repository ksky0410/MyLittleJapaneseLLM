# 実験097：監査照合の訂正とvalidation評価窓の修正

## 目的

実験096の監査で見つかった二つの実装上の問題を、既存の結果を消さずに訂正する。第一に、Issue #1の短いprompt監査でSFT採用例を照合する際、splitごとに振り直される`record_index`を使っていたため、validation/testに偽の採用判定が発生していた。第二に、PyTorchのvalidation評価で`eval_batches=20`を20個の窓として扱っており、batch size 8なら実際には3バッチしか評価していなかった。

本作業では、監査結果の意味を正しくし、今後のtrain/evaluateのvalidation lossをより安定して比較できる状態にする。重みの学習やデータの削除は行わない。

## 事前仮説

1. SFT manifestと会話データは、`source`・`conversation_id`・`target_index`で照合すれば、SFT採用例はtrain splitだけに残る。
2. `eval_batches=20`、`batch_size=8`のvalidationは、従来の20例・3物理バッチではなく、最大160例・20物理バッチになる。
3. 最後のバッチが短い場合でも、各バッチlossを単純平均せず、サンプル数で重み付けすれば評価値の偏りを減らせる。

## 変更内容

- `scripts/audit_issue1_short_prompts.py`のSFT採用照合キーを`(source, conversation_id, target_index)`へ変更した。
- v1の監査JSON/Markdownは過去記録として保持し、修正版をv2として別ファイルに保存する。
- `scripts/train_torch.py`の評価窓生成を、`batch_size * batches`例まで作る方式へ変更した。
- `scripts/train_torch.py`と`scripts/evaluate_torch.py`で、短い最終バッチをサンプル数で重み付けするようにした。
- 分割境界の誤照合、評価バッチ数、短い最終バッチの重み付けをテストする。

## 実行前の状態

修正コードはコミット`755d6ea`としてGitHubの`main`へpush済みである。ユーザーのローカルチャットアプリ関連の未コミット変更は対象外として保持する。v1監査のJSONとMarkdownは削除・上書きしない。

## 実行予定コマンド

```text
PYTHONPATH=scripts uv run pytest -q tests/test_audit_issue1_short_prompts.py tests/test_evaluation_batching.py tests/test_train_torch.py

PYTHONPATH=scripts uv run python scripts/audit_issue1_short_prompts.py \
  --evaluation artifacts/evaluations/issue1-both-50m-functional-mps-best-step10000-chat-test-v1.json \
  --selection experiments/evaluation/chat-test-v1.json \
  --selected-manifest artifacts/sft/issue1-functional-770k-each-v1/manifest.json \
  --checkpoint artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-functional-v1-mps-10k/best.pt \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --source-file rpc=train=artifacts/corpus/conversation-sft-sources-v1/rpc/train.jsonl \
  --source-file rpc=validation=artifacts/corpus/conversation-sft-sources-v1/rpc/validation.jsonl \
  --source-file rpc=test=artifacts/corpus/conversation-sft-sources-v1/rpc/test.jsonl \
  --source-file mrmp=train=artifacts/corpus/conversation-sft-sources-v1/mrmp/train.jsonl \
  --source-file mrmp=validation=artifacts/corpus/conversation-sft-sources-v1/mrmp/validation.jsonl \
  --source-file mrmp=test=artifacts/corpus/conversation-sft-sources-v1/mrmp/test.jsonl \
  --output artifacts/analysis/issue1-short-prompt-audit-v2-corrected.json \
  --markdown-output artifacts/analysis/issue1-short-prompt-audit-v2-corrected.md
```

## 成功判定

修正テストが成功し、v2監査でvalidation/testの`sft_selected`偽陽性が消え、trainの採用数だけが残ること。評価バッチのテストが、指定数の物理バッチとサンプル数重み付けを確認できること。結果とハッシュを本ノートへ追記し、コード・成果物・ノートを別々にcommitしてpushすること。

## 結果

対象テストは`9 passed`となった。最初の監査再実行は、私がTokenizerのパスを`mixed-ja-80-10-80-v2-unigram.model`と誤入力したため失敗した。既存ファイルを変更する前に`mixed-ja-80-10-10-v2-unigram.model`へ訂正して再実行し、監査自体は成功した。この失敗は学習や成果物を変更していない。

修正版監査は、コード修正後のGitコミット`755d6ea9fda54861da176b9caa710486d7c659e2`で実行した。修正版JSONは`artifacts/analysis/issue1-short-prompt-audit-v2-corrected.json`、Markdownは`artifacts/analysis/issue1-short-prompt-audit-v2-corrected.md`である。JSONのSHA-256は`24054ba73e4934d8d846ec565a91dd05f754e30e6961ac0f9ab1ba38b27eed6e`、MarkdownのSHA-256は`b4f637037215dd14cde6f7aca7f2303e81804808450e7396757e0cd32f3ca04a`である。

SFT採用判定は`source`・`conversation_id`・`target_index`で照合され、全406件がtrain splitに残った。validation/testの偽陽性は0件となった。Issue #1の8表現について、完全一致は`それな`、`やば`、`おつかれ`の各1件のみであり、`今日なにしてた？`と`明日ひま？`は引き続き0件だった。部分一致は`まじで`161件、`それな`711件、`やば`746件、`なんかさ`5件、`いやそれは`6件、`おつかれ`38件である。v1でvalidation/testに出ていたSFT採用数は、splitローカルな`record_index`の偶然一致による偽陽性だった。

評価窓は`eval_batches=20`、`batch_size=8`なら最大160例を20物理バッチに分ける。データが少ないときだけ最後までの実例数に合わせて短くなり、lossは実際のバッチ例数で重み付けする。これにより、今後のPyTorch pretrainingとdomain評価のvalidation lossを旧実装と同じ数値として比較してはいけない。既存checkpointの重みは変更していないため、旧結果は過去記録として保持する。

## 解釈と次の一手

監査の偽陽性は修正できたが、短い口語表現の完全一致不足と、093の自然な日本語の弱さは解消していない。評価処理の信頼性が上がったので、次は現代的な一般日本語を含む事前学習データを増やし、同じ50Mモデルをより多い総提示token数まで継続学習する。学習開始前にデータハッシュ、初期checkpoint、累積提示token数、評価セットを新しい実験ノートへ固定する。
