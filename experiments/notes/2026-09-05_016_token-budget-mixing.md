# 実験016：Token予算による一般・会話・医療コーパス混合

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。Issue #1の会話データと医師国家試験データを一般日本語へ混ぜる際、論理単位数の比率だけでは、一般文・会話・問題の長さの違いによって実際の学習Token比率がずれるという仮説を検証します。今回導入したToken予算経路では、同じSentencePieceモデルで各単位を数え、8:1:1のweightをToken比率として指定します。

事前の予想は、一般文は短い単位が多く、医療問題は長い単位が多いため、`target_units`の8:1:1と、`target_tokens`の8:1:1では採用単位数が一致しないことです。また、会話は会話単位を分割しないため、target token数は厳密には一致せず、最後に入る単位の大きさだけ手前で止まると予想します。

入力は、拡張した青空文庫一般コーパス`artifacts/corpus/aozora-general-v1.txt`、Issue #1由来の会話train `artifacts/corpus/conversation-v1/train.txt`、医師国家試験v2 train `artifacts/corpus/medical-qb-v2/train.txt`です。Tokenizerは既存の混合v2用Unigram `artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`を使い、source重複はsource指定順で除きます。入力の取得元、ライセンス、元SQLiteを変更していないこと、各ハッシュは既存manifestへ記録済みです。元データはGitへ追加せず、small_llm側へ加工したファイルだけを使います。

実行前のGitコミットは`3c5148e`（`feat: add token-budget corpus mixing`）です。学習はまだ行わず、今回は混合結果とmanifestの検証に限定します。使用コマンドは次のとおりです。

```bash
.venv/bin/python scripts/mix_corpora.py \
  --source general=artifacts/corpus/aozora-general-v1.txt \
  --source conversation=artifacts/corpus/conversation-v1/train.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --weight general=8.0 \
  --weight conversation=1.0 \
  --weight medical=1.0 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --target-tokens 1000000 \
  --seed 42 \
  --output artifacts/corpus/mixed-ja-token-budget-1m.txt \
  --manifest artifacts/corpus/mixed-ja-token-budget-1m.manifest.json
```

成功判定は、混合出力が入力sourceを上書きせず、manifestの`selected_token_count`が1,000,000以下で、同じコマンドを再実行したときに出力SHA-256とmanifestの集計が一致することです。各sourceのToken shareが希望比率へ近づくか、sourceの枯渇で再配分されるかも確認します。

## 実験中の記録

2026-09-05にコマンドを実行しました。処理時間は約1.59秒で、エラーや警告はありませんでした。入力23,535単位から重複5単位を除き、6,618単位を採用しました。出力は1,478,413文字、18,340行、SHA-256は`d0eb6691a25107fb2ab94b91c2a366e2a80a5fd720797fec27434a29d1cea000`です。

目標1,000,000 tokenに対して、実測は999,997 tokenでした。論理単位を壊さない仕様により3 token手前で止まりました。source別の実測は次のとおりです。

- general：5,758単位、508,278 token、50.83%
- conversation：326会話、245,674 token、24.57%
- medical：534問、246,045 token、24.60%

一般sourceは手元にある5,758単位をすべて使い切りました。そのため、指定した8:1:1のweightは、一般sourceの枯渇後に会話と医療へ再配分され、最終的なtoken比率はおおむね5:2.5:2.5になりました。これは実装の不具合ではなく、現在の一般コーパスがtoken予算の80%を満たすだけの量を持たないことによるものです。再実行では出力SHA-256、採用token数、source別比率が一致し、出力本文も完全一致しました。

manifestと入力データのSHA-256、実行条件は次の記録へ保存しています。

- [Token予算混合manifest](../../artifacts/corpus/mixed-ja-token-budget-1m.manifest.json)
- [混合出力のsha256記録](../../artifacts/corpus/mixed-ja-token-budget-1m.manifest.json)

## 結果と解釈

事前の予想どおり、単位数とtoken数の比率は大きく異なりました。会話は1会話を一単位としているため、採用単位数は326と少ない一方、1会話あたりの発話数と文字数が多く、tokenでは24.57%を占めました。医療も1問あたりの説明を含むため、534問で24.60%になりました。従来の単位数ベース混合で生じる「長いsourceがtokenでは重くなる」問題を、manifest上で明示的に観測できました。

ただし、今回の実装は指定weightをsource間の希望token比率として解釈しているものの、sourceが枯渇した後は残ったsourceへ再配分します。一般文を本当に80%前後に保ちたい場合は、青空文庫の追加作品を取り込むか、target tokenを約630,000へ下げる必要があります。今後の学習比較では、一般sourceを増やす方がデータ内容の多様性にも寄与するため、追加作品の拡張を優先します。

## 次に試すこと

まず、このToken予算混合コーパスからTokenizerとToken列を作り、既存の単位数ベース混合モデルと同じ500 stepでvalidation lossとIssue #1固定プロンプトを比較します。次に青空文庫の一般作品を追加し、8:1:1のtoken比率を実際に達成できるか確認します。なお、会話SFTを導入する際は、会話sourceをpretrainingと同じtoken予算に置くだけでなく、話者境界と応答側loss maskingを別に評価します。
