# 実験025：会話SFTの学習step数スケーリング

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験019〜024では、5M級モデルを500 stepだけ会話SFTし、rehearsalによる忘却抑制を確認しました。しかしtrain batchで実際に見るSFT例は約4,000件に過ぎず、396,966例ある会話trainの一部しか見ていません。今回は、SFT-onlyとrehearsal 0.25を同じ2,000 stepまで延長し、500 stepで見えた差が学習不足によるものか、方法の差として残るかを調べます。

仮説は、学習stepを増やすとSFT-onlyの条件付き応答とheld-out overlap F1が改善する一方、通常domain lossの忘却は強くなる可能性があり、rehearsalはその悪化を抑えるというものです。SFT-onlyとrehearsalを同じbase checkpointから独立に再開し、差分はrehearsal objectiveの有無だけにします。

両方ともモデルはdim 240・6層・6 heads・context 256・absolute position embedding、Tokenizer・SFT data・seed 42・batch size 8・学習率5e-5・minimum learning rate 5e-6・warmup 50・weight decay 0.01です。SFT validation lossは応答maskだけで計算します。rehearsal側はpretraining Token列を25%の独立full lossとして結合します。`--max-steps 2000`で学習stepだけを設定ファイルの500から変更します。

学習後は、SFT mask validation、通常のgeneral・conversation・medical validation、Issue #1構造化prompt、held-out validation会話24例のoverlap指標を両モデルで比較します。長時間実験のため、1000 stepを超えてmetrics記録を空けないよう、既存の100 step間隔を保ちます。SFT-onlyが自然になったように見えても、生成TXT・domain loss・overlap F1を分けて評価します。

実験前のGitコミットは`10b5fbd`（`exp: record multiseed heldout chat eval`）です。使用コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train_sft.py \
  --config configs/token-budget-chat-sft-5m-smoke.toml \
  --base-checkpoint artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz \
  --train-data artifacts/sft/chat-v1-context256/train.npz \
  --validation-data artifacts/sft/chat-v1-context256/validation.npz \
  --output-dir artifacts/checkpoints/token-budget-chat-sft-5m-2k \
  --samples-dir artifacts/samples/token-budget-chat-sft-5m-2k \
  --max-steps 2000

.venv/bin/python scripts/train_sft.py \
  --config configs/token-budget-chat-rehearsal-sft-5m-smoke.toml \
  --base-checkpoint artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz \
  --train-data artifacts/sft/chat-v1-context256/train.npz \
  --validation-data artifacts/sft/chat-v1-context256/validation.npz \
  --rehearsal-tokens artifacts/tokens/mixed-ja-token-budget-1m-train.bin \
  --rehearsal-ratio 0.25 \
  --output-dir artifacts/checkpoints/token-budget-chat-rehearsal-sft-5m-2k \
  --samples-dir artifacts/samples/token-budget-chat-rehearsal-sft-5m-2k \
  --max-steps 2000
```

成功判定は両モデルが2,000 stepまで完了し、100 step間隔のmetrics、全stepの生成サンプル、summary、checkpoint metadataが保存されることです。SFT-onlyとrehearsalの優劣は、事前の仮説に合わない結果も含めて評価します。

## 実験中の記録

学習は2026-09-05に開始し、SFT-onlyとrehearsal 0.25を並列に実行しています。100 step間隔でmetricsを保存し、1,000 step時点ではSFT-onlyのvalidation lossが4.4665（perplexity 87.05）でした。rehearsal側は900 step時点でvalidation loss 4.5039（perplexity 90.37）であり、途中経過ではSFT-onlyがやや先行しています。これは学習率が減衰中の途中値であり、忘却とheld-out応答の結果は学習完了後に評価します。

1,000 stepまでのログは各モデルの`metrics.jsonl`に保存されています。学習中の固定prompt生成も各stepのTXTとして保存される設定です。

両モデルとも2,000 stepを完走しました。SFT-onlyは約771.6秒、rehearsal 0.25は約816.6秒でした。両方ともstep 1、100、200から2,000まで100 step間隔のmetricsとcheckpoint metadata、stepごとの固定prompt生成を保存できました。最終かつ最良のSFT validation lossは、SFT-onlyが4.4015（perplexity 81.57）、rehearsalが4.4089（perplexity 82.18）でした。SFT validationだけではSFT-onlyが0.0074良好ですが、これは会話応答mask上の指標であり、通常コーパスの保持を表しません。

domain評価は、最初にREADMEの標準例にあるAozora・会話・医療validationで実行してしまいました。その結果は比較条件を混ぜないよう、`artifacts/evaluations/token-budget-chat-sft-5m-2k-domains-aozora-neko.json`と`artifacts/evaluations/token-budget-chat-rehearsal-sft-5m-2k-domains-aozora-neko.json`へそのまま保存しています。その後、過去の500 step評価と同じ`mixed-ja-80-10-10-v2-*` validationへ揃えて再評価しました。以降の比較には後者だけを使います。

## 結果と解釈

同じmixed validationで測ったbase（token-budget pretrainingのstep 500）、今回のSFT-only、rehearsal 0.25のlossは次のとおりです。

| 条件 | general | conversation | medical |
| --- | ---: | ---: | ---: |
| base pretraining step 500 | 5.6064 | 3.8523 | 4.9090 |
| SFT-only step 2,000 | 6.0144 | 4.2655 | 5.5052 |
| rehearsal 0.25 step 2,000 | 5.5742 | 3.6211 | 4.8147 |

SFT-onlyはbaseと比べてgeneralが0.4080、conversationが0.4132、medicalが0.5962悪化しました。会話SFT validation lossが下がっていても、通常のToken列に対する予測能力を失っており、2,000 stepまで同じ学習率スケジュールで続けたことによる忘却が明確です。rehearsal 0.25はbaseよりgeneralが0.0321、conversationが0.2313、medicalが0.0943改善しました。少なくとも今回の評価範囲では、rehearsalは忘却を抑えるだけでなく、pretraining dataとの混合学習として通常domainのlossも改善しています。

held-out validation会話24例では、SFT-onlyがEOS停止11/24、平均生成43.83 token、Token overlap precision 0.1177、recall 0.2738、F1 0.1264でした。rehearsalはEOS停止15/24、平均生成32.75 token、precision 0.1833、recall 0.2692、F1 0.1473でした。rehearsalはSFT-onlyより生成を短くし、停止率・precision・F1を改善しましたが、recallはほぼ同程度でわずかに下がりました。F1の差は0.0208で、500 stepの複数seed評価で見えた傾向と同じ方向です。ただし、Token overlapは意味や自然さを直接測らない補助指標です。

Issue #1の構造化固定promptでは、2,000 stepへ延長しても両モデルとも「こんにちは」「こんばんは」「そうなんですね」などの定型挨拶へ強く偏りました。raw promptではSFT-onlyが長く生成し、rehearsalは比較的短く停止しましたが、どちらも入力の「まじで」「今日なにしてた？」「明日ひま？」へ内容に応じた返答はできず、古風な文体・医療問題・数字・話者markerの混在も観察されました。したがって、学習stepを増やすだけではIssue #1の短い現代会話応答は得られませんでした。

代表的な最終固定prompt生成として、SFT-onlyは`今日はおかるいからしうときに行っていました!...`のように長く崩れた文を生成し、rehearsalは`今日は今日と聞いたことが多い。「其事を不立物を付けて...`のように短めでも古典的な文体を残しました。これらの崩れた生成も削除せず、stepごとのTXTと評価TXTへ保存しています。

今回の仮説は二つに分かれました。学習stepを増やせばSFT-onlyの会話応答指標が改善するという予想は、held-out F1が500 stepの0.1659から0.1264へ下がり、支持されませんでした。rehearsalが通常domainの悪化を抑えるという予想は、3 domainすべてでbaseを下回ったため支持されました。なお、2,000 step runのSFT-onlyとrehearsalは学習率スケジュールも同じであり、差分はrehearsal objectiveの有無です。

成果物は次の場所に保存しています。

- [SFT-only summary](../../artifacts/checkpoints/token-budget-chat-sft-5m-2k/summary.json)
- [rehearsal summary](../../artifacts/checkpoints/token-budget-chat-rehearsal-sft-5m-2k/summary.json)
- [SFT-only metrics](../../artifacts/checkpoints/token-budget-chat-sft-5m-2k/metrics.jsonl)
- [rehearsal metrics](../../artifacts/checkpoints/token-budget-chat-rehearsal-sft-5m-2k/metrics.jsonl)
- [SFT-only domain evaluation](../../artifacts/evaluations/token-budget-chat-sft-5m-2k-domains.json)
- [rehearsal domain evaluation](../../artifacts/evaluations/token-budget-chat-rehearsal-sft-5m-2k-domains.json)
- [SFT-only held-out JSON](../../artifacts/evaluations/token-budget-chat-sft-5m-2k-heldout-chat.json)
- [rehearsal held-out JSON](../../artifacts/evaluations/token-budget-chat-rehearsal-sft-5m-2k-heldout-chat.json)
- [SFT-only generated samples](../../artifacts/samples/token-budget-chat-sft-5m-2k)
- [rehearsal generated samples](../../artifacts/samples/token-budget-chat-rehearsal-sft-5m-2k)

## 次に試すこと

2,000 stepまで増やしても短い固定promptが改善しなかったため、さらに同じSFTを長く回す前に、Issue #1の評価promptを相づち・質問・同意/不同意・誘い・別れに層別化し、各カテゴリへ適合した返答の割合を数える評価を追加します。そのうえで、短い応答を含むSFT例を適切に再サンプリングする実験を一つだけ行い、rehearsal 0.25を基準条件として比較します。モデルやTokenizerを同時に変更せず、データ構成の効果を切り分けます。
