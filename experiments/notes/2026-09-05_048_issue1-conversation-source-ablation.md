# 実験048：Issue #1の会話source別ablation

## Issueの確認と今回の目的

2026-09-05、GitHub Issue [#1「現代的な会話日本語コーパスを追加候補にする」](https://github.com/ksky0410/MyLittleJapaneseLLM/issues/1)を確認しました。Issueでは、整った文章だけでなく現代的な雑談日本語を取り込むこと、RealPersonaChatとMulti-Relational Multi-Party Chat Corpus（MRMP）を比較すること、通常の事前学習と会話形式SFTを分けること、固定promptと生成結果を残すことが提案されています。

既存実験では両コーパスを混ぜた会話sourceを一般・医療データへ加えており、会話SFTやheld-out chat評価も実装済みです。一方、RealPersonaChat単独とMRMP単独を、同じTokenizer、同じ総Token予算、同じモデル、同じseedで比較するsource ablationはまだ独立した実験として整理されていません。今回は、会話sourceの種類が一般日本語のloss、会話validation、医療validation、固定chat-testの生成へ与える違いを探索します。

## 仮説と比較条件

RealPersonaChatは1対1の比較的長い雑談、MRMPは複数話者・mention・関係性・短い相づちを含むため、同じ会話Token比率でも現れる能力が異なると予想します。RealPersonaChatだけを加えた条件は自然な応答の継続や話題維持に、MRMPだけを加えた条件は短い相づちや複数話者markerに有利かもしれません。ただし、lossやToken overlapが下がっても、短くEOSへ到達しただけの可能性があるため、生成TXTを目視できる状態で残します。

全条件でgeneral 1、medical 0.1を基礎sourceとし、総Token予算を約1,000,000 Tokenへそろえます。会話を含む条件では総会話weightを1とし、両source条件では0.5ずつに分けます。

| 条件 | general | medical | RealPersonaChat | MRMP | 目的 |
| --- | ---: | ---: | ---: | ---: | --- |
| core | 9.0 | 1.0 | なし | なし | 会話を入れない基準 |
| rpc | 8.0 | 1.0 | 1.0 | なし | 1対1雑談の追加 |
| mrmp | 8.0 | 1.0 | なし | 1.0 | 複数話者会話の追加 |
| both | 8.0 | 1.0 | 0.5 | 0.5 | 両sourceの併用 |

単位の途中分割や複製は行いません。そのため、実際のsource別Token比率は論理単位の長さと予算により希望weightからずれる可能性があります。各manifestの実測Token数と比率を優先して解釈します。coreは「標準文のみ」ではなく、医師国家試験データを継続的に活用する本プロジェクトの方針に合わせた「一般文 + 医療」の基準です。医療専用モデルを作る実験ではありません。

## データと再現性

入力は既存の`artifacts/corpus/aozora-general-v1.txt`、`artifacts/corpus/medical-qb-v2/train.txt`、`artifacts/corpus/conversation-v1/train.jsonl`です。会話JSONLから`dataset`フィールドでRealPersonaChatとMRMPを分離し、話者markerと会話start/endを保持した派生テキストを`artifacts/corpus/conversation-sources-v1/`へ作成します。source分離の入力manifest、各splitの件数、文字数、出力SHA-256は同ディレクトリのmanifestへ保存します。

会話のvalidationとtestも同じ方法で分離し、source別validation Token列を評価用に作成します。学習へはtrain splitだけを使い、fixed chat-test-v1は`experiments/evaluation/chat-test-v1.json`で48例を固定します。一般validationは`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、医療validationは`artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin`を使います。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。モデルはPyTorch版の約5M parameter、dim 240、6層、6 heads、context length 256、absolute position embedding、GELU、LayerNormを使います。全条件でbatch size 8、500 step、AdamW、learning rate 3e-4から3e-5、warmup 50、weight decay 0.1、seed 42、eval/sample interval 100を固定します。今回はIssue #1のsource差を優先するため、RoPEやSwiGLUなどの構造変更は加えません。

## 実行前の予定

実験開始前の基準commit、分離スクリプト、データ加工コマンド、各混合manifest、Token列hash、4条件のconfig hashをこのノートへ追記します。予定コマンドは次の流れです。

```bash
.venv/bin/python scripts/split_conversation_sources.py \
  --input-dir artifacts/corpus/conversation-v1 \
  --output-dir artifacts/corpus/conversation-sources-v1

.venv/bin/python scripts/mix_corpora.py \
  --source general=artifacts/corpus/aozora-general-v1.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --weight general=9 --weight medical=1 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --target-tokens 1000000 --seed 42 \
  --output artifacts/corpus/issue1-core-1m.txt \
  --manifest artifacts/corpus/issue1-core-1m.manifest.json
```

同じ形式でrpc、mrmp、bothの3条件を作成し、既存TokenizerでToken化します。各条件のPyTorch学習では生成文をstep 0と100 stepごとに保存し、checkpoint metadataとmetricsを残します。学習後はgeneral、conversation、medical、RealPersonaChat、MRMPのdomain評価、固定chat-test-v1のJSON/TXT評価、最良checkpointのreloadを実施します。

成功条件は、4条件のデータ加工とToken化が入力hashの検証つきで完了し、各500 step学習がNaN、OOM、shape error、Token列不足なしに完走し、生成文・metrics・checkpoint metadata・評価結果を保存できることです。性能面では事前に数値の閾値を置かず、同じ条件間の差と生成内容を記録します。CPU実行になった場合は、速度をT4やMLXと比較せず、source差の探索結果として扱います。

## 実験中の記録

この節には、データ加工、Token化、各条件の開始・途中・終了、異常、実測metrics、生成文の変化、評価結果を作業中に追記します。失敗した条件や崩れた生成も削除しません。元の公開会話リポジトリ、`artifacts/corpus/conversation-v1/`、`/Users/koseki/projects/medilink_analysis`の原本は変更しません。

## 結果と解釈

4条件の結果を、一般言語モデリング、source別適応、会話の自然さ、医療validationの忘却または改善に分けて解釈します。Token overlapは意味的な正しさの代用ではないため、生成TXTの目視レビュー欄と併記します。医師国家試験データを含む出力を医学的助言や医学的正解として扱いません。

## 次に試すこと

source差が確認できた場合は、最も有望な条件について学習stepまたはToken予算を増やし、容量や構造の影響と分けて追試します。source差が小さい場合は、通常pretrainingでの会話追加より、応答maskを使うSFTとrehearsalの比較を優先します。その後、Issue #1にある固定promptの人手レビューを入力し、自動指標では見えない文脈適合・役割適合・崩壊を条件別に比較します。
