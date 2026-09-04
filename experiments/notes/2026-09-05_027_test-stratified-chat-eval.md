# 実験027：未使用test splitの層別会話評価

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験025と026では会話validationからheld-out例を選びましたが、そのvalidationはSFT学習中のloss監視とcheckpoint選択にも使われています。また、短文samplingの効果を検証するには、短い応答だけでなく中程度・長い応答を同じ固定例で比較する必要があります。今回は未使用の`artifacts/corpus/conversation-v1/test.jsonl`から評価例を作り、validationの再利用を避けます。

評価セットは応答本文のTokenizer Token数で三層に分けます。本文8 Token以下をshort、9〜24 Tokenをmedium、25 Token以上をlongとし、各層16例、合計48例を選びます。各会話から最大1例に限定し、seed 42で選択順を固定します。選択manifestには会話ID、record index、target turn、話者、source、参照本文、参照Token数、元の履歴Token数、context 256を超えたかを保存します。test入力のSHA-256とTokenizerの識別情報も記録し、同じmanifestを三つのcheckpointへ渡します。

比較対象は、token-budget pretrainingのstep 500、実験025の通常rehearsal 0.25 step 2,000、実験026の短文sampling + rehearsal 0.25 step 2,000です。生成条件はtemperature 0.8、top-k 40、最大64 Token、評価seed 42を維持します。主に層別のEOS停止、生成長、Token overlap precision・recall・F1を保存しますが、Token overlapは意味理解を測らないため、この実験単独で会話能力の成功を認定しません。

仮説は、実験026の短文samplingによるEOS停止率とoverlap F1の改善が、validationの選択偶然によらずtestのshort層でも再現するというものです。反対に、shortだけ改善してmedium・longが悪化するなら、短文samplingは一般的な会話能力ではなく出力長の調整にとどまると判断します。controlとtreatmentは同じ48例・同じ生成seedで比較し、checkpointは最終step 2,000に固定します。

会話本文のtrain/test重複は、選択時に完全一致本文の候補を照合してフラグにします。一般的な挨拶のような頻出文は重複しても直ちに除外せず、明確な完全重複を記録したうえで、結果の解釈に注意書きを付けます。test JSONLや大きな元データはGitへコピーせず、manifestと評価成果物だけを追跡します。元データは読み取り専用で扱い、削除・変更しません。

開始前のGitコミットは、実験026の結果を記録した`2467999`（`exp: record stratified short response results`）です。選択スクリプトと評価CLIを実装し、52件のテスト、ruff check、ruff format --checkを通過しました。各層のsource偏りを確認したところ、長文層がRealPersonaChatへ寄っていたため、各層でRealPersonaChatとMRMPをできるだけ均等に選ぶ修正を`c66ed9f`へ追加し、評価開始前に再生成します。

実行するコマンドは次のとおりです。

```bash
.venv/bin/python scripts/select_chat_eval_set.py \
  --config configs/token-budget-chat-sft-5m-smoke.toml \
  --input artifacts/corpus/conversation-v1/test.jsonl \
  --train-input artifacts/corpus/conversation-v1/train.jsonl \
  --output experiments/evaluation/chat-test-v1.json \
  --per-stratum 16 --seed 42
```

選択manifest作成後、同じmanifestを次の三つへ渡します。

```bash
.venv/bin/python scripts/evaluate_chat_dataset.py \
  --config configs/token-budget-chat-sft-5m-smoke.toml \
  --checkpoint CHECKPOINT \
  --input artifacts/corpus/conversation-v1/test.jsonl \
  --selection-file experiments/evaluation/chat-test-v1.json \
  --output OUTPUT_JSON \
  --text-output OUTPUT_TXT \
  --max-new-tokens 64 --seed 42
```

三つのcheckpointは、token-budget pretraining step 500、通常rehearsal 0.25 step 2,000、短文sampling + rehearsal step 2,000です。

選択manifestは2026-09-05に作成しました。short・medium・longは各16例、各層のRealPersonaChatとMRMPは各8例、全48例の会話IDは一意です。train本文との完全一致候補は7例、context 256を超える履歴は33例でした。評価JSONにもこの情報を引き継ぎ、重複候補と履歴切り詰めの影響を後から分解できるようにします。

## 実験中の記録

未実施です。開始前に選択manifestとそのハッシュをGitHubへ保存し、評価中はcheckpointごとのJSON・TXTを確認します。

2026-09-05の再実行前に、固定manifest使用時のメタデータを補正しました。CLIの`--examples`既定値は24ですが、manifestには48例が固定されており、この場合は既定値を評価例数として記録するとJSONの説明が実際の評価と食い違います。そこで`selection_file`を使ったときの`max_examples`を`null`として保存するように変更しました。変更後に53件のテスト、`ruff check .`、`ruff format --check .`が成功しています。この補正を含むコミットを評価再実行前に作成し、3条件のJSONとTXTを再生成します。

## 結果と解釈

未実施です。

## 次に試すこと

test層別評価の結果を確認した後、必要なら48例へ人手判定用の三項目（文脈適合、応答役割、明らかな崩壊）を付けます。評価層やモデル条件を同時に変えず、短文samplingの内容面の効果を分離します。
