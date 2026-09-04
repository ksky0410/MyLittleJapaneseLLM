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

2026-09-05に、固定manifestを使って三つのcheckpointを評価しました。評価開始時点のコードとノートに対する補正は`f980e18`（`fix: record fixed evaluation example metadata`）で、評価前にGitHubへpush済みです。補正後にも53件のテスト、`ruff check .`、`ruff format --check .`が成功しました。

三つの評価は同じ`experiments/evaluation/chat-test-v1.json`、seed 42、temperature 0.8、top-k 40、最大64 Tokenで実行しました。固定manifestのSHA-256は`ab2f372d4c6d5000ab0a8ec91c8d8c22837b6ffa2005e79db3f63fdc7a8ab530`であり、各JSONは48例を評価しています。manifest使用時はCLIの既定値24を`max_examples`へ記録せず、`null`として保存されることも確認しました。

実際に使ったcheckpointは、ベース条件が`artifacts/checkpoints/token-budget-mixed-ja-5m-smoke/step_000500.npz`、通常rehearsal条件が`artifacts/checkpoints/token-budget-chat-rehearsal-sft-5m-2k/step_002000.npz`、短文sampling条件が`artifacts/checkpoints/token-budget-chat-rehearsal-short-sft-5m-2k/step_002000.npz`です。評価結果のJSONと、48例それぞれのprompt・参照応答・生成文を含むTXTは、対応する`artifacts/evaluations/`と`artifacts/samples/`に保存しました。3条件とも評価処理はエラーなく終了しました。

## 結果と解釈

全体では、ベース条件は48例中48例でEOSに到達し、平均生成長は4.94 Token、Token overlapのprecision・recall・F1はそれぞれ0.1267、0.0457、0.0505でした。通常rehearsal条件はEOSが33例、平均生成長が35.35 Token、precision・recall・F1が0.1569、0.2688、0.1443でした。短文sampling条件はEOSが41例、平均生成長が24.60 Token、precision・recall・F1が0.1991、0.2277、0.1518でした。したがって、短文sampling条件は通常rehearsal条件と比べてEOSが8例増え、平均生成長が10.75 Token短くなり、precisionは上がりましたがrecallは下がりました。全体F1の改善は0.0075にとどまっています。

short層では、通常rehearsal条件のF1が0.1436、EOSが12/16、平均生成長が31.75 Tokenだったのに対し、短文sampling条件はF1が0.1992、EOSが13/16、平均生成長が31.19 Tokenでした。今回の仮説である「短文samplingの改善が未使用testのshort層でも再現する」は、Token overlapという限定的な指標では支持されました。medium層ではF1が0.1240から0.1302へ小幅に上がり、EOSは11/16から15/16へ増え、平均生成長は37.50から16.88 Tokenへ大きく短くなりました。一方でrecallは0.2049から0.1301へ下がっており、正しい内容を十分に出せたというより、短く停止しやすくなった影響が大きいと解釈します。long層ではF1が0.1654から0.1261へ下がり、EOSは10/16から13/16へ増え、平均生成長は36.81から25.75 Tokenへ短くなりました。長い応答を必要とする例では、短文samplingが内容の保持を犠牲にしている可能性があります。

生成TXTを見ると、短文sampling条件では「そうです!?」「確かに!?」のように短い反応として読める例が一部増えましたが、話題に合わないメンション、文法の崩れ、途中で別の話題へ飛ぶ出力も残っています。通常rehearsal条件より出力を止める能力は改善しましたが、文脈に沿った返答や話者役割の適合が改善したとはまだ言えません。ベース条件はほぼすべてが数Tokenで終了しており、step 500のpretraining checkpointは会話生成の比較対象としては未成熟です。

今回の評価セットは各層16例、各層でMRMPとRealPersonaChatが8例ずつになるように固定されています。ただし、選択時点でtrain本文との完全一致候補が7例あり、履歴がモデルのcontext長を超えて切り詰められた例も33例あります。これらは評価JSONの各例にフラグとして残しています。Token overlapは語彙の一致しか測らず、意味・文脈適合・話者役割を評価しないため、今回の結果だけで会話能力の向上を認定しません。人手判定はまだ実施していません。

2026-09-05に、`scripts/create_chat_review_template.py`を追加し、実験027の三条件を人手確認へ回せるよう未記入テンプレートを生成しました。各templateは元の評価JSONのSHA-256を持ち、`context_fit`、`role_fit`、`not_collapsed`を48例すべて`null`、`review_status`を`pending_human_review`として保存します。まだ人手判定は入力していないため、これらは判定結果ではなく、確認作業を再開するためのテンプレートです。実験028のFineWeb追加条件についても同じ形式を生成し、モデル条件をまたいだ目視比較へ使えるようにします。

今回の結論は、短文sampling + rehearsalは「短く終わる」挙動をtestでも再現し、short層の表面的なoverlap F1を改善した一方、medium・long層を含む一般的な会話内容の改善までは示さなかった、というものです。仮説は長さとEOS停止については部分的に支持されましたが、内容面の仮説は未支持です。

## 次に試すこと

まず48例に対して、文脈に合っているか、求められた応答役割を果たしているか、明らかな崩壊がないかを別々に判定できるレビューテンプレートを作成します。人手で確認していない値を自動で埋めることはせず、Token overlapと意味評価を分離します。

その後は、今回確認できた出力長の変化だけをモデル性能と取り違えないようにしながら、一般日本語データを増やした20M級モデルを同じtest manifestで評価します。データセットを追加する場合は、元データを変更せず、取得元・ライセンス・commitまたは取得日時・ファイルSHA-256・使用Token数を別ノートへ記録してから学習へ進みます。短文samplingの改良は、データ量またはモデル容量を変えた実験と同時に行わず、比較条件を分けます。

## 成果物のハッシュ

評価JSONと生成TXTは、再生成後に次のSHA-256になっています。ベース条件のJSONは`0cd7e22edd85f39da4476261f4fe54ea268aafa1f8f575f16bf430680263e9ce`、通常rehearsal条件のJSONは`e3f189fcfe0d376c600621748abf60b7ffb2f724854a34701a92761386b201be`、短文sampling条件のJSONは`5b18c1683b79ed96891dac1851e3aab3c3850c8603428c80501022be2852716`です。対応する生成TXTは順に`7fe9f6583be94098ffa58a9442e4ac5edbe02cd4784cf3a4d9ff27f1d16d1453`、`43108ffba97dd74848358266edcbf3c7cf7addb65cbddc5573a5b4eaf8f56f9a`、`5b0d2090184caf0164b5c585f86634901a18e49f6248e9f1d7981fc248c9ad0a`です。
