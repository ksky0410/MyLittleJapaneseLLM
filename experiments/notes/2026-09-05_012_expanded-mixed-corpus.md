# 実験ノート：一般作品を増やした混合コーパス

## 基本情報

- 実験番号：012
- 記録日：2026-09-05
- 担当者：ユーザーとCodex
- 実験開始時のGitコミット：`bc29860`
- ブランチ：`main`
- 状態：混合コーパス作成前。Tokenizerと学習は未実施

## 仮説

実験008では一般sourceが一作品だけだったため、一般80%を維持できる総単位数が少なく、会話の一単位が長いこともあり、単位数比率とtoken寄与が大きくずれた。今回は一般sourceへ坊っちゃん・こころ・それからを追加し、一般sourceのunique単位数を5,758まで増やした。仮説は、一般sourceの容量と作品の多様性を増やせば、会話・医療を過剰採用せずに、より大きな80/10/10混合コーパスを作れるというものである。

## 実行前の条件

入力はすべてtrain splitに限定する。一般の元データは、吾輩は猫である。trainと、実験011で作った坊っちゃん・こころ・それからの結合本文である。会話は`conversation-v1/train.txt`、医療はchallengeを除いた`medical-qb-v2/train.txt`を使う。

- 一般：`artifacts/corpus/aozora-general-v1.txt`
- 会話：`artifacts/corpus/conversation-v1/train.txt`
- 医療：`artifacts/corpus/medical-qb-v2/train.txt`
- seed：42
- 重み：一般8.0、会話1.0、医療1.0
- target units：7,000
- 期待する採用単位数：一般5,600、会話700、医療700
- 出力：`artifacts/corpus/mixed-ja-80-10-10-v2.txt`
- manifest：`artifacts/corpus/mixed-ja-80-10-10-v2.manifest.json`

完全一致の重複単位はsource指定順で一度だけ採用し、会話ブロックは一会話単位のまま出力する。医療700回challengeは入力trainに含めない。Tokenizerはこの混合train本文だけで新規に学習し、validation・test・challengeは使用しない。

実行コマンドは次のとおりである。

```bash
.venv/bin/python scripts/mix_corpora.py \
  --source general=artifacts/corpus/aozora-general-v1.txt \
  --source conversation=artifacts/corpus/conversation-v1/train.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --weight general=8.0 \
  --weight conversation=1.0 \
  --weight medical=1.0 \
  --target-units 7000 \
  --seed 42 \
  --output artifacts/corpus/mixed-ja-80-10-10-v2.txt \
  --manifest artifacts/corpus/mixed-ja-80-10-10-v2.manifest.json
```

## 成功条件

採用単位数が期待値と一致し、会話start/end数が一致し、出力に`試験回：700`が含まれないこと。入力・出力SHA-256、sourceごとの採用数・文字数・実比率がmanifestに保存されること。失敗した場合は本文を削除せず、失敗内容をこのノートへ追記する。

## 結果

混合コマンドは成功し、出力は7,000単位、32,281行、2,037,962文字となった。出力SHA-256は`454d6031a634485e047b6377e4f72bfd4eb1f6657052f5ad4cd18978ee227401`である。入力23,535単位から完全一致5単位を除き、unique単位23,530となった。

採用単位数は一般5,600、会話700、医療700で、実際の比率は一般80%、会話10%、医療10%と期待どおり一致した。一般の採用文字数は694,054文字、会話は917,515文字、医療は419,393文字で、文字比率は一般34.17%、会話45.18%、医療20.65%だった。会話markerはstart/endとも700個で一致し、医療700回challengeが出力に含まれないことも確認した。したがって、単位比率は制御できたが、長い会話のためtoken寄与は別途測定する必要がある。

一般sourceは5,758 unique単位を持ち、target 7,000に対して一般5,600を無理なく確保できた。吾輩は猫である。だけに依存していた実験008と比べ、坊っちゃん・こころ・それからの追加により、一般sourceの作品多様性と容量を増やせた。混合manifestは`artifacts/corpus/mixed-ja-80-10-10-v2.manifest.json`に保存した。Token化後にはsourceごとのtoken寄与を別途測定する。学習は、混合結果とTokenizer結果を確認してから、別の学習実験として開始条件を記録して行う。
