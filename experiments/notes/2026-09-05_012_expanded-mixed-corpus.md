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

混合コマンドの直後に、実際の採用数、重複数、文字数、会話marker数、医療challenge除外確認、出力SHA-256を追記する。Token化後にはsourceごとのtoken寄与を別途測定し、単位数80/10/10がtoken数でも保たれているかを確認する。学習は、混合結果とTokenizer結果を確認してから、別の学習実験として開始条件を記録して行う。
