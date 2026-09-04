# 実験ノート：会話コーパスと医師国家試験データの取り込み

## 基本情報

- 実験番号：007
- 記録日：2026-09-05
- 担当者：ユーザーとCodex
- Gitコミット：会話Importerと医療Importerを確認後に確定する
- ブランチ：`main`
- 状態：データ取り込み完了。混合学習は未実施

ユーザーから、通常の日本語学習データへIssue #1の会話データと医師国家試験データも加える方針が示された。元データをリポジトリへ大量追加せず、取得元・コミット・ハッシュを記録し、small_llm側の`artifacts/`へ加工済みデータとmanifestを出力する準備を行った。医療データの元ファイル`/Users/koseki/projects/medilink_analysis/data/qb.sqlite`は読み取り専用で開き、元ディレクトリへ変更を加えていない。

このノートは、依頼を受けてから先に実施した安全な読み取り・変換の記録を、その直後に残すものである。以後、モデル学習を開始する場合は、別の実験番号を発行して開始前の条件を先に記録する。

## 会話データの出所と条件

Issue #1で候補になっていた次の2つの公開リポジトリを、深さ1のcloneとして`/tmp/my-little-japanese-llm-data/conversations/`へ取得した。取得元リポジトリ自体は変更していない。

- RealPersonaChat：https://github.com/nu-dialogue/real-persona-chat
- MRMP：https://github.com/nu-dialogue/multi-relational-multi-party-chat-corpus
- RealPersonaChatの取得コミット：`28d0b6b3865b29cabc26c230a2db37cdf315e937`
- MRMPの取得コミット：`e6e39cb896df88781c7a2e0451226a527c5f1e2e`
- ライセンス：両方ともCC BY-SA 4.0。利用時は公開元のREADME・LICENSEに従う

公開元の説明に従い、個人情報がマスクされた公開版を使う。ただし、個人特定や特定話者へのなりすましに用いないという注意を守り、話者の属性・評価・timestamp・persona・mention情報は学習本文へ出さない。Importerは会話ID、元ファイル、話者ID、ターン順、発話本文だけをJSONLへ保存し、TXTでは`<|startofconversation|>`、`<|speaker:ID|>`、`<|endofconversation|>`で境界を保持する。

## 医療データの出所と条件

- 入力：`/Users/koseki/projects/medilink_analysis/data/qb.sqlite`
- 入力SHA-256：`5499dff6f181a845b7a087a55b78606869aa5664e0c351c8f8889b719df1ec14`
- 対象テーブル：`questions`、`descriptions`
- 取り込み出力：`artifacts/corpus/medical-qb-v1/`
- 問題数：6,986問。採用6,986問。スキップ0問
- 説明欠損：0問。選択肢解説欠損：37問
- 画像問題：1,808問。画像URLは保存せず、`[図表あり]`へ置換
- 既定split：119回をvalidation 400問、120回をtest 400問、その他6,186問をtrain
- `exam_version=700`：44問。予想問題のため、次の変更でchallenge splitへ分離する予定

問題文、選択肢、正解、ポイント、選択肢解説を構造化JSONLと1問1行TXTへ変換した。元の医療データには画像URLやHTMLが含まれるため、標準ライブラリのHTML parserでタグを除き、画像はプレースホルダだけ残した。正解と解説を含むため、医療testは年度単位で学習から分離し、医療能力の評価では未学習年度だけを使用する。

## 実施した処理

会話Importerは会話ファイルを決定的な順序で読み、発話IDでターン順をそろえ、seed 42で会話単位のtrain/validation/testへ分割した。RealPersonaChatは13,583会話・408,619発話、MRMPは960会話・100,680発話で、合計14,543会話・509,299発話、10,521,594文字だった。全体のsplitはtrain 11,635会話、validation 1,454会話、test 1,454会話である。同一会話がsplitを跨がないことをfixtureと実データで確認した。

出力manifestは`artifacts/corpus/conversation-v1/manifest.json`と`artifacts/corpus/medical-qb-v1/manifest.json`に保存した。公開元ファイルのSHA-256、取得コミット、入力件数、出力件数、文字数、split条件、出力SHA-256を記録している。会話本文JSONL/TXTと医療本文JSONL/TXTは大きいためGit管理対象外とし、manifestと取得・変換手順だけをGitへ残す。

## 結果と解釈

会話と医療の両方について、元データを壊さず、元ディレクトリへ書き込まず、後から同じ条件で加工できる入口を作れた。会話データはIssue #1の目的どおり、現代の短文、相槌、砕けた表現、複数ターンの文脈を含むため、一般日本語の補助データとして使う価値がある。医療データは問題文・選択肢・説明がまとまっており、専門語彙の補強に使える。

ただし、これらをそのまま大量に混ぜればよいとは限らない。医療の正解記号や会話の特殊な区切りを覚えただけの変化を、一般的な日本語能力の向上と誤認しないため、まず一般文書を主軸にして、一般80%、会話10%、医療10%を最初の混合比率とする。一般・会話・医療それぞれのvalidation lossと固定prompt生成を分けて見る。

## 次に試すこと

まず医療の700回44問をchallengeへ分けた改訂版を`medical-qb-v2`として作成する。旧`medical-qb-v1`は、challenge分離前の処理結果として上書きせずに残す。実行前の仮説は、予想問題を学習から外しておくことで、医療評価の未学習データとして扱えるようになり、trainへの意図しない混入を防げるというものである。実行コマンドは次のとおりである。

```bash
.venv/bin/python scripts/import_medical_qb.py \
  --input /Users/koseki/projects/medilink_analysis/data/qb.sqlite \
  --output-dir artifacts/corpus/medical-qb-v2
```

実行後は、`challenge`が44問、trainが6,142問となること、元SQLiteのSHA-256が変わらないこと、元ディレクトリに変更がないことを確認する。その後、一般文書・会話・医療のtrain splitだけを決定的に混ぜる。Tokenizerは混合trainだけで学習し、validation・test・challengeはTokenizer学習にも混合にも使わない。5Mモデルのsmoke学習を100万token程度で行い、一般・会話・医療の評価を別々に記録してから、5,000 step級の比較へ進む。

## 追加処理の結果

上記コマンドを実行し、旧`medical-qb-v1`を変更せずに`artifacts/corpus/medical-qb-v2/`へ改訂版を書き出した。manifestの`format`は既存Importerとの互換性を保つ`medical-qb-v1`であるが、出力ディレクトリは版を分けている。入力SQLiteのSHA-256は`5499dff6f181a845b7a087a55b78606869aa5664e0c351c8f8889b719df1ec14`で、以前の記録と一致した。元ディレクトリ`/Users/koseki/projects/medilink_analysis`のGit作業ツリーにも変更はなかった。

改訂後の分割は、train 6,142問、validation 400問、test 400問、challenge 44問である。`challenge.txt`は44行となり、train本文には`試験回：700`が含まれないことを確認した。したがって、700回の予想問題が通常学習へ混入する問題は解消できた。出力manifestには各splitの件数、文字数、JSONL/TXTのSHA-256、入力SHA-256が残っている。次はこの`v2/train.txt`を混合学習の医療sourceに使用する。
