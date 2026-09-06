# 実験091：応答機能を考慮したIssue #1会話SFTデータの選別

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。GitHub Issue #1は、RealPersonaChat（RPC）とMulti-Relational Multi-Party Chat Corpus（MRMP）を一般日本語baselineへ追加し、短文、相づち、砕けた表現、複数ターンの文脈を学ばせる候補として扱っています。086では会話response tokenを増やすと5領域validation lossは改善しましたが、held-out chat F1は低下しました。087では長い応答を25%へ増やすとchat F1の一部は改善したものの、5領域lossが悪化し、固定promptは挨拶へ縮退しました。

今回の仮説は、同じSFT response token予算でも、定型挨拶・初回発話を制限し、質問を含む履歴と非質問の話題継続を一定比率で選ぶ方が、単純な無作為subsetより自然な会話性能に有利だというものです。長文の一律oversamplingは行わず、response 24 token以上の比率は元の分布を保ちます。sourceごとに770,975 response tokenを目標とし、RPCとMRMPの比率は086と同じです。

データ選別器は、前の発話が質問文であるか、targetが定型挨拶だけか、会話の2発話目かを決定的な文字列規則で分類します。質問履歴は可能な範囲でresponse tokenの50%を目標にし、定型挨拶は各sourceの2%、初回発話は5%を上限とします。候補が不足するsourceでは、無理に条件を満たさず、残りの候補から予算を埋めます。強いLLMによる分類・蒸留・生成データは使いません。

成功条件は、086と同じresponse token予算で、held-out chat-testのshort・medium・longの複数層または全体F1を改善し、5領域validation lossを大きく悪化させないことです。生成が長くなっただけ、定型挨拶の種類が変わっただけ、validation lossだけが改善した場合は成功としません。選別前後の件数・token数・カテゴリ比率・入力hashをmanifestへ記録します。

## 再現条件

ここに実験開始時のGit commit、Tokenizer hash、RPC/MRMP train JSONLのhash、選別コマンド、出力NPZとmanifestのhashを追記します。出力は`artifacts/sft/issue1-quality-aware-770k-each-v1/train.npz`とし、validation・評価セットは086と同じものを使います。

## 実験中の記録

ここに候補数、選別結果、不足カテゴリ、エラー、出力検証を追記します。元の会話JSONL、`medilink_analysis`、既存SFTデータは変更・削除しません。

## 実験終了後の結果と解釈

ここにカテゴリ別件数とToken数、全体のresponse token数、manifest・NPZのSHA-256、086との差、次の学習条件を追記します。データ作成だけで学習を実施しなかった場合も、そのことを明記します。

## 次に試すこと

選別結果が妥当なら、同じbase・同じ10,000 step・同じrehearsal条件でSFTし、086・087とpaired comparisonします。結果が悪ければ、規則分類の誤りを修正するのではなく、カテゴリ配分を変えた別実験として記録します。GPUが復旧しない間は、選別器のfixtureテストとデータ分布の監査を進めます。
