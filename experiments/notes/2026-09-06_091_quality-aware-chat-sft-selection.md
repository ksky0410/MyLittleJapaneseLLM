# 実験091：応答機能を考慮したIssue #1会話SFTデータの選別

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。GitHub Issue #1は、RealPersonaChat（RPC）とMulti-Relational Multi-Party Chat Corpus（MRMP）を一般日本語baselineへ追加し、短文、相づち、砕けた表現、複数ターンの文脈を学ばせる候補として扱っています。086では会話response tokenを増やすと5領域validation lossは改善しましたが、held-out chat F1は低下しました。087では長い応答を25%へ増やすとchat F1の一部は改善したものの、5領域lossが悪化し、固定promptは挨拶へ縮退しました。

今回の仮説は、同じSFT response token予算でも、定型挨拶・初回発話を制限し、質問を含む履歴と非質問の話題継続を一定比率で選ぶ方が、単純な無作為subsetより自然な会話性能に有利だというものです。長文の一律oversamplingは行わず、response 24 token以上の比率は元の分布を保ちます。sourceごとに770,975 response tokenを目標とし、RPCとMRMPの比率は086と同じです。

データ選別器は、前の発話が質問文であるか、targetが定型挨拶だけか、会話の2発話目かを決定的な文字列規則で分類します。質問履歴は可能な範囲でresponse tokenの50%を目標にし、定型挨拶は各sourceの2%、初回発話は5%を上限とします。候補が不足するsourceでは、無理に条件を満たさず、残りの候補から予算を埋めます。強いLLMによる分類・蒸留・生成データは使いません。

成功条件は、086と同じresponse token予算で、held-out chat-testのshort・medium・longの複数層または全体F1を改善し、5領域validation lossを大きく悪化させないことです。生成が長くなっただけ、定型挨拶の種類が変わっただけ、validation lossだけが改善した場合は成功としません。選別前後の件数・token数・カテゴリ比率・入力hashをmanifestへ記録します。

## 再現条件

実験開始時点のGit commitは`559231e`です。選別器`f67b317eae1e5b1715a17d136d71d29f61425aa8d29a0fed4f52b76f1b2675c6`、Tokenizer`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`、RPC train JSONL`aba75dbbba72b2d1839c11cdc96e36ea5b87e4f3a8351175a1259dc21a3bb610`、MRMP train JSONL`93a85f6be0d300980f1c9bcc6cb65845ff7671cd0243390feee6df0a816e9c1e`です。各sourceのtarget response token予算は770,975、context lengthは256、seedは9101、質問履歴の目標比率は50%、定型挨拶の上限は2%、初回発話の上限は5%です。出力は`artifacts/sft/issue1-quality-aware-770k-each-v1/train.npz`とし、validation・評価セットは086と同じものを使います。

実行コマンドは次のとおりです。

```bash
uv run python scripts/prepare_quality_chat_sft.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input rpc=artifacts/corpus/conversation-sft-sources-v1/rpc \
  --input mrmp=artifacts/corpus/conversation-sft-sources-v1/mrmp \
  --output artifacts/sft/issue1-quality-aware-770k-each-v1/train.npz \
  --manifest artifacts/sft/issue1-quality-aware-770k-each-v1/manifest.json \
  --context-length 256 \
  --target-response-tokens 770975 \
  --seed 9101 \
  --question-token-fraction 0.5 \
  --max-greeting-token-fraction 0.02 \
  --max-first-turn-token-fraction 0.05
```

## 実験中の記録

2026-09-06にデータ選別を実行しました。RPCは全315,584例・5,132,071 response tokenから、46,349例・771,000 response tokenを選びました。その内訳は、質問履歴が22,153例・385,506 token、定型挨拶が668例・2,652 token、初回発話が1,092例・7,124 tokenです。質問履歴は目標の50%にほぼ一致し、定型挨拶と初回発話は指定した上限内に収まりました。

MRMPは全81,382例・770,975 response tokenで、候補全体が目標token数と同じだったため、全例を保持しました。内訳は、質問履歴が10,352例・100,921 token、定型挨拶が1,934例・6,733 token、初回発話が784例・2,781 tokenです。したがって、MRMPについては候補不足のため、定型挨拶2%・初回発話5%の上限を適用できていません。この制約を隠さず、manifestにも記録しています。

全体では127,731例・1,541,975 response token、質問履歴486,427 token（31.55%）、定型挨拶9,385 token（0.61%）、初回発話9,905 token（0.64%）となりました。出力配列のshapeは`[127731, 256]`で、`input_ids`、`target_ids`、`loss_mask`を検証しました。NPZのSHA-256は`fc2a9367a16c0cf963fbc2e96530a9944d41e4d2d8e4029874dface2d71b5d57`、manifestのSHA-256は`e08e50b038524b72cf5656bcd607909e563e437b2c12454683352fc135758346`です。

元のRPC・MRMP JSONL、Tokenizer、`medilink_analysis`内のデータは変更していません。今回の処理はデータ選別だけで、モデル学習・生成・評価はまだ実施していません。

## 実験終了後の結果と解釈

今回のデータ作成はエラーなく完了しました。RPCでは質問履歴を約半分まで増やしながら、定型挨拶と初回発話を抑えられました。これは、086・087で観測した固定promptの挨拶への縮退が、単なるresponse token数ではなく、会話の機能分布にも関係するという仮説を検証するための入力条件になります。一方、MRMPは全候補を使い切っているため、sourceごとの制約差が残ります。学習結果を解釈するときは、RPCだけに効いた選別効果と、MRMP由来のデータ差を分けて考える必要があります。

086・087との性能差は、まだ学習していないため未計測です。次の092では、086・087と同じ50M base、同じvalidation・評価セット、同じ10,000 stepと比較可能な学習条件を使います。これにより、データ総量や計算量ではなく、今回の会話機能の選別がshort・medium・long応答と固定promptの自然さに与える影響を確認します。

## 次に試すこと

次は092として、今回のデータで同じbase・同じ10,000 step・同じrehearsal条件のSFTを行い、086・087とpaired comparisonします。結果が悪ければ、規則分類の誤りを修正するのではなく、カテゴリ配分を変えた別実験として記録します。GPUが復旧しない間は、選別器のfixtureテストとデータ分布の監査を進めます。
