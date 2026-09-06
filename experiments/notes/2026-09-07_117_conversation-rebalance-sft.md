# 実験117：RPCを増やした会話データ再配分SFT

## 学習完了時の記録

2026-09-07、step 4,000まで正常に完走した。step 4,000のvalidation lossは2.728911、validation perplexityは15.316194、learning rateは5.000e-7、経過時間は325.93秒だった。最良checkpointはstep 3,500で、validation lossは2.728217、perplexityは15.305569だった。exp116 bestのvalidation loss 2.747276と比べて0.019059低く、今回の会話データ再配分がvalidation上では明確な改善を示した。

最良重み`best.pt`のSHA-256は`333456ca779477bba5191e1acce5414a9a48b4d7acb1336207f97a8da301ad20`である。学習全体の最大GPUメモリ使用量はallocated 1,490,586,112 bytes、reserved 1,598,029,824 bytesだった。NaN、OOM、shape errorは発生しなかった。250 stepごとのmetrics、生成サンプル17本、step 4,000の最終checkpoint metadata、学習ログを保存した。

固定プロンプト「今日は天気がいいですね。」に対するstep 4,000の出力は、短く「こんばんは!」と返す形だった。日本語として壊れてはいないが、天気への応答や話題の維持までは確認できないため、validation lossの改善だけで自然な会話能力の向上とは判断しない。次に一般会話と医療の固定評価を実施し、exp115・116と比較する。

## 実施前の計画

### 目的

実験116では、同じ一般会話・医療SFTを追加8,000 step行った結果、validation lossと医療の完全一致は改善したが、一般会話F1が低下した。今回は学習stepを増やす方法をいったん止め、一般会話SFTの中身を変更する。短い応答が多いMRMPのresponse token予算を減らし、質問文へ具体的に返す例が比較的多いRPCの予算を増やして、自然な日本語への影響を測定する。

### 仮説

現行の一般会話データはRPC約770k response tokensとMRMP約770k response tokensを含む。MRMPには数Tokenの短い相槌が多数含まれるため、RPCを1.2M tokens、MRMPを0.5M tokensへ再配分すれば、質問への応答、話題の維持、具体的な文の生成が改善すると予想する。一方で、MRMP固有の複数話者の短いやり取りが減るため、一般会話のEOSや短い相槌が悪化する可能性もある。医療データは同じものを使うため、医療正答率は大きく変わらないと予想する。

### 比較条件

exp116 bestを初期checkpointとし、モデル、tokenizer、validation、医療SFT、rehearsalを固定する。変更するのはRPC・MRMPのresponse token配分だけである。

- 初期checkpoint：exp116 best、step 15,750、重みSHA-256 `eaa3b779778be238ba5bbdfaae28bdabceb7e3c996971b5b20d1326c08870406`
- RPC：response token予算1,200,000、quality-aware選別、seed 1171
- MRMP：response token予算500,000、quality-aware選別、seed 1172
- 一般会話合計：response token約1,700,000。現行の約1,541,975 tokensから約10%増加
- 医療：通常医療2,945例とanswer-focus医療2,945例を現行と同じ割合で連結
- validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、ratio 0.2、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 学習：4,000 step、batch size 8、learning rate 5e-6から5e-7、warmup 200、weight decay 0.01、seed 117

### 入力データの出所

RPCとMRMPは、既存の派生会話JSONLだけを読み込む。元データや医師国家試験原本は変更しない。Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`を使う。RPC入力JSONLのSHA-256は`aba75dbbba72b2d1839c11cdc96e36ea5b87e4f3a8351175a1259dc21a3bb610`、MRMPは`93a85f6be0d300980f1c9bcc6cb65845ff7671cd0243390feee6df0a816e9c1e`である。

### 成功・失敗の判定

NaN、OOM、shape errorなく4,000 stepを完走し、250 stepごとのvalidation loss、生成文、checkpoint metadataを保存する。exp116 bestと比べて一般会話F1が改善し、EOS 48/48を維持できれば、会話データ再配分を次の主線へ採用する。F1が改善しても「わけわかりません」のような文脈破綻が増える場合は人手レビューを優先する。医療の完全一致は副指標とし、一般会話データだけの変更で医療性能が変わった場合も記録する。

### データ準備コマンド

```bash
PYTHONPATH=scripts uv run python scripts/prepare_quality_chat_sft.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input rpc=artifacts/corpus/conversation-sft-sources-v1/rpc \
  --output artifacts/sft/issue1-rpc-1200k-quality-v2/train.npz \
  --manifest artifacts/sft/issue1-rpc-1200k-quality-v2/manifest.json \
  --context-length 256 --target-response-tokens 1200000 --seed 1171

PYTHONPATH=scripts uv run python scripts/prepare_quality_chat_sft.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input mrmp=artifacts/corpus/conversation-sft-sources-v1/mrmp \
  --output artifacts/sft/issue1-mrmp-500k-quality-v2/train.npz \
  --manifest artifacts/sft/issue1-mrmp-500k-quality-v2/manifest.json \
  --context-length 256 --target-response-tokens 500000 --seed 1172
```

選別後にRPC・MRMP・通常医療・answer-focus医療を`concat_sft_npz.py`で連結し、入力NPZとmanifestのSHA-256を記録してからRunpodへ転送する。学習開始前にこのノートへ実際の例数とresponse token数を追記する。

### データ選別の結果

quality-aware選別はエラーなく完了した。RPCは72,136例、response 1,200,043 tokens、questionsとして分類された直前発話は34,455例、挨拶だけのresponseは1,035例だった。MRMPは52,758例、response 500,012 tokens、questionsとして分類された直前発話は10,352例、挨拶だけのresponseは1,170例だった。現行のquality-aware-770k-each条件と比べ、RPCを約430k tokens増やし、MRMPを約271k tokens減らす構成になっている。

生成したNPZのSHA-256は、RPCが`0a726889528f30adb19d5beff7cac104f7beb7f75fe1fff56a1a84d51d5822ba`、MRMPが`154be47ff6cd3f1cff1f0fafc14565f6b17b0fbde14c3a2a7f7046efb0049168`である。manifestはそれぞれ`artifacts/sft/issue1-rpc-1200k-quality-v2/manifest.json`と`artifacts/sft/issue1-mrmp-500k-quality-v2/manifest.json`へ保存した。選別処理は既存の`prepare_quality_chat_sft.py`を使用し、元のJSONLへ書き込みは行っていない。

RPCとMRMPを連結した一般会話NPZは124,894例、response 1,700,055 tokens、SHA-256 `41777539e38dc8edec5d6a6bee311dde4a19c0246c3e49766373edb7fcd2bb5a`となった。通常医療とanswer-focus医療を加えた最終SFT NPZは130,784例、response 1,894,122 tokens、SHA-256 `30d4c6de43391fcfedfc46966067d09c0f9f86d41a04aa15060d64214bd09e26`である。最終NPZと連結manifestは`artifacts/sft/issue1-conversation-rebalance-medical-answer-focus-v2/`へ保存した。exp116のSFT trainは133,621例だったため、今回は医療を含む例数は少し減るが、一般会話のresponse tokenは増えている。

## 学習中の記録

データ準備前。学習を開始した場合は、少なくとも1,000 step以内ごとにvalidation loss、learning rate、経過時間、生成文、警告を追記する。失敗した場合も削除せず記録する。

### Runpod転送時の保存先エラー

最終SFT NPZをRunpodへ転送する最初の試行は、`artifacts/sft/issue1-conversation-rebalance-medical-answer-focus-v2/`がRunpod側に存在しなかったため失敗した。ローカルのNPZ、元データ、医師国家試験原本、exp116 checkpointには影響がなく、学習プロセスも起動していない。Runpod側に保存先を作成して同じファイルを再送し、SHA-256を照合してから学習を開始する。

保存先を作成して再送した結果、Runpod上の最終SFT NPZはSHA-256 `30d4c6de43391fcfedfc46966067d09c0f9f86d41a04aa15060d64214bd09e26`、設定は`aa32e24a522e2d8a2fc51346682bae10dd78f5a80ebda136c4bc60d8b69163be`、初期checkpointとして使うexp116 bestは`eaa3b779778be238ba5bbdfaae28bdabceb7e3c996971b5b20d1326c08870406`であることを確認した。学習開始前の入力照合は成功した。

### 2026-09-07：step 1〜250

Runpod A40上でexp116 bestから学習を開始した。step 1のvalidation lossは2.747247、step 250は2.748958、step 250のlearning rateは4.998e-6、経過時間は21.43秒だった。warmup中のためvalidation lossは一時的に上昇しているが、NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 500〜750

step 500のvalidation lossは2.743028、step 750は2.740917だった。warmup終了後にlossは低下し、step 750のlearning rateは4.772e-6、経過時間は62.19秒となった。exp116 bestの2.747276をすでに下回っており、会話データ再配分によるvalidation改善の可能性が見えている。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 1,000〜1,500

step 1,000のvalidation lossは2.742237、step 1,250は2.739406、step 1,500は2.740307だった。step 1,250で現時点の最良値を更新し、exp116 bestから0.007870改善した。step 1,500のlearning rateは3.823e-6、経過時間は122.78秒だった。step 750以降は小さな揺らぎがあるものの、再配分条件のvalidation lossはexp116より低い状態を保っている。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 1,750〜2,250

step 1,750のvalidation lossは2.737971、step 2,000は2.738409、step 2,250は2.736940だった。step 2,250で現時点の最良値を更新し、exp116 bestから0.010336改善した。step 2,250のlearning rateは2.474e-6、経過時間は183.06秒だった。学習率が下がる中でもvalidation lossは改善傾向を保ち、データ再配分の効果が一時的なwarmup後の揺らぎだけではない可能性が高まった。

### 2026-09-07：step 2,500〜3,000

step 2,500のvalidation lossは2.733098、step 2,750は2.731355、step 3,000は2.729865だった。step 3,000で現時点の最良値を更新し、exp116 bestから0.017412改善した。step 3,000のlearning rateは1.227e-6、経過時間は244.29秒だった。validation lossは学習後半でも改善を続けている。NaN、OOM、shape errorは発生していない。

### 2026-09-07：step 3,250〜3,750

step 3,250のvalidation lossは2.729510、step 3,500は2.728217、step 3,750は2.728646だった。step 3,500で現時点の最良値を更新し、exp116 bestから0.019059改善した。step 3,750のlearning rateは5.483e-7、経過時間は305.80秒だった。step 3,750では小さな反発があったが、全体としてexp116を明確に下回る状態を保っている。NaN、OOM、shape errorは発生していない。
