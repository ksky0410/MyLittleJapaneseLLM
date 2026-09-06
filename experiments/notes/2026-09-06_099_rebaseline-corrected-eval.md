# 実験099：既存50M・10M事前学習checkpointの修正版再評価

## 目的

実験097でPyTorch validationの評価窓を修正した。実験080以前のdomain評価は、`eval_batches=20`でも20例・3物理バッチしか使わず、最後の短いバッチを他と同じ重みで平均していたため、修正後の値と直接比較できない。実験098の20M・40,000 step条件を判断する前に、既存の080 checkpointを同じ修正版`evaluate_torch.py`で再評価し、比較基準を確定する。

## 事前仮説

1. 160評価窓へ増やすと、旧080のvalidation lossは少し変わるが、既存checkpointの順位や大まかな傾向は保たれる。
2. FineWeb testとWikipedia validationを含めて再評価することで、旧080ノートのgeneral validationだけでは分からなかった現代日本語・百科事典日本語への適合を比較基準へ追加できる。
3. checkpointの重みや生成コードは変更していないため、chat-testの生成結果は旧評価と同じになる。domain lossのみ、評価窓修正の影響を受ける。

## 条件

- checkpoint：`artifacts/checkpoints/issue1-both-50m-pretrain-10m-5k/best.pt`、実験080のbest step 4,900
- config：`configs/issue1-both-50m-pretrain-10m-5k.toml`
- backend：PyTorch CPU、AMPなし
- 評価窓：20 batches、batch size 8、修正後は最大160窓
- domain：従来general、FineWeb test、Wikipedia validation、conversation、medical、RPC、MRMP
- chat：`experiments/evaluation/chat-test-v1.json`の48例、seed 42、最大64 Token

## 実行予定コマンド

```bash
PYTHONPATH=scripts uv run python scripts/evaluate_torch.py domains \
  --config configs/issue1-both-50m-pretrain-10m-5k.toml \
  --checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-10m-5k/best.pt \
  --device cpu --eval-batches 20 \
  --domain general=artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin \
  --domain fineweb=artifacts/tokens/fineweb2-edu-japanese-v1-test.bin \
  --domain wikipedia=artifacts/tokens/wikimedia-wikipedia-ja-validation-v1.bin \
  --domain conversation=artifacts/tokens/mixed-ja-80-10-10-v2-conversation-val.bin \
  --domain medical=artifacts/tokens/mixed-ja-80-10-10-v2-medical-val.bin \
  --domain RPC=artifacts/tokens/issue1-real-persona-chat-validation.bin \
  --domain MRMP=artifacts/tokens/issue1-mrmp-validation.bin \
  --output artifacts/evaluations/issue1-both-50m-pretrain-10m-5k-domains-corrected-v2.json

PYTHONPATH=scripts uv run python scripts/evaluate_torch.py chat \
  --config configs/issue1-both-50m-pretrain-10m-5k.toml \
  --checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-10m-5k/best.pt \
  --selection experiments/evaluation/chat-test-v1.json \
  --input artifacts/corpus/conversation-v1/test.jsonl \
  --device cpu --max-new-tokens 64 --seed 42 \
  --output artifacts/evaluations/issue1-both-50m-pretrain-10m-5k-chat-test-corrected-v2.json \
  --text-output artifacts/evaluations/issue1-both-50m-pretrain-10m-5k-chat-test-corrected-v2.txt
```

## 成功判定

domain JSONとchat JSON/TXTが生成され、各入力・checkpointのSHA-256と評価条件を記録できれば実行面の成功とする。旧domain JSONと異なる値になっても、これは再評価条件が変わったためであり、モデル改善や悪化とは解釈しない。生成本文は全文保存する。

## 結果

実験099は完了した。修正済み評価コードのdomain再評価は約68秒、chat-test再評価は約48秒で完了し、OOMや入力不一致はなかった。checkpointは変更せず、生成結果は旧080と同じになった。

domain評価では、general 4.636483（PPL 103.181）、FineWeb test 3.597812（PPL 36.518）、Wikipedia validation 3.618478（PPL 37.281）、conversation 2.558181（PPL 12.912）、medical 2.713700（PPL 15.085）、RPC 2.669679（PPL 14.435）、MRMP 2.350337（PPL 10.489）となった。domain JSONは`artifacts/evaluations/issue1-both-50m-pretrain-10m-5k-domains-corrected-v2.json`、SHA-256は`52ad8cd9929481327822f68d17a69ba621e6a016fec1466c38f5adf108c50094`である。

chat-test 48例はEOS 48/48、平均生成長8.875 Token、token overlap F1 0.074119だった。shortは0.081483、mediumは0.067581、longは0.073293である。chat JSONは`artifacts/evaluations/issue1-both-50m-pretrain-10m-5k-chat-test-corrected-v2.json`、SHA-256は`fa6d6c2c99d77d90073175324a538897ee8ba7e3ba837fd1b5318dcc35df87e7`、全文TXTのSHA-256は`2f56602d25bf8977fd1131c0a254dad084f5c0ad1deea9b29ee4169d5087eded`である。

旧080ノートのdomain値は旧評価窓の影響を受けているため、今後の比較基準にはこの099の修正版値を使う。chat生成は同じcheckpoint・同じseedで再現し、評価コード修正は生成結果へ影響しなかった。080 checkpointは、validation loss 4.6365の現代FineWeb test適合を持つ一方、chat-test F1は0.0741にとどまり、自然な日本語生成の根拠にはならない。

## 解釈と次の一手

仮説1と3は支持された。domain値は評価窓を増やしても大きくは崩れず、chat本文も一致した。一方、仮説2の「自然さ改善」は080から確認できない。実験098では、20M Token列と40,000 stepを主線として実行し、FineWeb test 3.597812、Wikipedia 3.618478、conversation 2.558181、medical 2.713700、RPC 2.669679、MRMP 2.350337、chat F1 0.074119を比較基準にする。特に、validation lossの低下と会話生成の自然さを別々に判定する。
