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

実行後に追記する。
