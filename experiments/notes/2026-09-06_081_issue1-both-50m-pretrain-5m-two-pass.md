# 実験081：50M日本語モデルの5M Token二周相当事前学習

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。080では約10M Tokenの新しい混合コーパスを5,000 step学習し、075の約5M Token・2,500 stepより5領域validation lossと固定chat F1が改善しました。ただし080はデータsource比率も変わっていたため、Tokenの多様性を増やした効果と、同じデータを繰り返し学習した効果を分離できていません。

081では、075と同じ約5M Token列を使い、080と同じ5,000 step・同じ50M構造・同じseed・同じlearning-rate scheduleで学習します。batch size 8、context length 256なので、学習で見るToken数は約10.24Mとなり、同じ5M Token列を概ね二周見せる条件です。080の「10M unique tokensを一周」と081の「5M tokensを二周」を比較し、データの多様性と反復回数のどちらがvalidation性能と自然な日本語に効くかを調べます。強い教師モデルの蒸留は行いません。

仮説は、5M列を二周すると、075で得られた一般・会話・医療データを繰り返し最適化できるため、080と同程度のvalidation lossになる一方、重複学習による過学習で固定chat-testの自然さや生成多様性が伸びにくい、です。もし081が080を上回る場合は、限られたMacBook規模ではデータを増やす前に高品質データを複数周回する方が有効な可能性があります。逆に080が上回る場合は、重複を増やさずデータを広げる方を主線へ採用します。

成功条件は、5,000 stepを完走し、100 step間隔のmetricsと生成文、500 step間隔のcheckpoint metadata、summary、5領域評価、固定chat-test 48例を保存することです。良い出力だけでなく、空応答、文字崩れ、繰り返し、文脈無視も含めてGitHubへ保存します。

## 再現条件

モデルは080と同じRoPE・LayerNorm・SwiGLU、dim 576、12層、9 heads、context length 256、MLP倍率4、50,207,616 parametersです。Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab 4,096、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-5m-v1-train.bin`、4,999,958 Token、SHA-256は`54eb3fab617c94bda59899db4f78e6ac65665606219414a710e40cc8ccb8603c`です。general validationは`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。

設定ファイルは`configs/issue1-both-50m-pretrain-5m-5k.toml`、seed 42、batch size 8、最大5,000 step、eval/sample interval 100、checkpoint interval 500、eval batches 20、learning rate 3e-4から3e-5、warmup 500、weight decay 0.1です。設定ファイルのSHA-256は学習開始前に計算して追記します。元の`medilink_analysis`と医師国家試験原データは変更せず、small_llm側の既存加工済みToken列を読み取り専用で使います。

開始前に確認したSHA-256は、設定ファイルが`db5604539e1da48122f160c55cffa5eb1b46d690738f038ceab0ef2696e8f32b`です。新規Colabスクリプトは`py_compile`を通過しました。

ローカルでの再現コマンドは次のとおりです。

```bash
uv run python scripts/train_torch.py \
  --config configs/issue1-both-50m-pretrain-5m-5k.toml \
  --device mps
```

Colab T4が利用できる場合は、`scripts/colab_bootstrap_081.py`から同じ設定を実行し、終了後に`colab_package_081.py`で軽量成果物とbest checkpointを回収します。Colab失敗時は同じ条件のMPSへ切り替え、失敗ログも削除しません。

開始時点で`colab sessions`は`No active sessions found on server.`でした。Colab送信用bundleは`/tmp/small_llm-colab-081.tar.gz`、約6.7MB、SHA-256は`4565b8aaf5d32635d528aff631d7bffb6cba320acff98ec8df879e78ce89f0c0`です。元JSONL、医師国家試験原本、`medilink_analysis`はbundleに含めていません。

## 実験中の記録

開始前にGit commit、設定hash、入力hash、bundle hash、Colab session状態を記録します。学習中は100 step間隔のmetricsと生成文を保存し、1,000 step以上ノート更新を空けません。途中停止、OOM、warning、空応答もそのまま記録します。

Colab T4の新規session `exp081-both-50m-pretrain-5m-two-pass`は作成に成功し、bundle uploadと入力hash検証が完了しました。step 1はtrain loss 8.895750、validation loss 8.819675、PPL 6766.06、learning rate 6.0e-7、経過2.71秒でした。step 100はtrain loss 6.931328、validation loss 7.205592、PPL 1346.94、learning rate 6.0e-5、経過11.96秒、step 200はvalidation loss 6.941548、PPL 1034.37、learning rate 1.2e-4、経過22.05秒でした。step 300はvalidation loss 6.646078、PPL 769.76、learning rate 1.8e-4、経過33.20秒、step 400はtrain loss 5.919453、validation loss 6.347234、PPL 570.91、learning rate 2.4e-4、経過44.40秒でした。NaN、OOM、shape errorは発生しておらず、学習は継続中です。

step 500はtrain loss 5.476576、validation loss 6.112171、PPL 451.32、learning rate 3.0e-4、経過53.85秒でした。step 600はvalidation loss 5.843829、PPL 345.10、learning rate 2.9968e-4、経過68.77秒、step 700は5.709621、PPL 301.76、learning rate 2.9870e-4、経過80.15秒、step 800はtrain loss 4.542830、validation loss 5.551695、PPL 257.67、learning rate 2.9707e-4、経過90.00秒でした。step 900はvalidation loss 5.450522、PPL 232.88、learning rate 2.9480e-4、経過102.09秒、step 1,000はtrain loss 4.442537、validation loss 5.368098、PPL 214.45、learning rate 2.9189e-4、経過111.89秒、step 1,100はtrain loss 3.985813、validation loss 5.303004、PPL 200.94、learning rate 2.8837e-4、経過122.76秒でした。step 1,100時点でもvalidation lossは改善しており、学習は継続中です。

## 実験終了後の結果と解釈

080と同じCPU・同じ5領域・同じ48例のchat-test・同じgeneration seed 42で評価し、validation loss、EOS、平均生成長、全体・長さ別F1、生成の自然さを比較します。結果が改善しなくても、重複学習の過学習やsource別の弱点を次の実験へ引き継ぎます。

## 次に試すこと

081で080に近いか上回る結果が出た場合は、5M列をさらに高品質化し、会話・医療・一般文書の比率を一つずつ調整します。080が明確に上回る場合は、FineWeb2・Wikipediaなどの追加データを増やし、10Mから20M、さらにcontext length 512へ進みます。どちらの場合も、蒸留は主線へ入れず、フルスクラッチのデータ量・反復回数・自然な日本語生成の関係を優先して研究します。
