# 実験051：20M会話SFTとrehearsalの比較

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験050で、Issue #1の会話sourceを通常のnext-token pretrainingへ混ぜた`both`条件が、20MでもRPC・MRMP・conversation validationを改善することを確認しました。しかし、会話履歴から応答を返す目的に対して、全Tokenへ一様にlossをかける方法と、応答部分だけへlossをかけるSFTの差はまだ未検証です。051では、会話JSONLから作成済みのresponse-only maskを使い、SFT-onlyとrehearsal併用を比較します。

まず5Mの短いsmokeでPyTorch版SFTのforward、mask、checkpoint reload、生成保存を検証します。その後、同じ20Mの`both` best checkpointを初期値にして、SFT-onlyとrehearsal ratio 0.25を比較する計画です。PyTorch版を追加する理由は、実験050のColab T4で得た`.pt` checkpointをMLXの`.npz` loaderへ変換せず、そのまま再現可能にするためです。

比較条件は次の二つです。SFT-onlyは会話応答Tokenと直後のEOSだけへmasked cross entropyをかけます。rehearsalは同じSFT batchに通常の`mixed-ja-80-10-10-v2-train.bin`から作るfull causal LM batchを加え、SFT loss 0.75とrehearsal loss 0.25を結合します。初期値、Tokenizer、モデル、batch size、optimizer、learning rate、seed、step数は揃え、差分をloss objectiveに限定します。

仮説は、SFT-onlyの方が会話応答のvalidation lossと固定chatのEOS・文脈適合を改善する一方、general・medical lossを悪化させ、rehearsal 0.25がその忘却を抑えるというものです。response-onlyのloss対象Token数とrehearsalに使った全Token数は別々に記録します。固定chatのoverlapだけで自然さを断定せず、生成TXTを全step保存します。

## データとモデル

会話SFTデータは`artifacts/sft/chat-v1-context256/train.npz`と`validation.npz`で、manifestは`artifacts/sft/chat-v1-context256.manifest.json`です。trainは396,966例、response対象5,903,046 Token、validationは49,045例、response対象738,660 Tokenです。train NPZのSHA-256は`400b8ffbc5b3752eaa16e003dab168c75e0a77046ac61c39630ef2409a73e609`、validation NPZは`5f52b3f4269e914184834d6e13d800604827abfd96f2b4c1ff5f665cd3f8f7b4`です。元JSONL、元の医師国家試験データ、`/Users/koseki/projects/medilink_analysis`は変更しません。

20Mの初期checkpointは実験050の`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`で、SHA-256 `326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`、best step 1,700、19,308,032 parametersです。5M smokeの初期checkpointは実験049の`artifacts/checkpoints/issue1-both-5m-smoke/best.pt`で、SHA-256 `ac07c9a835bea9b3f6e94322c621c7958cd86fe12dbed4384f7118b647376865`です。Tokenizerはvocab 4,096の`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。

5M smoke設定は`configs/issue1-both-5m-sft-torch-smoke.toml`、20M設定は`configs/issue1-both-20m-sft-torch-colab-1k.toml`です。実験開始時点の基準commitは実験050成果物をpushした`a4ee1ff`です。PyTorch版SFTスクリプト`scripts/train_sft_torch.py`と単体テスト`tests/test_train_sft_torch.py`を追加し、SHA-256はそれぞれ`c19b73d85972a0ab877c445ebf6a8c7315376e7be261b463737358a2538dbf6b`、`41ed2a6528dadce5c5c2caf2c0cb3a0bdd5eabe4b761e46ff0ed9c4cbd87fcd7`です。追加テストは5件、全体テストは74件が通過しました。実行コードを確定したcommitは`2eeaee9`です。

## 実行コマンド

```bash
.venv/bin/python scripts/prepare_chat_sft.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input artifacts/corpus/conversation-v1 \
  --output artifacts/sft/chat-v1-context256 \
  --manifest artifacts/sft/chat-v1-context256.manifest.json \
  --context-length 256 --seed 42

.venv/bin/python scripts/train_sft_torch.py \
  --config configs/issue1-both-5m-sft-torch-smoke.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-5m-smoke/best.pt \
  --train-data artifacts/sft/chat-v1-context256/train.npz \
  --validation-data artifacts/sft/chat-v1-context256/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-5m-sft-torch-smoke \
  --samples-dir artifacts/samples/issue1-both-5m-sft-torch-smoke \
  --device cpu

.venv/bin/python scripts/train_sft_torch.py \
  --config configs/issue1-both-5m-sft-torch-smoke.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-5m-smoke/best.pt \
  --train-data artifacts/sft/chat-v1-context256/train.npz \
  --validation-data artifacts/sft/chat-v1-context256/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-5m-rehearsal-torch-smoke \
  --samples-dir artifacts/samples/issue1-both-5m-rehearsal-torch-smoke \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.25 --device cpu
```

20M本学習は5M smokeと同じ引数でconfig、base checkpoint、出力先だけを20M用へ置き換え、Colab T4で実行します。SFT-onlyとrehearsalの両条件を同じVM・同じseedで順番に走らせるか、割当が取れない場合は5M smokeの結果を先に確定します。

## 成功条件

5M smokeがmask対象Tokenのないbatch、NaN、shape error、checkpoint reloadエラーなく完走し、SFT-onlyとrehearsalのmetrics、summary、生成TXTを保存することです。20M比較では、同じ初期checkpointから両条件が完走し、general・medical・conversation・RPC・MRMPのdomain評価と48例の固定chat-testを実施することです。SFT loss対象Token、rehearsal Token、実際のoptimizer stepを分けて記録します。PyTorch版の不具合やColab割当失敗は成功結果へ混ぜず、原因と次の対策をこのノートへ残します。

## 実験中の記録

開始前、コード実装、smoke開始・途中・終了、Colab割当、20M各条件の開始・途中・終了、回収、評価をこのノートへ追記します。学習中の生成文は品質に関係なく全step保存し、短い応答、空出力、特殊Token混入も削除しません。

14:32 JST、5M smokeをSFT-onlyとrehearsal ratio 0.25で並列実行し、両条件が200 stepまで完走しました。実測parameter数は各5,205,120、PyTorch 2.14.0 CPU、AMP無効です。SFT validation lossはSFT-onlyがstep 1で5.890153、step 50で5.364355、step 100で5.208715、step 150で5.154748、step 200で5.132536となり、best stepは200、PPLは169.446でした。rehearsalはstep 1で5.890440、step 50で5.374157、step 100で5.227571、step 150で5.182860、step 200で5.163926となり、best stepは200、PPLは174.850でした。SFT-onlyがvalidation lossでは0.031391低いものの、これは会話応答validationへの適合を測る短いsmokeの結果であり、通常domainの忘却を含めた優劣ではありません。

SFT-onlyの入力hashはconfig `0155c190c2c99a602f6083a63d4953c83beedae7f0521de997da968e1d3c46e6`、train NPZ `400b8ffbc5b3752eaa16e003dab168c75e0a77046ac61c39630ef2409a73e609`、validation NPZ `5f52b3f4269e914184834d6e13d800604827abfd96f2b4c1ff5f665cd3f8f7b4`、base checkpoint `ac07c9a835bea9b3f6e94322c621c7958cd86fe12dbed4384f7118b647376865`です。rehearsalはこれらに加えてrehearsal Token列 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`を使用しました。出力は`artifacts/checkpoints/issue1-both-5m-sft-torch-smoke/`、`artifacts/checkpoints/issue1-both-5m-rehearsal-torch-smoke/`、対応する`artifacts/samples/`へ保存しています。次に両best checkpointをreloadしてdomainと固定chatを評価します。

14:35 JST、両best checkpointをローカルCPUでreloadし、general、conversation、medical、RPC、MRMPの5領域を20 batchずつ評価しました。SFT-onlyはgeneral 6.7647（PPL 866.68）、conversation 4.5359（PPL 93.31）、medical 5.7838（PPL 325.00）、RPC 4.5439（PPL 94.05）、MRMP 4.1275（PPL 62.02）でした。rehearsalはgeneral 6.6009（PPL 735.76）、conversation 4.3145（PPL 74.78）、medical 5.6680（PPL 289.45）、RPC 4.3129（PPL 74.66）、MRMP 3.9734（PPL 53.17）でした。今回のdomain評価ではrehearsalが5領域すべてで低いlossとなり、SFT-onlyより一般・医療の忘却を抑えながら、会話領域も改善しました。

同じ48例の固定chat-testでは、SFT-onlyがEOS 48/48、平均生成長8.90 Token、Token overlap F1 0.1881、rehearsalがEOS 48/48、平均11.06 Token、F1 0.1681でした。SFT-onlyはshort・medium・longのF1がそれぞれ0.2925・0.1604・0.1115、rehearsalは0.2243・0.1498・0.1303でした。overlapだけではSFT-onlyが高く、domain lossと逆の結果でした。生成本文には、SFT-onlyの「私もし、「コレスキリースマーが」や「お??」、rehearsalの「それはに、今後していがらもけます?」や「おさでも、、そが、最おりかり。」のような文脈不一致・崩れた応答が残っています。自然さの判定は未実施のため、レビュー用JSONを`experiments/evaluation/issue1-both-5m-sft-torch-smoke-chat-review.json`と`issue1-both-5m-rehearsal-torch-smoke-chat-review.json`へ保存し、人手ラベルは空欄のままにしました。

5M smokeの範囲では、rehearsal ratio 0.25は通常domain lossを改善し、会話validation lossもSFT-onlyより低くなりました。一方、固定chatのToken overlap F1はSFT-onlyが高く、指標間の不一致が確認されました。現在の差は200 stepの短いCPU探索であり、SFT-onlyが自然な会話に優れる、またはrehearsalが常に優れるとは結論しません。20Mで同じ初期値・低い学習率・1,000 stepを比較し、応答mask、通常domain、実際の生成、人手レビューを併せて再検証します。

## 結果と解釈

最終および最良checkpoint、SFT validation loss、通常domain loss、固定chatのEOS・生成長・overlap、生成本文の人手観察を条件別に記録します。5M smokeの実測結果は上の実験中記録へ追記し、domain評価JSON、固定chatのJSON/TXT、レビュー用JSON、学習中の全生成TXT、metrics、checkpoint metadataをGitの追跡対象にします。SFT validationの改善は応答形式への適合を示すだけで、会話の事実性や医学的正確性を示さないことを明記します。

## 次に試すこと

rehearsalが忘却を抑えた場合は、ratio 0.10、0.25、0.50の比較へ進みます。SFT-onlyが十分に改善しない場合は、短い応答の層化sampling、会話source別SFT、または教師LLMによる日本語instruction dataの作成を検討します。20Mで安定した後に、より大きなモデルやLoRA/QLoRAとの比較へ進みます。
