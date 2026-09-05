# 実験051：20M会話SFTとrehearsalの比較

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験050で、Issue #1の会話sourceを通常のnext-token pretrainingへ混ぜた`both`条件が、20MでもRPC・MRMP・conversation validationを改善することを確認しました。しかし、会話履歴から応答を返す目的に対して、全Tokenへ一様にlossをかける方法と、応答部分だけへlossをかけるSFTの差はまだ未検証です。051では、会話JSONLから作成済みのresponse-only maskを使い、SFT-onlyとrehearsal併用を比較します。

まず5Mの短いsmokeでPyTorch版SFTのforward、mask、checkpoint reload、生成保存を検証します。その後、同じ20Mの`both` best checkpointを初期値にして、SFT-onlyとrehearsal ratio 0.25を比較する計画です。PyTorch版を追加する理由は、実験050のColab T4で得た`.pt` checkpointをMLXの`.npz` loaderへ変換せず、そのまま再現可能にするためです。

比較条件は次の二つです。SFT-onlyは会話応答Tokenと直後のEOSだけへmasked cross entropyをかけます。rehearsalは同じSFT batchに通常の`mixed-ja-80-10-10-v2-train.bin`から作るfull causal LM batchを加え、SFT loss 0.75とrehearsal loss 0.25を結合します。初期値、Tokenizer、モデル、batch size、optimizer、learning rate、seed、step数は揃え、差分をloss objectiveに限定します。

仮説は、SFT-onlyの方が会話応答のvalidation lossと固定chatのEOS・文脈適合を改善する一方、general・medical lossを悪化させ、rehearsal 0.25がその忘却を抑えるというものです。response-onlyのloss対象Token数とrehearsalに使った全Token数は別々に記録します。固定chatのoverlapだけで自然さを断定せず、生成TXTを全step保存します。

## データとモデル

会話SFTデータは`artifacts/sft/chat-v1-context256/train.npz`と`validation.npz`で、manifestは`artifacts/sft/chat-v1-context256.manifest.json`です。trainは396,966例、response対象5,903,046 Token、validationは49,045例、response対象738,660 Tokenです。train NPZのSHA-256は`400b8ffbc5b3752eaa16e003dab168c75e0a77046ac61c39630ef2409a73e609`、validation NPZは`5f52b3f4269e914184834d6e13d800604827abfd96f2b4c1ff5f665cd3f8f7b4`です。元JSONL、元の医師国家試験データ、`/Users/koseki/projects/medilink_analysis`は変更しません。

20Mの初期checkpointは実験050の`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`で、SHA-256 `326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`、best step 1,700、19,308,032 parametersです。5M smokeの初期checkpointは実験049の`artifacts/checkpoints/issue1-both-5m-smoke/best.pt`で、SHA-256 `ac07c9a835bea9b3f6e94322c621c7958cd86fe12dbed4384f7118b647376865`です。Tokenizerはvocab 4,096の`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。rehearsal Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。

5M smoke設定は`configs/issue1-both-5m-sft-torch-smoke.toml`、20M設定は`configs/issue1-both-20m-sft-torch-colab-1k.toml`です。実験開始時点の基準commitは実験050成果物をpushした`a4ee1ff`です。PyTorch版SFTスクリプト`scripts/train_sft_torch.py`と単体テスト`tests/test_train_sft_torch.py`を追加し、確定版のSHA-256はそれぞれ`a10826ace585be3b719ed43f4a4675e70f591c51bb72ed5c9e09e7621e1d31ba`、`41ed2a6528dadce5c5c2caf2c0cb3a0bdd5eabe4b761e46ff0ed9c4cbd87fcd7`です。追加テストは5件、全体テストは74件が通過しました。実行コードを確定したcommitは`2eeaee9`です。

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

20M Colab用wrapperは`scripts/colab_bootstrap_051.py`、軽量成果物の回収用archive作成は`scripts/colab_package_051.py`、分割bundleの連結・hash検証は`scripts/colab_concat_051.py`です。wrapperは`/content/small_llm_051`へ展開し、実験050のbase checkpointを上書きせず、SFT-onlyとrehearsalの出力先を分けます。20M configのSHA-256は`e2b01afa98a28ea7b863a7b1ffe02e86088cd73009fc3e5422a241fd2c3a177b`、wrapperは`ace9c0c7de89c6b3891e96158ad806b9e75bbc53cad97e49bb5dbb1ca9be560e`、package scriptは`52250b1c5e7b1dbfd4ab4c14cd4ba4a8bac4be0c253102927eac45d07b4daff3`、concat scriptは`70629ac7475e6905a49a69354d7761a042d030d67cfb3504e7cdc96b45f255b9`です。予定コマンドは次のとおりです。

```bash
tar -czf /tmp/small_llm-colab-051.tar.gz \
  configs/issue1-both-20m-sft-torch-colab-1k.toml \
  scripts/train_sft_torch.py scripts/train_torch.py scripts/_common.py \
  src artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.json \
  artifacts/sft/chat-v1-context256/train.npz \
  artifacts/sft/chat-v1-context256/validation.npz \
  artifacts/tokens/mixed-ja-80-10-10-v2-train.bin

colab new --session exp051-20m-sft --gpu T4
split -b 45m -d -a 2 /tmp/small_llm-colab-051-c4076e0.tar.gz /tmp/exp051_bundle_part_
colab upload --session exp051-20m-sft /tmp/exp051_bundle_part_00 /content/exp051_bundle_part_00
colab upload --session exp051-20m-sft /tmp/exp051_bundle_part_01 /content/exp051_bundle_part_01
colab upload --session exp051-20m-sft /tmp/exp051_bundle_part_02 /content/exp051_bundle_part_02
colab upload --session exp051-20m-sft /tmp/exp051_bundle_part_03 /content/exp051_bundle_part_03
colab upload --session exp051-20m-sft /tmp/exp051_bundle_part_04 /content/exp051_bundle_part_04
colab upload --session exp051-20m-sft /tmp/exp051_bundle_part_05 /content/exp051_bundle_part_05
colab exec --session exp051-20m-sft --timeout 120 --file scripts/colab_concat_051.py
colab exec --session exp051-20m-sft --timeout 1800 --file scripts/colab_bootstrap_051.py
colab exec --session exp051-20m-sft --timeout 120 --file scripts/colab_package_051.py
colab download --session exp051-20m-sft /content/exp051-lightweight.tar.gz /tmp/exp051-lightweight.tar.gz
colab download --session exp051-20m-sft /content/exp051-manifest.json /tmp/exp051-manifest.json
colab download --session exp051-20m-sft /content/small_llm_051/artifacts/checkpoints/issue1-both-20m-sft-torch-colab-1k/best.pt /tmp/issue1-both-20m-sft-torch-colab-1k-best.pt
colab download --session exp051-20m-sft /content/small_llm_051/artifacts/checkpoints/issue1-both-20m-rehearsal-torch-colab-1k/best.pt /tmp/issue1-both-20m-rehearsal-torch-colab-1k-best.pt
colab stop --session exp051-20m-sft
```

## 成功条件

5M smokeがmask対象Tokenのないbatch、NaN、shape error、checkpoint reloadエラーなく完走し、SFT-onlyとrehearsalのmetrics、summary、生成TXTを保存することです。20M比較では、同じ初期checkpointから両条件が完走し、general・medical・conversation・RPC・MRMPのdomain評価と48例の固定chat-testを実施することです。SFT loss対象Token、rehearsal Token、実際のoptimizer stepを分けて記録します。PyTorch版の不具合やColab割当失敗は成功結果へ混ぜず、原因と次の対策をこのノートへ残します。実行前に確定した20M bundleは`/tmp/small_llm-colab-051-c4076e0.tar.gz`、サイズ259MB、SHA-256 `94f28a741d6e3bebf922031ed8feafa1ecf2eaaacba54da8c38ab1e2950cbd35`です。実行用コードcommitは`c4076e0`です。
14:39 JST、新規session `exp051-20m-sft`のT4割当は成功しましたが、259MBのbundleを一括uploadするとColab CLIがHTTP 400 `Bad Request`を返しました。bundleのローカルhashと内容は変更されておらず、学習も開始していません。upload上限またはruntime proxyの制約と考え、45MB単位へ分割してuploadし、Colab側で連結後に元bundleのSHA-256を照合する方式へ切り替えます。

## 実験中の記録

開始前、コード実装、smoke開始・途中・終了、Colab割当、20M各条件の開始・途中・終了、回収、評価をこのノートへ追記します。学習中の生成文は品質に関係なく全step保存し、短い応答、空出力、特殊Token混入も削除しません。

14:32 JST、5M smokeをSFT-onlyとrehearsal ratio 0.25で並列実行し、両条件が200 stepまで完走しました。実測parameter数は各5,205,120、PyTorch 2.14.0 CPU、AMP無効です。SFT validation lossはSFT-onlyがstep 1で5.890153、step 50で5.364355、step 100で5.208715、step 150で5.154748、step 200で5.132536となり、best stepは200、PPLは169.446でした。rehearsalはstep 1で5.890440、step 50で5.374157、step 100で5.227571、step 150で5.182860、step 200で5.163926となり、best stepは200、PPLは174.850でした。SFT-onlyがvalidation lossでは0.031391低いものの、これは会話応答validationへの適合を測る短いsmokeの結果であり、通常domainの忘却を含めた優劣ではありません。

SFT-onlyの入力hashはconfig `0155c190c2c99a602f6083a63d4953c83beedae7f0521de997da968e1d3c46e6`、train NPZ `400b8ffbc5b3752eaa16e003dab168c75e0a77046ac61c39630ef2409a73e609`、validation NPZ `5f52b3f4269e914184834d6e13d800604827abfd96f2b4c1ff5f665cd3f8f7b4`、base checkpoint `ac07c9a835bea9b3f6e94322c621c7958cd86fe12dbed4384f7118b647376865`です。rehearsalはこれらに加えてrehearsal Token列 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`を使用しました。出力は`artifacts/checkpoints/issue1-both-5m-sft-torch-smoke/`、`artifacts/checkpoints/issue1-both-5m-rehearsal-torch-smoke/`、対応する`artifacts/samples/`へ保存しています。次に両best checkpointをreloadしてdomainと固定chatを評価します。

14:35 JST、両best checkpointをローカルCPUでreloadし、general、conversation、medical、RPC、MRMPの5領域を20 batchずつ評価しました。SFT-onlyはgeneral 6.7647（PPL 866.68）、conversation 4.5359（PPL 93.31）、medical 5.7838（PPL 325.00）、RPC 4.5439（PPL 94.05）、MRMP 4.1275（PPL 62.02）でした。rehearsalはgeneral 6.6009（PPL 735.76）、conversation 4.3145（PPL 74.78）、medical 5.6680（PPL 289.45）、RPC 4.3129（PPL 74.66）、MRMP 3.9734（PPL 53.17）でした。今回のdomain評価ではrehearsalが5領域すべてで低いlossとなり、SFT-onlyより一般・医療の忘却を抑えながら、会話領域も改善しました。

同じ48例の固定chat-testでは、SFT-onlyがEOS 48/48、平均生成長8.90 Token、Token overlap F1 0.1881、rehearsalがEOS 48/48、平均11.06 Token、F1 0.1681でした。SFT-onlyはshort・medium・longのF1がそれぞれ0.2925・0.1604・0.1115、rehearsalは0.2243・0.1498・0.1303でした。overlapだけではSFT-onlyが高く、domain lossと逆の結果でした。生成本文には、SFT-onlyの「私もし、「コレスキリースマーが」や「お??」、rehearsalの「それはに、今後していがらもけます?」や「おさでも、、そが、最おりかり。」のような文脈不一致・崩れた応答が残っています。自然さの判定は未実施のため、レビュー用JSONを`experiments/evaluation/issue1-both-5m-sft-torch-smoke-chat-review.json`と`issue1-both-5m-rehearsal-torch-smoke-chat-review.json`へ保存し、人手ラベルは空欄のままにしました。

5M smokeの範囲では、rehearsal ratio 0.25は通常domain lossを改善し、会話validation lossもSFT-onlyより低くなりました。一方、固定chatのToken overlap F1はSFT-onlyが高く、指標間の不一致が確認されました。現在の差は200 stepの短いCPU探索であり、SFT-onlyが自然な会話に優れる、またはrehearsalが常に優れるとは結論しません。20Mで同じ初期値・低い学習率・1,000 stepを比較し、応答mask、通常domain、実際の生成、人手レビューを併せて再検証します。

20M best checkpointの通常domain評価は、SFT-onlyがgeneral 6.2304（PPL 507.98）、conversation 3.3092（PPL 27.36）、medical 3.4373（PPL 31.10）、RPC 3.2868（PPL 26.76）、MRMP 2.8528（PPL 17.34）でした。rehearsalはgeneral 5.6421（PPL 282.05）、conversation 2.9997（PPL 20.08）、medical 3.2562（PPL 25.95）、RPC 2.9630（PPL 19.36）、MRMP 2.5305（PPL 12.56）でした。rehearsalは5領域すべてで低いlossとなり、通常domainの保持と会話Tokenへの適合の両面で有利な結果です。ただし、これらはSFT学習に使ったsourceと同じ形式を含むvalidationであり、未見の会話への自然さや知識の正確性を直接証明しません。

20Mの固定chat-test v1では、SFT-onlyがEOS 48/48、平均生成長11.08 Token、Token overlap F1 0.1878、rehearsalがEOS 48/48、平均生成長11.48 Token、F1 0.1916でした。層別F1はSFT-onlyがshort 0.2248、medium 0.1878、long 0.1508、rehearsalがshort 0.2704、medium 0.1430、long 0.1613でした。rehearsalは全体F1でわずかに高いものの、mediumではSFT-onlyが高く、差も小さいため、これだけで優劣を固定しません。生成本文には両条件とも文脈に近い短い応答と、文脈から外れた応答が混在しています。評価JSON/TXTは`artifacts/evaluations/issue1-both-20m-sft-torch-colab-1k-chat-test-v1.json`・同名TXT、rehearsalは同名ファイルへ保存し、レビュー用JSONは`experiments/evaluation/issue1-both-20m-sft-torch-colab-1k-chat-review.json`と`issue1-both-20m-rehearsal-torch-colab-1k-chat-review.json`へ保存しました。

5Mと20Mを合わせると、rehearsal ratio 0.25は通常domain lossを安定して押し下げ、20Mでは固定chat F1もわずかに押し上げました。一方、5MではSFT-onlyの固定chat F1が高く、同じ指標が常に同じ結論を返すわけではありません。今回の目的はSFTとrehearsalの挙動を観察することであり、200〜1,000 stepの小規模比較から最適なratioを断定しません。次はrehearsal ratio 0.10、0.25、0.50を同じ20M初期checkpointで比較し、通常domain loss・SFT loss・固定chat・人手レビューを併記します。

20M比較は新規T4 session `exp051-20m-sft`で実行しました。分割uploadした6個のpartを連結し、bundle 271,720,679 bytesのSHA-256が予定値 `94f28a741d6e3bebf922031ed8feafa1ecf2eaaacba54da8c38ab1e2950cbd35`と一致してから学習を開始しました。SFT-onlyとrehearsal ratio 0.25はともにstep 1,000まで完走し、NaN、OOM、shape error、mask対象不足、checkpoint reloadエラーはありませんでした。実測parameter数は各19,308,032、PyTorch 2.11.0+cu128、CUDA 12.8、Tesla T4、AMP有効です。SFT-onlyのpeak allocatedは787,474,432 bytes、reservedは832,569,344 bytes、経過58.61秒、best step 1,000、SFT validation loss 3.8992346168、PPL 49.3647でした。rehearsalのpeak allocatedは737,864,192 bytes、reservedは803,209,216 bytes、経過85.30秒、best step 1,000、SFT validation loss 3.9057643771、PPL 49.6880でした。Colab側の軽量archiveは32ファイル、6,038 bytes、SHA-256 `0d80d358eba1b18a7123fc71875962134fa17a6824e1679a2bfd7de2d6713c69`で、best checkpointのSHA-256はSFT-only `8c0c936c4b42432c7ebbfd49f5aeacfb4770f76ec04c0f6fa23f2e3c7fa08b6d`、rehearsal `82eec9f7b4f178c852583117c2cc4314952f203caf7889a4b308f8dd916f5de6`です。全checkpoint hashは`artifacts/checkpoints/issue1-20m-sft-colab-1k/colab_checkpoint_manifest.json`へ保存し、学習終了後にsessionを停止して`colab sessions`が空であることを確認しました。次に両best checkpointをreloadしてdomainと固定chatを評価します。

## 結果と解釈

最終および最良checkpoint、SFT validation loss、通常domain loss、固定chatのEOS・生成長・overlap、生成本文の人手観察を条件別に記録します。5M smokeの実測結果は上の実験中記録へ追記し、domain評価JSON、固定chatのJSON/TXT、レビュー用JSON、学習中の全生成TXT、metrics、checkpoint metadataをGitの追跡対象にします。SFT validationの改善は応答形式への適合を示すだけで、会話の事実性や医学的正確性を示さないことを明記します。

## 次に試すこと

rehearsalが忘却を抑えた場合は、ratio 0.10、0.25、0.50の比較へ進みます。SFT-onlyが十分に改善しない場合は、短い応答の層化sampling、会話source別SFT、または教師LLMによる日本語instruction dataの作成を検討します。20Mで安定した後に、より大きなモデルやLoRA/QLoRAとの比較へ進みます。
