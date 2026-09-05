# 実験039：Wikipedia中間比率7.5M Token列の5,000 step学習

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験037ではWikipediaを含む約10M Token列を5,000 step学習し、general validation loss 5.216578、Wikipedia test loss 4.275204、fixed chat F1 0.063502となりました。実験038ではWikipediaなし約5M Token列を同じ5,000 step学習し、general loss 4.986886、Wikipedia test loss 4.674947、fixed chat F1 0.063110となりました。

今回の仮説は、037のWikipedia比率が現在の5M級モデルには高く、Wikipediaを半分程度に下げれば、038に近いgeneral・conversation・medical・FineWeb性能を保ちながらWikipedia testも改善するというものです。Wikipediaをゼロから少量へ増やすことで、domainの広がりと既存分布への反復のバランスを確認します。fixed chat F1は037・038と同程度か、038を上回ることを期待しますが、自動Token overlapだけで会話能力を断定しません。

混合器には、実験037と同じ5 sourceを用い、weightを`aozora=8`、`fineweb=8`、`wikipedia=4`、`conversation=1`、`medical=1`と指定します。targetは7,500,000 Tokenです。weightは論理単位の選択優先度であり、実際のToken比率はsourceごとの単位長と上限で変わるため、作成後のmanifestにある実測Token比率を採用します。これは「Wikipediaを正確に半分のToken比率にする」実験ではなく、037と038の中間を狙う探索条件です。

モデル、Tokenizer、batch size、seed、optimizer、learning-rate schedule、最大stepは037・038と固定します。037・038と同じMacBook MLX backendで学習し、Colab PyTorch backendとの差を混ぜません。

## 使用するデータ、Tokenizer、モデル

sourceは次の5つです。元の医師国家試験データは`/Users/koseki/projects/medilink_analysis`側を変更せず、リポジトリ内の既存加工済み`artifacts/corpus/medical-qb-v2/train.txt`を読み取り専用で使います。

- 青空文庫一般本文：`artifacts/corpus/aozora-general-v1.txt`
- FineWeb2 Edu Japanese：`artifacts/corpus/fineweb2-edu-japanese-v1/train.txt`
- Wikimedia Wikipedia日本語本文：`artifacts/corpus/wikimedia-wikipedia-ja-v1.txt`
- 公開会話コーパス：`artifacts/corpus/conversation-v1/train.txt`
- 医師国家試験データの加工済み本文：`artifacts/corpus/medical-qb-v2/train.txt`

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。学習Token列、混合manifest、各sourceのhashはデータ作成後に追記します。

モデルはdim 240、6層、6 heads、context length 256、MLP倍率4、absolute position embedding、概算5,197,920 parametersです。batch size 8、最大5,000 step、evaluation/sample interval 100、evaluation batches 20、AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42です。設定は`configs/fineweb2-wikipedia-mid-ja-5m-5k.toml`です。

## 実行前の再現情報

実験038を完了したcommitは`e84db48`です。config SHA-256は`e0673e6e44f8ac652e78d0b60ffec5cbb7e4e996a72c8be424b58e2425bd25c4`です。データ混合とToken化を先に実行し、manifestとToken列のhashを確認してから学習を開始します。

2026-09-05、weight `8:8:4:1:1`、seed 42、target 7,500,000 Tokenで混合を完了しました。混合本文は`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.txt`、実際のToken数は7,499,997、本文SHA-256は`6a0ea02b8d0e5baf83c05e1d487c53fce6e9b8b1056d06c440caa546948aca31`です。manifestは`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.manifest.json`、SHA-256は`1206a38179282b5e9f271b73100bfc028c712de2cee67d7577d941cfab49a69f`です。実測Token比率は、青空文庫6.777%、FineWeb53.282%、Wikipedia26.632%、会話6.651%、医療6.657%でした。Wikipediaは037の42.189%と038の0%の中間に近い比率になりました。

Token化も完了しました。Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin`、7,499,997 Token、SHA-256は`3bad9f5f9546d98fc598d602a053648679d6e7817161f0add7a219b020c7440a`です。Token metadataは`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin.json`、SHA-256は`0d187e0964b403d38d95d70fb82128f495e32e82beb23f0048a1dbf41ed20fd3`です。vocab size 4,096、EOS ID 3、Tokenizer SHA-256は計画どおりです。学習開始条件を満たしました。

予定コマンドは次のとおりです。

```bash
.venv/bin/python scripts/mix_corpora.py \
  --source aozora=artifacts/corpus/aozora-general-v1.txt \
  --source fineweb=artifacts/corpus/fineweb2-edu-japanese-v1/train.txt \
  --source wikipedia=artifacts/corpus/wikimedia-wikipedia-ja-v1.txt \
  --source conversation=artifacts/corpus/conversation-v1/train.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --weight aozora=8 \
  --weight fineweb=8 \
  --weight wikipedia=4 \
  --weight conversation=1 \
  --weight medical=1 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --target-tokens 7500000 \
  --seed 42 \
  --output artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.txt \
  --manifest artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.manifest.json

.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1.txt \
  --output artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-mid-7p5m-v1-train.bin

.venv/bin/python scripts/train.py --config configs/fineweb2-wikipedia-mid-ja-5m-5k.toml
```

成功基準は、混合とToken化が決定的に完了し、5,000 step学習をNaN、shape error、データ長エラー、メモリ不足なく完走することです。学習中は1,000 stepを超えない間隔でmetricsと生成TXTを記録し、完了後は037・038と同じdomain評価とfixed chat評価を実行します。

## 実験中の記録

データ混合とToken化は正常に完了しました。混合器のエラー、Token数不足、Tokenizer mismatchは発生していません。学習開始前の計画、実測source比率、本文とToken列のhashをこのノートへ記録済みです。2026-09-05に5,000 stepのMLX学習を開始しました。step 1ではtrain loss 8.781590、general validation loss 8.800259、perplexity 6635.964でした。warmup中のstep 300ではtrain loss 5.823933、validation loss 6.379888、perplexity 589.862、learning rate `3.000000e-4`、経過時間31.31秒でした。step 1,000ではtrain loss 4.936625、validation loss 5.706295、perplexity 300.755、learning rate `2.855307e-4`、経過時間123.05秒、step 1,300ではtrain loss 4.772246、validation loss 5.586837、perplexity 266.890、step 1,600ではtrain loss 4.473612、validation loss 5.485989、perplexity 241.287、step 1,700ではtrain loss 4.479951、validation loss 5.478712、perplexity 239.538、step 1,800ではtrain loss 4.845807、validation loss 5.441229、perplexity 230.726、step 1,900ではtrain loss 4.241998、validation loss 5.422994、perplexity 226.556、step 2,000ではtrain loss 4.191769、validation loss 5.415304、perplexity 224.821、step 2,100ではtrain loss 4.514877、validation loss 5.371833、perplexity 215.257、step 2,400ではvalidation loss 5.336113、step 2,500では5.321655、step 2,600では5.331264、step 2,700では5.292813、step 2,800では5.274889、step 2,900では5.248127、step 3,000では5.254659、step 3,100では5.223692、step 3,200では5.233642、step 3,300では5.219729、step 3,400では5.216045、step 3,500では5.191690、step 3,600では5.194812、step 3,700では5.174340、step 3,800では5.167134、step 3,900では5.158949、step 4,100では5.140823まで下がりました。step 2,500と3,000、3,200、3,600、4,000で小さな反発はありましたが、その後に回復しています。step 4,100時点でNaN、shape error、データ長エラー、メモリ不足は発生しておらず、学習は継続中です。step 300から4,100までの生成結果は`artifacts/samples/fineweb2-wikipedia-mid-ja-5m-5k/`へ保存されています。

step 4,200ではvalidation loss 5.138404、step 4,300では5.129627、step 4,400では5.124960、step 4,500では5.118013、step 4,700では5.104936、step 4,900では5.100070まで下がりました。step 4,800では5.108015へ小さく反発しましたが、その後に回復しています。step 5,000ではvalidation loss 5.110371、perplexity 165.732となり、最終stepまでNaN、shape error、データ長エラー、メモリ不足は発生しませんでした。step 4,900を最良checkpointとして評価します。

## 結果と解釈

2026-09-05、5,000 stepを正常に完走しました。最終stepのtrain lossは3.734555、general validation lossは5.110371、perplexityは165.732でした。validation lossが最小だったstep 4,900を最良checkpointとし、train loss 4.325948、general validation loss 5.100070、perplexity 164.033でした。metrics上の学習時間は948.38秒、summary上は948.94秒です。成功基準としていた異常なしの完走、1,000 step以下の記録間隔、checkpoint metadata、固定prompt生成TXTの保存を満たしています。最良checkpointは`artifacts/checkpoints/fineweb2-wikipedia-mid-ja-5m-5k/step_004900.npz`です。

実験037のWikipedia追加約10M Token列・5,000 step条件と比べると、039のgeneral validation lossは5.100070で、037の5.216578より0.116508低くなりました。実験038のWikipediaなし約5M Token列・5,000 step条件の4.986886よりは0.113184高く、general lossだけでは038が最良です。ただし、039は約7.5M Token列を約1.37周見ており、038は約5M Token列を約2周、037は約10M Token列を約1周見ています。したがって、差はWikipedia比率だけでなく、sourceの多様性と各sampleの反復回数を含みます。

最良checkpointのドメイン別評価では、general loss 5.100070（PPL 164.033）、conversation loss 3.147053（PPL 23.267）、medical loss 3.497018（PPL 33.017）、FineWeb test loss 4.227962（PPL 68.577）、Wikipedia test loss 4.362479（PPL 78.451）でした。039は037と038の中間に近く、037と比べてgeneral、conversation、medical、FineWebがそれぞれ0.116508、0.040609、0.129482、0.022499改善し、Wikipediaも4.275204からは悪化するものの038の4.674947より0.312468改善しました。Wikipedia比率を約26.6%に抑えたことで、general性能を大きく損なわずにWikipedia文体への適応を戻せるという仮説は支持されました。

固定chat-test-v1の48例では、EOS到達が48/48、平均生成Token数は6.417、overlap precisionは0.147848、recallは0.081640、F1は0.089495でした。037のF1 0.063502、038のF1 0.063110を上回り、今回の3条件では最良でした。層別F1はshort 0.134579、medium 0.101528、long 0.032377で、短・中の応答で特に改善し、longは038の0.059802を下回りました。固定chatは自動Token overlapであるため意味的な正しさを完全には測れませんが、少量Wikipediaを加えた039はEOSを保ったまま短い日本語応答の一致率を改善しました。

固定prompt `今日は`への最良checkpoint生成には、「今日はすぐに出かけない!」「このサイトにとつかると思います」など一部の日本語らしい断片がある一方、「新さ」「新典」「アプリリース」や英字・記号の連続も残っています。自然な日本語生成という最終目標は未達です。この悪い出力を含め、step 0から5,000までの生成TXTとfixed chatの全文TXTを削除せず保存しました。

今回の結果から、現在の5M級モデルではWikipediaを約42%まで増やすより、約27%へ抑えた中間比率の方が、general・Wikipedia・fixed chatのバランスに優れていました。general lossのみならWikipediaなし038が最良ですが、会話F1とWikipedia testを同時に見ると039が有望です。次は039のsource比率を固定して20M級へ拡張するのが自然ですが、構造変更前に同じ7.5M Token列で学習総Token数やschedule horizonをそろえた対照を追加すると、反復回数の影響をさらに分離できます。

成果物のSHA-256は、metricsが`96698573a2322a4d29f3c412d6ab26bb925b7ba08ec2ec50a1b7a04b9d8df3c8`、summaryが`c90595212766521f455811999699e9c1b1f63369a9aef8cad13c13a8520cc394`、step 4,900 metadataが`d964aa364c61e41ce6e137c3ec80d6230e75aca5d3e0e9ef4dc34b8521011e1e`、step 4,900生成TXTが`467ac65a79a6a305bc0029c8e22751c7c954e986336e2edf17743e0e9dc04e36`です。domain評価JSONは`28fa2a577ed895dab916d16314995ec0e10189aaee787033ff85416da1cdd41e`、fixed chat評価JSONは`7b4001a14ba3de3d34c6e9fcb8c2ec06cede7379cff3681d92a5c6fdfa6ad148`、fixed chat生成TXTは`b53e22d2a1c00ed2e55d65728730e7d4476d49ceae7b717cc45bc42f1e9a78f1`です。

## 次に試すこと

039の約27% Wikipedia比率は、general loss、Wikipedia loss、fixed chat F1のバランスが最もよい候補になりました。次はこのsource比率を使って、20M級モデルをColab GPUで学習します。その前に、同じ7.5M Token列を学習総Token数に合わせて十分に反復する条件を必要に応じて追加し、038・039の差を確認します。構造変更はデータ条件が決まってからRoPE、RMSNorm、SwiGLU、GQAの順に一つずつ試し、各変更でMacBook MLXとColab PyTorchを混同しないように記録します。
