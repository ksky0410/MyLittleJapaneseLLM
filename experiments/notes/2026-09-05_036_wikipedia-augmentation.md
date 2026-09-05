# 実験036：日本語Wikipedia追加によるデータ量比較

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。実験029では約5M TokenのFineWeb混合コーパスを2,500 stepまで学習し、実験030では同じデータと学習条件のまま5M級から約20M級へ拡張しました。20M化によってvalidation lossは改善しましたが、固定promptの文法と固定chat-testの意味対応はまだ弱く、約5M Tokenが容量に対して少なすぎる可能性があります。

今回の仮説は、FineWebに加えて日本語Wikipedia本文を約5M Token投入し、学習候補を約10M Tokenへ増やせば、5Mモデルでも一般validationとWikipedia由来validationのlossが改善するというものです。一方でWikipediaは百科事典文体であり会話ではないため、会話validationや固定chat-testが改善するとは仮定しません。会話lossが悪化する場合は、source比率の問題として次のsource ablationで調べます。

実験029・030との比較可能性を優先し、モデル、Tokenizer、batch size、seed、optimizer、learning rate schedule、context length、最大stepを固定します。変更するのは学習Token列のsource構成だけです。Wikipediaの取得・抽出処理は実験031に分離し、元parquetを再取得せず、その出力本文を使用します。

## データとTokenizer

新しい学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1-train.bin`です。作成時には次の5 sourceを使い、weightを`aozora=8`、`fineweb=8`、`wikipedia=8`、`conversation=1`、`medical=1`とします。

- 青空文庫一般本文：`artifacts/corpus/aozora-general-v1.txt`
- FineWeb2 Edu Japanese抽出本文：`artifacts/corpus/fineweb2-edu-japanese-v1/train.txt`
- Wikimedia Wikipedia日本語抽出本文：`artifacts/corpus/wikimedia-wikipedia-ja-v1.txt`
- 公開会話コーパス：`artifacts/corpus/conversation-v1/train.txt`
- 医師国家試験データ：`artifacts/corpus/medical-qb-v2/train.txt`

Token予算は10,000,000とし、論理単位を分割・複製せず、既存Tokenizerで測ったEOS込みToken数が上限を超えないように選びます。source間の同一本文は混合器のsource順で最初の一つだけを採用します。混合条件、実際のsource別Token数、入力hash、出力hashは`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.manifest.json`へ保存します。

Tokenizerは`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、vocab size 4,096、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`を固定します。一般・会話・医療validationとFineWeb testは実験030と同じToken列を使用し、Wikipedia追加の影響は後続でWikipedia専用validationを作成して確認します。

## モデルと学習条件

モデルは実験029と同じ5,197,920 parameter概算のdim 240、6層、6 heads、context length 256、MLP倍率4、absolute position embedding、重み共有を使用します。batch size 8、最大2,500 step、evaluation interval 100、sample interval 100、evaluation batches 20、AdamW、learning rate 3e-4、minimum learning rate 3e-5、warmup 300、weight decay 0.1、seed 42です。設定は`configs/fineweb2-wikipedia-augmented-ja-5m-2p5k.toml`へ固定します。

本学習の前に`configs/fineweb2-wikipedia-augmented-ja-5m-smoke.toml`で100 stepのsmokeを行い、データ長、モデルshape、Metal、NaNの有無を確認します。smoke出力と本学習出力は別ディレクトリに保存し、既存の実験029のcheckpointとsampleを上書きしません。

## 実行コマンド

```bash
.venv/bin/python scripts/mix_corpora.py \
  --source aozora=artifacts/corpus/aozora-general-v1.txt \
  --source fineweb=artifacts/corpus/fineweb2-edu-japanese-v1/train.txt \
  --source wikipedia=artifacts/corpus/wikimedia-wikipedia-ja-v1.txt \
  --source conversation=artifacts/corpus/conversation-v1/train.txt \
  --source medical=artifacts/corpus/medical-qb-v2/train.txt \
  --weight aozora=8 \
  --weight fineweb=8 \
  --weight wikipedia=8 \
  --weight conversation=1 \
  --weight medical=1 \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --target-tokens 10000000 \
  --seed 42 \
  --output artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.txt \
  --manifest artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.manifest.json

.venv/bin/python scripts/encode_data.py \
  --tokenizer artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  --input artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.txt \
  --output artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1-train.bin

.venv/bin/python scripts/inspect_model.py --config configs/fineweb2-wikipedia-augmented-ja-5m-2p5k.toml
.venv/bin/python scripts/train.py --config configs/fineweb2-wikipedia-augmented-ja-5m-smoke.toml
.venv/bin/python scripts/train.py --config configs/fineweb2-wikipedia-augmented-ja-5m-2p5k.toml
```

学習開始前の公開commitは`79cfda1`です。smoke configのSHA-256は`37868d1dded50b3d3c2310a2acf720fb005b22e45b71fa53c632aed147625ae3`、本学習configは`667a6752cbf9565aa50d0737a383cc1fc17cb7c09547431403a8871c4166a6d0`、`scripts/train.py`は`e8f600df408f53772b3f0729c1d8047a656e1f63b0e8907e04d6502eae612ee0`です。実行環境はPython 3.13.1、既存の`.venv`へ導入済みのMLXです。smokeを先に完走させ、成功後に本学習を開始します。

smokeは同日実行し、100 stepまで正常に完走しました。step 1のtrain lossは8.755156、general validation lossは8.782042、perplexityは6516.169、step 100のtrain lossは6.232876、validation lossは6.992349、perplexityは1088.274でした。学習時間は10.93秒で、NaN、shape error、データ長エラー、メモリ不足は発生していません。smokeのmetrics SHA-256は`432560367eb735e47e74c1c8b84110d5cb19cedb4cbd28b4610d37552f408606`、summaryは`659da2566aff563350791638dd25dedf87c67d885ddc062588d48cb83efd1f71`、step100 metadataは`88107bc091742b59415ccac47e50b6594f61bfac1cd8d8f5cbeac8dd6dc89ab0`、生成TXTは`a065624550faf46186704fb52d05fdef6d3ba2280447c8570f65ecbb241dfe02`です。本学習へ進む条件を満たしました。

本学習は同日開始し、step 1,000まで正常に進行しています。step 500ではtrain loss 5.263438、general validation loss 6.219234、perplexity 502.318、step 700ではtrain loss 4.915477、validation loss 6.002453、perplexity 404.420、step 1,000ではtrain loss 4.763634、validation loss 5.777946、perplexity 323.095でした。step 1,000時点の学習率は`2.381486e-4`、経過時間は112.93秒です。ここまでNaN、shape error、データ長エラー、メモリ不足は発生していません。学習は継続中です。

## 実験中の記録

2026-09-05、実験031で抽出したWikipedia本文と既存4 sourceを用いた混合Token列を準備しました。混合manifestは`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.manifest.json`、出力本文のSHA-256は`4ddbc8da19ab87663a3d94e44db2d5a881993679f38c19c42df41c813fd8b305`、実際のToken数は9,999,973です。Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1-train.bin`、SHA-256は`d043d06180d2c6deb0e0c14038fd1b3f736f86f062cf61260bd19282f8ce48e4`です。Token metadataではvocab size 4,096、EOS ID 3を確認しました。

混合後の実Token比率は、青空文庫5.08%、FineWeb42.18%、Wikipedia42.19%、会話5.27%、医療5.27%でした。指定weightは単位採用の希望比率であり、sourceごとの文書長が違うためToken比率とは一致しません。Wikipediaは738記事しかないため646記事でToken予算に到達しており、追加sourceとしての影響は確認できますが、日本語Wikipedia全体の代表性を示すものではありません。

Wikipedia専用validationも作成しました。本文manifestは`artifacts/corpus/wikimedia-wikipedia-ja-validation-v1.manifest.json`で、Token化後のToken数は998,845、Token列`artifacts/tokens/wikimedia-wikipedia-ja-validation-v1.bin`のSHA-256は`2898e8ab7385dc7beb26e4ba956639eaa791b059a1a7e763ae9d4b958e09d269`です。学習はまだ開始していません。学習実行前に混合manifest、Token列hash、config hash、Python・MLXの実行環境を確認し、smokeと本学習の各節目を追記します。

## 結果と解釈

データ混合、Wikipedia専用validationのToken化、5Mモデルのsmoke、本学習、domain評価、固定chat評価まで完了しました。本学習はstep 2,500までNaN、shape error、データ長エラー、メモリ不足なく完走しました。step 2,400のgeneral validation lossが5.522978で最良だったため、domain・chat評価には`step_002400.npz`を使いました。step 2,500のtrain lossは4.190691、general validation lossは5.525606、perplexityは251.038、学習時間は404.60秒、summary上の経過時間は405.07秒でした。step 2,400のvalidation lossは5.522978、perplexityは250.380でした。実行中のRSSはおおむね88〜118MBでしたが、Apple Siliconのunified memory全体の最大使用量は未計測です。

本学習の中間値は、step 500でvalidation loss 6.219234、step 1,000で5.777946、step 1,500で5.651157、step 2,000で5.550597、step 2,400で5.522978、step 2,500で5.525606でした。step 1,000からstep 2,000までvalidation lossは改善し、step 2,400で最小になった後、step 2,500でわずかに悪化しました。したがって最終stepだけでなく最良checkpointを評価した判断は妥当です。

実験029の5Mモデルと比較すると、general validation lossは5.290503から5.522978へ0.232475悪化しました。会話validation lossは3.341280から3.455209、医療validation lossは3.883457から4.221654、FineWeb test lossは4.560703から4.596619へ悪化しました。Wikipedia専用validation lossは4.594581、perplexityは98.947でしたが、追加前のWikipedia評価はないため、これだけで改善とは判断しません。

固定chat-test-v1の48例ではEOS到達数は48/48でしたが、平均生成Token数は5.0417、overlap precisionは0.123215、recallは0.045912、F1は0.056060でした。実験029のF1 0.072297より低く、short 0.087482、medium 0.042325、long 0.038373のすべてで悪化しました。生成は`今日ははなく、それは、何のもといれ、私は、もちなのかつつでのか、...`のように日本語らしい断片を含みますが、文法と意味の一貫性はまだ弱い状態です。悪い途中生成を含む全stepのTXTと固定chatの全文はGitで追跡します。

今回の結果は、「Wikipediaを追加すれば同じ2,500 stepで既存domainも改善する」という仮説を支持しませんでした。ただし、学習Token列を約5Mから約10Mへ増やしてstepを固定したため、総学習window数は約5.12Mのままで、各データへの反復回数は下がっています。データの多様性が悪いのか、学習Token数が足りないのかを分離できていないため、Wikipedia自体が有害だとは結論づけません。今回の主な知見は、データを増やす場合はstepまたは総学習Token数も合わせて設計する必要があることです。

成果物のSHA-256は、metricsが`58d7281a67b99a8a1b7e91eff8ef283a8fe28a4b2c2ee95b37a3f92a5e34ab12`、summaryが`0615be21cb1f57d42d19df688d19e283694482d4a00f5d887a3d9dea0c53b10a`、最良step2400 metadataが`545fc1fc0d8d27946559bfc998ed6b9e87233a3936eea154c9c3f556a8c716e5`、最良step2400生成TXTが`def912f257236a14df5ce4ff66a636e67d0e2327719dcd900b262734a904ba3b`です。domain評価JSONは`77463ca6984cffa52f3a1619e21db750ba992e6b1e6e80076dbaa0abea15b279`、fixed chat評価JSONは`06c760e76e5657636c3a516b6f710facb030b470ba331de49eb3b99bab00d739`、fixed chat生成TXTは`a1e62bc7ea7bee41d8ab855bb45efd2689c226e7da5b21483e37faf0742b1dfe`です。

## 次に試すこと

Wikipedia追加でvalidation lossが改善しなかったため、次は同じ10M Token列でstepを5,000へ増やし、データへの反復不足が主因かを確認します。その後、Wikipediaなし5M Token列を5,000 stepへ学習する対照条件も作り、総学習Token数をそろえた比較へ進みます。会話testの悪化が続く場合は、Wikipediaのsource比率を下げるablationを行います。現代的な構造変更は、データ比較を終えてからRoPE、RMSNorm、SwiGLUの順に一要素ずつ導入します。
