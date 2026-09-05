# 実験047：10M日本語Token・20M級SwiGLUモデルの拡大実験

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。5M級で行った実験045・046では、SwiGLUがGELUよりvalidation lossを低下させましたが、学習Token数は約1.34Mと少なく、生成文には医療問題の断片や英数字の崩れが残りました。実験047では、SwiGLU・LayerNorm・RoPEを固定したままモデルを約5Mから20M級へ拡大し、さらにFineWeb2 Edu JapaneseとWikipediaを含む約10M Tokenの学習corpusを使います。目的は、モデル容量と学習データを同時に現実的な規模へ増やしたとき、5Mで観測したloss改善が残るか、自然な日本語の生成がどこまで安定するかを確認することです。

今回の仮説は、20M級モデルは5M級よりvalidation lossが低くなり、WikipediaやFineWeb由来の一般文書に対するlossも改善する可能性がある、というものです。ただし、モデルサイズとデータ量を同時に増やすため、改善の原因を容量だけへ帰属させません。比較対象は既存の20M GELU・absolute条件、5M SwiGLU・LayerNorm・RoPE条件、同じ10M corpusで追加する20M結果とし、学習Token数、更新回数、context長、backendを表にして管理します。20M本学習の前に100 step smokeを実施し、shape、Metal/CUDA、NaN、checkpoint保存、生成を確認します。

smoke設定は`configs/fineweb2-wikipedia-10m-20m-swiglu-rope-layernorm-smoke.toml`、本学習設定は`configs/fineweb2-wikipedia-10m-20m-swiglu-rope-layernorm-2p5k.toml`です。vocab size 4,096、dim 384、10層、6 heads、context length 512、RoPE、LayerNorm、SwiGLU、MLP倍率4、batch size 8、seed 42を固定します。SwiGLUの中間次元は1,024で、重み行列のFFN parameter予算はGELUの中間次元1,536と揃えます。概算parameter数は約19.3M、biasを含む実際のparameter数は約19.4Mとなる見込みです。

## 使用するデータ、Tokenizer、コード

学習Token列は`artifacts/tokens/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1-train.bin`で、Token数は9,999,973、SHA-256は`d043d06180d2c6deb0e0c14038fd1b3f736f86f062cf61260bd19282f8ce48e4`です。混合元corpusのmanifestは`artifacts/corpus/mixed-ja-token-budget-fineweb2-wikipedia-10m-v1.manifest.json`で、本文SHA-256は`4ddbc8da19ab87663a3d94e44db2d5a881993679f38c19c42df41c813fd8b305`です。manifestに記録されたToken比率は、青空文庫5.0828%、FineWeb2 Edu Japanese42.1840%、Wikipedia42.1891%、会話5.2725%、医療5.2715%です。元の医師国家試験データや`medilink_analysis`の原本は読み取り専用で扱い、加工済みcorpusだけを使用します。

検証には`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`（11,780 Token、SHA-256 `c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`）を使います。追加domain評価ではFineWeb2 Edu Japaneseのtest Token列（`36d8d5c8bc92de1e168b8c3de9dd4ee975dec66f6b644b83bfbf9b239877161c`）とWikipedia validation Token列（`2898e8ab7385dc7beb26e4ba956639eaa791b059a1a7e763ae9d4b958e09d269`）を使います。

TokenizerはSentencePiece Unigram、語彙数4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。使用コードはMLXの`train.py`とPyTorchの`train_torch.py`です。過去のLayerNorm・GELU checkpointと混同しないよう、checkpoint signatureへ`position_embedding`、`norm_type`、`ffn_type`を記録します。

smokeの予定コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py \
  --config configs/fineweb2-wikipedia-10m-20m-swiglu-rope-layernorm-smoke.toml
```

smoke完了後、本学習は次の設定で実行します。MacBookのMetalが利用できない場合は、同じ設定をCUDA用bundleへまとめ、PyTorchのT4など別backendで実行した場合はMLX結果と別の実験runとして記録します。

```bash
.venv/bin/python scripts/train.py \
  --config configs/fineweb2-wikipedia-10m-20m-swiglu-rope-layernorm-2p5k.toml
```

## 成功条件と判定方法

smokeは100 step、本学習は2,500 stepをNaN、OOM、shape error、Token列不足なく完走し、metrics、checkpoint metadata、stepごとの生成TXTが保存されることを成功条件とします。checkpoint reload後に固定promptが生成できることも確認します。比較ではgeneral、conversation、medical、FineWeb、Wikipediaのvalidation lossとperplexity、Issue #1のEOS到達数・空completion数・平均生成Token数を記録します。生成結果は、自然な文だけでなく崩れた文や医療問題の混入も削除せず保存します。

20M化と10M corpus化を同時に行うため、結果が良くても「SwiGLUだけの効果」「データだけの効果」「モデル容量だけの効果」とは表現しません。必要に応じて、同じ10M corpusの5Mモデル、同じモデルの7.5M corpus、複数seedの追加対照を行います。

## 実験中の記録

smoke開始前の基準commitは`86ad135`です。smoke設定のSHA-256は`13935351be7f97b4a7595bface483547bbd90b88a3e8d606a0b4145cd8abdf44`、本学習設定のSHA-256は`e27c4d9b5fffa8465b617b7c4bcf1d56d8a0dd5eaa6a936d5a515db49890ca4e`です。入力Token列とTokenizerのhashは前節の記載どおりで、設定を読み込んだところ20M級の概算parameter数は19,283,712でした。smokeを開始する前に、実行環境、MLXまたはCUDAのdevice、Python・PyTorch・MLX versionを追記します。学習中は100 stepを超えない間隔でmetricsと生成文を確認し、異常や予定変更があればその時点で追記します。長時間の本学習では、checkpoint保存間隔と学習Token数、累積処理Token数、メモリ使用量を記録します。

2026-09-05、開始前の主実験コマンドをこの実行環境で一度だけ試しましたが、MLX import時に`ImportError: [metal::load_device] No Metal device available`となり、step 0のcheckpointやmetricsは作成されませんでした。これはheadless・sandboxed環境でMetal deviceが見えないことによる実行失敗です。失敗を隠すためにPyTorch CPUへ切り替えることはせず、046までのMLX実行成果物も上書きしていません。SwiGLU・20M級の主実験は未実施として扱い、Metalが利用できるMac実行環境またはCUDA bundleへ移してから再開します。

2026-09-05 13:32 JST、Metalが利用できないため、同じ047設定をPyTorch/CUDA bundleへまとめ、Colab T4でsmokeを実行する計画を追加しました。まず100 stepのsmokeでRoPE・LayerNorm・SwiGLUのshape、AMP、checkpoint、生成文を確認し、成功した場合だけ同じbundleを使って2,500 stepの本学習へ進みます。Colab側では041と異なる`047`専用出力先を使い、生成文は100 stepごと、重みcheckpointは500 stepごとに保存します。開始前のこの計画とbundleのhashをcommitした後にsessionを作成します。

13:34 JST、commit `67e51f9`を基準に047 bundle `/tmp/small_llm-colab-047.tar.gz`を作成しました。bundleサイズは約12MB、SHA-256は`d93990206422225bbfff8dc13bc40c9753f12ec33d5628ca4adddbc80f745468`です。bundleには047のsmoke・本学習config、PyTorch学習コード、SwiGLU/RoPE対応のモデルコード、Tokenizer、10M Token列、general validation Token列を含め、Python cacheは除外しました。開始前の予定・hashをこのcommitへ固定してから、Colab sessionを作成します。

13:35 JST、`colab new --session torch20m-swiglu-rope-colab-047 --gpu T4`で新規T4 sessionを作成しようとしましたが、HTTP 412 `Precondition Failed`、`TooManyAssignmentsError`で割り当てられませんでした。041の完走直後でColab側の割当上限に達した可能性があります。`colab sessions`に047 sessionは残らず、bundleのuploadや047のColab学習stepには到達していません。既存sessionの流用は行わず、失敗を記録したうえで、構造確認だけをローカルPyTorch CPU smokeへ切り分けます。

13:37 JST、ColabのGPU割当失敗を性能結果と混同しないため、出力先を分離した`configs/fineweb2-wikipedia-10m-20m-swiglu-rope-layernorm-cpu-smoke.toml`を作成しました。これは047のsmokeと同じモデル・データ・100 step条件を使い、CPUで構造、NaN、checkpoint reloadに相当する成果物、生成処理だけを確認する補助実験です。CPUの速度やlossをColab T4の本実験結果とは比較しません。設定SHA-256は`89074fe4299c3726ca142300fedec5c17cf98b37c0c65d30a9737c7f8217a4e3`です。実行コマンドは`.venv/bin/python scripts/train_torch.py --config configs/fineweb2-wikipedia-10m-20m-swiglu-rope-layernorm-cpu-smoke.toml --device cpu`です。この計画をcommitへ固定してから実行します。

## 結果と解釈

実験終了後、smokeと本学習を混ぜずに、backend、最終および最良checkpoint、train・validation loss、domain評価、固定chat評価、reload生成、失敗・停止理由、成果物hashを追記します。20Mモデルの出力を医療助言や医学的正解として扱わず、医師国家試験データを含むことによる見かけの専門性と、一般日本語の生成能力を分けて評価します。

## 次に試すこと

047の結果が安定していれば、同じ20M構造で学習Token数を増やすか、vocab 8,192・16,384のTokenizer対照を行います。その後、instruction tuning用の日本語会話データを別splitで作成し、pretrainingとSFTを分離して評価します。新しいdatasetを追加する場合は、出所・revision・ライセンス・取得日時・hash・混合比率を先に記録し、原本を変更せずに加工済みデータを別パスへ保存します。
