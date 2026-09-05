# 実験050：Issue #1のcore対bothを20Mへ拡張

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験049では、約5M parameter・500 stepの同一条件で、会話sourceを含めたモデルがRPCまたはMRMPのvalidation lossを大きく改善することを確認しました。一方で、固定chat-testのoverlapだけでは自然さを判定できず、モデルも十分に学習していません。050では、049のcoreとbothを約20M parameterへ拡張し、会話混合による差がモデルサイズを増やしても観測できるかを確かめます。

比較条件は、FineWeb2 Edu Japanese 90%＋医療10%の`core`と、FineWeb2 Edu Japanese約80%＋医療10%＋RPC約5%＋MRMP約5%の`both`です。049と同じ1M Token train列を使い、Tokenのsource比率以外は揃えます。モデルはdim 384、10層、6 heads、context length 256、RoPE、LayerNorm、SwiGLU、約20M parameterです。学習はbatch size 8、2,500 step、約5.12M exposure Token、AdamW、learning rate 3e-4から3e-5、warmup 300、weight decay 0.1、seed 42です。T4上でAMPを有効にし、2条件を同一fresh Colab VMで順番に実行します。

仮説は、bothがgeneral validationではcoreにやや劣る一方、RPC・MRMP・会話固定テストではcoreを上回るというものです。20M化で会話形式の学習が安定する可能性がありますが、train Tokenが約1Mしかないため、2,500 stepでは同じデータを繰り返し見る点に注意します。会話の自然さをlossだけで断定せず、同じdomain評価、固定chat-test、学習途中の生成全文を保存します。

## データ、実装、再現条件

入力は049で作成済みのtrain列とvalidation列です。core trainは999,987 Token、SHA-256 `ebca09587890bfbfb76b6a0d968b198be55943993fc011115a1736d88914e9a4`、both trainは999,970 Token、SHA-256 `758b46f6bb946afd7e2c3604714db71166d79564f8c652e8cc950b23d3338879`です。general validationは`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`、RPC validationは948,172 Token、MRMP validationは156,475 Tokenです。Tokenizerはvocab 4,096の`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。元の医師国家試験データと`/Users/koseki/projects/medilink_analysis`は変更しません。

設定は`configs/issue1-core-20m-colab-2p5k.toml`と`configs/issue1-both-20m-colab-2p5k.toml`です。実行用wrapperは`scripts/colab_bootstrap_050.py`、軽量成果物のarchive作成は`scripts/colab_package_050.py`です。実験開始時点の基準commitは049完了後の`271febb`です。新しいconfig、wrapper、ノートをcommit・pushしてからbundleを作成し、Colab側でbundle内の入力hashを出力して照合します。実行前に確定したコードcommitは`6d0dc2d`です。計画時点のconfig SHA-256はcore `66adc8f8ab73e45e6eada1f81287ca9c6a94c5c49618f12d059e2bc51fb191a7`、both `f62f154127faad211772fe431b848186d4e8df9435cbff083f7b3bcb5b434603`、`scripts/train_torch.py` `c8fb40406ec74635ba63159f86fcd55ef71724edc7cb8ffda53453222640203e`、`scripts/colab_bootstrap_050.py` `f2746c971785cf0f18a92908780eeeaa7c8fd98e18121441c396bcce6c806d39`、`scripts/colab_package_050.py` `0838420581672a567e481bf95b0ed757004cce977e0f08ce533b0bfc9cf6c645`です。bundleは`/tmp/small_llm-colab-050-6d0dc2d.tar.gz`、サイズ約2.8MB、SHA-256 `c67b98a5bfeabd8e508bc341beac9fd92159195a0e3f6b007c429a1abaff8356`です。

## 実行コマンド

```bash
tar --exclude='*.pt' --exclude='*.npz' --exclude='*.vocab' \
  -czf /tmp/small_llm-colab-050.tar.gz \
  configs/issue1-core-20m-colab-2p5k.toml \
  configs/issue1-both-20m-colab-2p5k.toml \
  scripts/train_torch.py scripts/_common.py scripts/colab_bootstrap_train.py \
  src artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  artifacts/tokens/issue1-core-1m-fineweb-train.bin \
  artifacts/tokens/issue1-both-1m-fineweb-train.bin \
  artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin

colab new --session exp050-20m-core-both --gpu T4
colab upload --session exp050-20m-core-both /tmp/small_llm-colab-050.tar.gz /content/exp050_bundle.tar.gz
colab exec --session exp050-20m-core-both --timeout 1800 --file scripts/colab_bootstrap_050.py
colab exec --session exp050-20m-core-both --timeout 120 --file scripts/colab_package_050.py
colab download --session exp050-20m-core-both /content/exp050-lightweight.tar.gz /tmp/exp050-lightweight.tar.gz
colab download --session exp050-20m-core-both /content/exp050-manifest.json /tmp/exp050-manifest.json
colab stop --session exp050-20m-core-both
```

Colab割当に失敗した場合は、失敗時刻、CLIのエラー、session状態を記録し、重い学習を開始しません。学習途中に停止した場合も、回収できたmetrics・生成文・metadataを失敗成果物として残します。

## 成功条件

2条件が同じGPU・seed・モデル構造・学習stepで完走し、NaN、OOM、shape error、Token列不足がないことです。各条件について、step 0と100 stepごとの生成TXT、metrics、summary、checkpoint metadata、best checkpointのhash一覧を保存します。完走後はgeneral、conversation、medical、RPC、MRMPのdomain評価と、48例の固定chat-testを同じseedで実行します。GPU割当や学習が失敗しても、その事実自体を実験結果として記録します。

## 実験中の記録

開始前、Colab割当、bundle検証、各条件の開始・途中・終了、回収、評価の節目をこのノートへ追記します。生成文は品質に関係なく削除しません。重い`.pt`本体はGitHubへpushせず、Colab側manifestとbest checkpointのSHA-256を残します。

## 結果と解釈

実験終了後に、GPU環境、実測parameter数、学習時間、最終および最良loss、source別domain loss、固定chatのEOS・生成長・overlap、生成本文の観察を追記します。20M・約5M exposure Tokenという条件の結果であり、大規模な日本語LLMの性能や医学的正確性を意味しないことを明記します。

## 次に試すこと

完走した場合は、coreとbothの差を踏まえて、bothの学習Token量を増やすか、049のSFT用応答マスクを20M checkpointへ適用します。Colab割当が継続して不安定なら、既存GPU成果物を優先して評価し、MacBookではデータ処理・評価・小型smokeを進めます。
