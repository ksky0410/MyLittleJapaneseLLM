# 実験045：SwiGLU・RoPE・context 512の日本語5M smoke

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験042のLayerNorm・RoPE・context 512条件を基準にし、実験044ではFFNだけをGELUからSwiGLUへ変更します。目的は、現代的なdecoder-only Transformerで広く使われるSwiGLUをこの日本語小型モデルへ導入したとき、同程度のparameter予算で学習の安定性、validation loss、生成結果、実行時間がどう変わるかを確認することです。

事前の仮説は、SwiGLUのgateによって表現力が増し、同じ学習条件でvalidation lossがGELU条件と同程度か少し低くなる可能性がある、というものです。ただし、500 stepの短いsmokeであり、モデルも約5M parameterにすぎないため、差がほとんど見えないか、初期化とsamplingの揺らぎに埋もれる可能性も高いと考えます。SwiGLUが常に優れるとは仮定せず、まず実装とcheckpoint reloadが成立するかを確認し、042との数値差は探索上の参考として扱います。

今回の042との差分は`model.ffn_type = "swiglu"`だけです。LayerNorm、RoPE、dim 240、6層、6 heads、context length 512、MLP倍率4、batch size 8、最大500 step、評価・生成間隔100、AdamW、学習率3e-4から3e-5、warmup 300、weight decay 0.1、seed 42、学習・検証Token列、Tokenizerを固定します。SwiGLUはgate・up・downの3射影を使うため、中間次元を`2/3 × dim × mlp_ratio`として640にし、GELUのup・downの中間次元960とほぼ同じFFN parameter数に揃えます。RoPEでは位置Embeddingを持たないため、概算parameter数は042と同じ5,136,480です。

## 使用するデータ、Tokenizer、コード

学習Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`で、1,336,619 Token、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`で、11,780 Token、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。既存の一般・会話・医療混合コーパスから作ったToken列を読み取り専用で使用し、元の医師国家試験データや`medilink_analysis`の原本には変更を加えません。

TokenizerはSentencePiece Unigram、語彙数4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。設定は`configs/swiglu-context512-mixed-ja-5m-smoke.toml`です。SwiGLUをMLXとPyTorchの両方へ実装し、`ffn_type`をcheckpointのmodel signatureへ含めます。既存設定は既定値`gelu`として解釈し、過去のGELU checkpointはmetadataにこの項目がなくても読み込めるよう互換性を維持します。

予定している学習コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py \
  --config configs/swiglu-context512-mixed-ja-5m-smoke.toml
```

学習後は`evaluate_domains.py`でgeneral・conversation・medicalに加え、FineWeb2 Edu JapaneseとWikipediaのvalidation lossを測定し、`evaluate_chat_prompts.py`でIssue #1の固定会話promptを評価します。checkpointをロードして固定promptも再生成します。stepごとの生成文、崩れた出力、空の出力、評価JSON、checkpoint metadataは削除せず保存します。

## 成功条件と判定方法

500 stepをNaN、Metalエラー、OOM、Token列不足なく完走し、metrics、checkpoint metadata、stepごとの生成TXTが保存されれば実装上の成功とします。checkpoint reloadと固定prompt生成が成功することも確認します。品質については042のLayerNorm・RoPE・context 512条件とgeneral、conversation、medicalのvalidation loss、固定会話promptの停止状況を比較します。lossが低ければこの条件での改善候補、差が小さければ同等、明確に高ければ悪化と判定しますが、一回のsmokeだけでSwiGLU全体の優劣は結論づけません。

## 実験中の記録

学習開始前に、設定・入力Token列・TokenizerのSHA-256、現在のGit commit、MLXのdeviceをこのノートへ追記します。学習中は少なくとも100 stepごとにtrain loss、validation loss、perplexity、生成文、所要時間を保存します。Metal deviceが見えず実行できない場合も失敗として記録し、CPUや別backendで実行した場合は主実験と混ぜません。

## 結果と解釈

実験終了後、実際に使ったbackend、開始・終了時刻、最終および最良checkpoint、train・validation loss、各domainの結果、Issue #1の生成文、reload結果、失敗や設定変更をその場で追記します。042との比較に使った数値には、同じseed・データ・Tokenizer・モデルサイズを使えたかを明記します。生成文は自然さを過大評価せず、医学的正確性もこのモデルの出力だけから認定しません。

## 次に試すこと

SwiGLUの結果が得られたら、次は同じFFNでLayerNormとRMSNormを揃えた比較、または学習Token予算を増やした10M・20M級モデルへ進みます。データ追加を行う場合は、新しいdatasetの出所・revision・ライセンス・取得日時・hash・混合比率を先に記録し、元データを変更せずに加工済みdatasetを別パスへ保存します。
