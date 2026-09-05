# 実験043：RMSNorm・context 512の日本語5M smoke

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験042で使用したRoPE・context length 512の5Mモデルを基準にし、正規化層だけをLayerNormからRMSNormへ変更します。今回の目的は、現代的なdecoder-only Transformerで広く使われるRMSNormを、この小型日本語モデルへ導入したときに、学習の安定性、validation loss、生成結果、実行時間がどのように変わるかを確認することです。

仮説は、RMSNormは平均を引かず、LayerNormより計算とパラメータが少ないため、同じ学習条件でも少なくとも同程度に学習できる可能性がある、というものです。ただし、このモデルは非常に小さく、学習Token数も少ないため、差がノイズに埋もれる可能性があります。RMSNormが常に優れるとは仮定せず、validation lossの推移と生成文の崩れ方をLayerNormの実験042と並べて判断します。

実験042との差分は`model.norm_type = "rmsnorm"`だけです。RoPE、context length 512、dim 240、6層、6 heads、MLP倍率4、batch size 8、最大500 step、評価間隔100、生成間隔100、AdamW、学習率3e-4から3e-5、warmup 300、weight decay 0.1、seed 42、学習・検証Token列、Tokenizerを固定します。RMSNormでは正規化のscaleだけを持つため、概算parameter数は5,133,360で、実験042のLayerNormモデル5,136,480より3,120少なくなります。

## 使用するデータ、Tokenizer、コード

学習Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`で、Token数は1,336,619、SHA-256は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`です。検証Token列は`artifacts/tokens/mixed-ja-80-10-10-v2-general-val.bin`で、11,780 Token、SHA-256は`c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`です。一般80%、会話10%、医療10%の混合コーパスから作った既存Token列を読み取り専用で使用し、元の医師国家試験データや`medilink_analysis`の原本には変更を加えません。

TokenizerはSentencePiece Unigram、語彙数4,096、`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。実験設定は`configs/rmsnorm-context512-mixed-ja-5m-smoke.toml`、SHA-256は`8658c4c9b5977189f57536242d531d9f3be791394d90b7297def4a444fc471cc`です。RMSNorm対応コードのcommitは`4d1b459`、実験設定を追加したcommitは`7e871fc`です。

予定している学習コマンドは次のとおりです。

```bash
.venv/bin/python scripts/train.py \
  --config configs/rmsnorm-context512-mixed-ja-5m-smoke.toml
```

学習後は、`scripts/evaluate_domains.py`でgeneral・conversation・medicalのvalidation lossを測定し、`scripts/evaluate_chat_prompts.py`でIssue #1の固定会話promptを評価します。さらに、checkpointをロードして固定promptを再生成します。stepごとの生成文、崩れた出力、空の出力、評価JSON、checkpoint metadataは削除せず保存します。

## 成功条件と判定方法

500 stepをNaN、Metalエラー、OOM、Token列不足なく完走し、metrics、checkpoint metadata、stepごとの生成TXTが保存されれば実装上の成功とします。checkpoint reloadと固定prompt生成が成功することも確認します。モデル品質については、実験042のLayerNorm条件とvalidation lossを比較します。RMSNorm側のlossが低ければ探索上の改善、差が小さければ同等、明確に高ければこの条件では悪化と判定します。生成文が短く崩れていても削除せず、lossと併せて解釈します。

## 実験中の記録

2026-09-05 12:34 JST、学習開始前の確認を行いました。実行環境はPython 3.13.1、macOS 15.5 arm64、MLXのdeviceは`Device(gpu, 0)`です。ノートと設定はcommit `cacf2de`でpush済みで、設定、入力Token列、TokenizerのSHA-256は前節の値と一致しました。学習中は設定の評価間隔に従って少なくとも100 stepごとにloss、perplexity、生成文、所要時間を保存します。異常や予定変更があれば、その時点で追記します。

12:38 JST、MLX学習終了後に、私が予定していないPyTorch CPUプロセスが同じ設定と出力先で動作していることを確認しました。コマンドは`/Library/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python scripts/train_torch.py --config configs/rmsnorm-context512-mixed-ja-5m-smoke.toml --device cpu --no-amp`でした。これはMLX実験043の計画には含めていないため、一部のMLX metricsとstep 100・200 metadataへ上書きが生じた時点でプロセスを停止しました。CPU側が生成した`step_000001.pt`、`step_000100.pt`、`step_000200.pt`と対応するmetrics・metadataは`artifacts/checkpoints/rmsnorm-context512-mixed-ja-5m-smoke-unexpected-concurrent-torch-before-stop/`へ退避し、削除していません。MLX側のmetricsとstep 100・200 metadataは、学習時の標準出力と残存するMLX checkpointをもとに復元しました。この予定外のCPU実行は、RMSNormの主結果には混ぜません。
退避したCPU重みのSHA-256は、step 1が`64bb265120b201fd44069b1be96f7da0faeeed966d75972c2f2e311d021912fc`、step 100が`a5dd72f61a814e1f7c214aeb84b8fc9a36a141349c8ff083cf801b67694b87cc`、step 200が`93b05dd7a59d7a8f98c25490316fe0f21b922fc4cb07ca1560f443301d72c3e7`です。partial metricsのSHA-256は`13ea1139ac97cb70cda1e7dbfa70332c16dc141ad46583fc865dac7a148c92d2`です。

domain評価の最初の試行では、存在しない`fineweb2-edu-japanese-fineweb-val.bin`と`wikipedia-ja-val.bin`を指定したため、入力確認の段階で`FileNotFoundError`になりました。学習成果物には変更がなく、実在する`fineweb2-edu-japanese-v1-test.bin`と`wikimedia-wikipedia-ja-validation-v1.bin`へ指定を直して再実行しました。この失敗も削除せず、実験上の出来事として残します。

MLX/Metalの主実験は12:34 JSTごろに開始し、500 stepまで完走しました。step 1のtrain lossは8.817914、validation lossは8.760859、step 100は5.235695と6.671456、step 200は5.257855と6.168312、step 300は4.028846と5.770660、step 400は4.350374と5.461384、step 500は4.335594と5.340078でした。学習中にNaN、Metalエラー、OOM、Token列不足は発生しておらず、validation lossは全記録点で低下しました。

## 結果と解釈

500 stepのMLX学習は正常に完走し、所要時間は170.17秒でした。step 500が最良checkpointで、train lossは4.3355941772、general validation lossは5.3400775592、perplexityは208.5288829855でした。実測parameter数は5,133,360で、LayerNormを使う042の5,136,480より3,120少なくなりました。metricsは`artifacts/checkpoints/rmsnorm-context512-mixed-ja-5m-smoke/metrics.jsonl`、summaryは同ディレクトリの`summary.json`、stepごとのmetadataは同ディレクトリのJSONです。checkpoint本体は`step_000500.npz`です。

042のRoPE・context 512・LayerNorm条件ではgeneral validation lossが5.3410979907でしたので、043の5.3400775592は0.0010204315低くなりました。conversation lossは3.3905173937で、042の3.3903319836より0.0001854102高く、medical lossは4.0437904994で、042の4.0347564220より0.0090340773高くなりました。今回の条件ではgeneralに対してごく僅かな改善を示しましたが、全domainで優れるとは言えません。RMSNormの構造差と小規模学習の揺らぎを一回のsmokeから完全に分離できないため、安定して学習できたことを主な成果とします。

学習中の生成文は`artifacts/samples/rmsnorm-context512-mixed-ja-5m-smoke/step_000000.txt`から`step_000500.txt`まで保存しました。step 0では医療語や断片的な語が連続し、step 300では日本語らしい助詞を含む長い断片が現れましたが、step 500では医師国家試験らしい選択肢と崩れた文字列が混ざりました。lossが低下し生成分布が変化したことは確認できますが、自然な物語や医学的に正しい説明を生成できたとは評価しません。

general・conversation・medical・fineweb・wikipediaの5 domain評価は`artifacts/evaluations/rmsnorm-context512-mixed-ja-5m-smoke-domains.json`へ保存しました。finewebには`artifacts/tokens/fineweb2-edu-japanese-v1-test.bin`、wikipediaには`artifacts/tokens/wikimedia-wikipedia-ja-validation-v1.bin`を使いました。general、conversation、medicalのlossは順に5.3400775592、3.3905173937、4.0437904994、finewebは5.9165558815、wikipediaは6.4525243441でした。このJSONのSHA-256は`31a453e938f732a61a690969c6025b88dcfc1157bafa63af9ee278d393b7f23c`です。finewebとwikipediaは042では測定していないため、LayerNormとの直接比較には使いません。

Issue #1の固定会話prompt 8件は`artifacts/evaluations/rmsnorm-context512-mixed-ja-5m-smoke-chat.json`と`artifacts/samples/rmsnorm-context512-mixed-ja-5m-smoke/chat-issue-1.txt`へ保存しました。JSONのSHA-256は`e6cbb98d10bff36c7f9ae43c1718c2e673f7f18be088b845392bb768a720a360`、可読TXTのSHA-256は`304316bddb1602fc1717be273d3725b0d206b799fc9263fe0a353b5aaec6fa08`です。RMSNorm側は8件中7件でEOSへ到達し、空completionは2件、平均completion長は35.5 Tokenでした。`それな`では医学問題の断片が160 Token続き、`おつかれ`でも医学問題の断片が80 Token続きました。042は8件すべてEOS、空completion 2件、平均4.75 Tokenでしたので、今回の長いcompletionは会話能力の改善ではなく、停止しにくい医学コーパス由来の断片と解釈するのが妥当です。

checkpoint reload後の`今日は`、`吾輩は`、会話marker、`問題：`の生成はすべて成功し、それぞれ`reloaded-today.txt`、`reloaded-story.txt`、`reloaded-conversation.txt`、`reloaded-medical.txt`へ保存しました。会話markerでは「こんにちは!」が返りましたが、他の出力は医療問題の断片や` sation|>`のような崩れを含みます。step 500 checkpointのSHA-256は`99dc0a74b45a17a32332c3971fc032a805a857a6805e55ce57c9ec805eea8bb7`です。reload生成のSHA-256は順に、`2d9b8df3c56cd49d02b6588d4f735edea617586d6529f1ea637b61f66ab8aaaf`、`71901d2c64b6044a5120a3a83e6d841a026f65d6a7ed1d3f0b9873b427de1d06`、`d8502b0ffc4614ce3fa6ff22f3bfc2a97ac739703b2dbf69c25114cee20f542f`、`704208fe746136be98c7cbbe4b6e56862a9361209c5aa22505808e9e18aa2f29`です。

以上から、実装上の成功条件は満たしましたが、品質面では「RMSNormを導入してもこの小規模・短時間条件では明確な優位性は見えない」と判定します。general lossの改善幅は小さく、conversationとmedicalは僅かに悪化しました。固定会話promptでは最大長まで医学断片が続く例があり、会話能力や医療能力の向上は認定しません。RMSNorm単独の導入、checkpoint reload、旧LayerNorm checkpoint互換性の検証には成功しました。

## 次に試すこと

次は、043と同じMLX/Metal条件で学習stepを250へ下げ、LayerNormのcontext 256・500 stepと総Token予算を近づける対照を作ります。その後、RMSNormを固定したままSwiGLUを一つだけ導入し、MLP構造の差を独立に比較します。いずれも開始前に別実験番号を発行し、今回の主結果と予定外のCPU成果物を混ぜません。043の結果だけから医学的な応答性能や実用性を評価しません。
