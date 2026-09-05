# 実験062：Issue #1のRPC単独・MRMP単独を20M事前学習で比較

## 開始前の計画

実施日は2026年9月5日、担当者はCodexです。Issue #1では、通常の日本語文書へ会話データを混ぜるだけでなく、RealPersonaChat（RPC）とMulti-Relational Multi-Party Chat Corpus（MRMP）の性質を分けて検証することが提案されています。実験049では約5M parameter、500 stepの短い探索で、RPCだけを含む条件はRPC validation、MRMPだけを含む条件はMRMP validationで強くなる傾向を確認しました。実験050では約20M parameterでRPCとMRMPを同時に含めた条件を評価し、両方のvalidationへ改善が移ることを確認しましたが、source単独の差を同じモデル規模で比較できていません。

実験062では、同じ20M級モデル、同じ1M Tokenの学習列、同じ2,500 step、同じ乱数seedで、RPCを約10%含む条件とMRMPを約10%含む条件を比較します。どちらもFineWeb2 Edu Japaneseを約80%、医療データを約10%、対象の会話sourceを約10%含みます。これにより、会話Tokenの総量と一般文書・医療文書の比率を揃えたうえで、RPCとMRMPの違いを調べます。

事前の仮説は、RPC条件ではRPC validation lossが、MRMP条件ではMRMP validation lossが、それぞれ相手条件より低くなることです。RPCは1対1の比較的長い会話や話題の継続に、MRMPは複数話者、mention、短い相づちや発話交替に寄与すると予想します。片方のsourceで他方のvalidationも改善する可能性はありますが、両方を同時に含む実験050より専門化し、もう一方の会話形式への転移が弱くなる可能性があります。固定chat-testのEOS、生成長、Token overlapだけでは自然さを断定できないため、各stepの生成本文と5領域のvalidation lossを併せて確認します。

## 再現条件

実験開始前の基準commitは`c6cfe6f2840608da3bbaf670b6519720752d35cf`です。実験用の設定とColab wrapperを追加した実行コードcommitは`f299cf7`（`exp: prepare 062 source-specific pretraining`）としてpush済みです。RPC configのSHA-256は`ab4233f586500e067f351e90daac3b4c239d1a6a9d44f40f8e809d5602ef5a6b`、MRMP configは`3601571798dfe7c4cf57f03aeca6102e45e0a4f61d6515e3517c5b543b90f1c7`です。実行用wrapperは`scripts/colab_bootstrap_062.py`、回収用scriptは`scripts/colab_package_062.py`です。学習はこのcommitのコードと設定で実行します。

モデルは`dim=384`、10層、6 heads、context length 256、RoPE、LayerNorm、SwiGLUで、実測約19.3M parameterとなる構成です。学習はbatch size 8、2,500 step、AdamW、learning rate 3e-4から3e-5、warmup 300、weight decay 0.1、seed 42、validation・生成間隔100 step、checkpoint間隔500 stepです。Colabの同一T4 runtime上でRPC条件を先に、MRMP条件を後に実行します。GPUが割り当てられない場合や途中停止した場合も、停止理由と回収できた成果物をこのノートへ残します。

RPCのtrain Token列は999,974 Token、SHA-256は`24b61c79c6144e74e8e0598d54182c7a8f109ab75fda0e7d79eea90d68c268b7`です。MRMPのtrain Token列は999,978 Token、SHA-256は`9dacab60e0b483526f9f61f2f4254290c86c48a24a4c13abcb2922b38a1c100c`です。general validationは`mixed-ja-80-10-10-v2-general-val.bin`（SHA-256 `c4698596cbcd1c2f06507f9f2d4c3876cc59e2e081d448823ae97e36edb62db4`）、RPC validationは948,172 Token（SHA-256 `30f8b66828b7a5c1171024af220613cc79d066089a4c09896456112f8754491c`）、MRMP validationは156,475 Token（SHA-256 `9431d4d7432f89f69d9656bf2c3eea7c18dbcb94e84467060cba3fa8d9445623`）です。Tokenizerはvocab 4,096の`mixed-ja-80-10-10-v2-unigram.model`で、SHA-256は`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`です。

学習コマンドはColab wrapperから次の2つを順に実行します。

```bash
python scripts/train_torch.py --config configs/issue1-rpc-20m-colab-2p5k.toml --device auto
python scripts/train_torch.py --config configs/issue1-mrmp-20m-colab-2p5k.toml --device auto
```

学習途中の生成は`artifacts/samples/issue1-rpc-20m-colab-2p5k/`および`artifacts/samples/issue1-mrmp-20m-colab-2p5k/`へ保存します。軽量なmetrics、summary、checkpoint metadata、生成本文はGit管理へ追加します。重い`.pt`本体はGitHubへ直接追加せず、回収後にSHA-256をmanifestへ記録します。学習後はbest checkpointをローカルへ回収し、general、conversation、medical、RPC、MRMPの5領域と、48例の固定chat-testを同じ評価手順で実行します。

## 成功条件

2条件が同じモデル構造、学習量、seed、GPUで2,500 stepまで完走し、NaN、OOM、shape error、Token列不足が発生しないことです。完走後、各条件でbest checkpoint、best step、5領域のlossとperplexity、固定chat-testのEOS・生成長・Token overlap、各stepの生成本文を保存します。差が小さい場合も失敗とはせず、今回のToken量とモデル規模ではsource差を検出できなかった結果として記録します。

## 実験中の記録

開始前のcommit、bundleのhash照合、Colab sessionの状態、各条件の開始・終了、GPUやPyTorchの情報、学習時間、NaNやOOMなどの異常、生成本文の回収状況を節目ごとに追記します。悪い生成、空に近い生成、文脈に合わない生成も削除しません。

2026年9月5日、実験用bundleを`/tmp/small_llm-colab-062-b82b334.tar.gz`として作成しました。bundleのサイズは約3.8MB、SHA-256は`c02ec8a56756ea66fffca3864cd147d72977c67bd6e1797d21289a2779e26e30`です。設定、学習コード、Tokenizer、RPC/MRMP train Token列、general・RPC・MRMP validation Token列を含め、checkpoint本体は含めていません。

Colab session `exp062-20m-rpc-mrmp`はT4でREADYになりました。まだbundleのuploadと学習開始前であり、既存sessionの成果物は変更していません。次にbundleをuploadし、wrapperによる必須ファイルと入力hashの照合を通過してから学習を開始します。

bundle upload後、wrapperは設定、学習コード、Tokenizer、RPC/MRMP train Token列、general・RPC・MRMP validation Token列のhashを照合しました。照合値は開始前に記録した値と一致し、RPC条件、MRMP条件の順に同じT4 runtimeで実行しました。RPC条件は155.06秒、MRMP条件は152.97秒で、いずれも2,500 stepまで完走しました。NaN、OOM、shape error、Token列不足は発生していません。Colab側の軽量archiveは68ファイル、17,403 bytes、SHA-256 `4274232224d586549fb4880b2641c8acbb5b37cb33ab3f9a77052c7d89b2a789`でした。

両条件ともparameter数は19,308,032、PyTorchは2.11.0+cu128、CUDAは12.8、GPUはTesla T4です。GPU総メモリは15,637,086,208 bytes、peak allocatedは787,465,728 bytes、peak reservedは834,666,496 bytesでした。各条件でstep 0からstep 2,500まで100 step間隔の生成本文、500 step間隔のcheckpoint metadata、metrics、summaryを回収しました。best checkpointはいずれもstep 1,600でした。RPCのbest checkpointは77,267,142 bytes、SHA-256 `4a9bd751205626962508edb2e22cb62edb709c7e176764fc51884d8d9871426d`、MRMPは同サイズでSHA-256 `b55fb95638122b081b2d89c2c40c37929dadf6ef591e1a6622eec3e65e921426`です。回収manifestは`artifacts/checkpoints/exp062-manifest.json`、SHA-256 `e8ad4d4e4a269012fd903219292ac14a42e159af629be8e33a33315666c4902b`です。

学習途中の固定prompt `今日なにしてた？` に対するstep 2,500の出力は、RPC条件では日本語に混じって英語風の技術語や崩れた固有名詞が連続し、MRMP条件では日本語の文らしい断片が増えた一方で、質問への直接的な応答にはなっていませんでした。具体的な全文は各条件の`artifacts/samples/`へ保存しており、品質が低い出力も削除していません。

学習終了後、best checkpointをローカルCPUへ回収し、同じ5領域を20 batchずつ評価しました。validation lossとperplexityは次のとおりです。

| 条件 | general | conversation | medical | RPC | MRMP |
| --- | ---: | ---: | ---: | ---: | ---: |
| RPC | 6.2070 / 496.19 | 3.2924 / 26.91 | 3.4261 / 30.76 | 3.1567 / 23.49 | 4.0776 / 59.00 |
| MRMP | 6.2120 / 498.71 | 3.9332 / 51.07 | 3.4176 / 30.50 | 3.9311 / 50.97 | 2.7290 / 15.32 |

仮説どおり、RPC条件はRPC validationで、MRMP条件はMRMP validationで明確に低いlossになりました。RPC条件のRPC lossはMRMP条件より0.7744低く、MRMP条件のMRMP lossはRPC条件より1.3486低くなっています。これは、RPCの1対1会話・話題継続と、MRMPの複数話者・短い発話交替が、同じ「会話データ」として一括りではなく、異なる分布としてモデルに学習されたことを示します。RPC条件はconversation全体でも低く、MRMPのvalidationへも一定の転移がありました。MRMP条件はgeneralでわずかに不利でしたが、medicalではわずかに低く、今回の差だけから一般能力や医学知識の優劣を決めることはできません。

固定chat-test v1の48例も、同じ選択manifest、seed 42、temperature 0.8、top-k 40、最大64 Tokenで評価しました。RPC条件はEOS到達48/48、平均生成長11.50 Token、Token overlap F1 0.0666でした。MRMP条件はEOS到達48/48、平均生成長8.29 Token、F1 0.0451でした。RPC条件のF1はMRMP条件より高いものの、両方とも早期EOSが多く、短く終わることでoverlapが変動しているため、自然な会話能力の証拠とは扱いません。各例のprompt、reference、completionは`artifacts/evaluations/issue1-rpc-20m-colab-2p5k-chat-test-v1.json`、同TXT、およびMRMPの同名ファイルへ保存しました。

評価JSONのSHA-256は、RPC domainが`3271b6a7815ec4cbb7e04bd3ab9f42cc6cb7f30264a46f39a24cf19b4353b90b`、MRMP domainが`05f46d5b5517dd8ce0a75fb2fa2a68cc63649fbf00b137aea1d089b790164e7e`、RPC chatが`8969d2fa9eab8648dd277af5ca9118eca280beaa3935d6826276d082b33622b7`、MRMP chatが`843e137f2b1c0bc6c13b3b47cd072e7eda6c69a86647acf57f7c879deb7fca68`です。Colab session `exp062-20m-rpc-mrmp`は成果物回収後に停止し、停止後のactive sessionはありません。

## 実験終了後の結果と解釈

ここへ実測parameter数、runtime、peak memory、学習時間、最終および最良loss、5領域の比較、固定chat-testの集計、代表的な生成の観察を追記します。validation lossは次Token予測の指標であり、会話の自然さ、知識の正確さ、医学的安全性を直接意味しないことを明記します。RPCまたはMRMPのvalidationが良くなっても、そのsourceに含まれる表現を記憶した可能性があるため、held-out生成とsource別評価を分けて解釈します。

実験062では、同じ20M級モデルと学習量でも、RPC単独とMRMP単独で改善するvalidation領域が分かれました。この結果はIssue #1の「会話sourceの性質を分けて比較する」という方針を支持します。一方、固定chat-testは両条件とも短いEOS出力が中心で、自然な応答、話題適合、現代的な口語表現を十分に評価できる段階ではありません。今回はsource分布への適合性を示す探索結果として扱い、チャット品質の結論とは分けます。

## 次に試すこと

結果に応じて、RPCとMRMPの配分を変えたboth条件、両sourceを同じToken数へ揃えたSFT、または同じpretraining checkpointから応答部分だけを学習するSFTを選びます。いずれの場合も、今回の5領域lossと固定chat-testを再利用し、比較条件を増やしすぎず一度に一つの仮説を検証します。

今回の結果から、次はRPCとMRMPを同じToken比率で混ぜたboth条件を長く学習するより先に、両sourceを同じpretraining checkpointから応答部分だけ学習するSFT比較を優先します。pretrainingではsource固有の分布が明確に現れましたが、固定chat-testの早期EOSが残っているため、履歴を条件として応答を生成する学習がどこまで改善するかを確認する必要があります。SFTではsource別のデータ量を揃え、学習対象をassistant相当の応答Tokenに限定する条件を作ります。
