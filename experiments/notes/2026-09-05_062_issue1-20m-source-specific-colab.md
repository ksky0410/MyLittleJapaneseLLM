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

## 実験終了後の結果と解釈

ここへ実測parameter数、runtime、peak memory、学習時間、最終および最良loss、5領域の比較、固定chat-testの集計、代表的な生成の観察を追記します。validation lossは次Token予測の指標であり、会話の自然さ、知識の正確さ、医学的安全性を直接意味しないことを明記します。RPCまたはMRMPのvalidationが良くなっても、そのsourceに含まれる表現を記憶した可能性があるため、held-out生成とsource別評価を分けて解釈します。

## 次に試すこと

結果に応じて、RPCとMRMPの配分を変えたboth条件、両sourceを同じToken数へ揃えたSFT、または同じpretraining checkpointから応答部分だけを学習するSFTを選びます。いずれの場合も、今回の5領域lossと固定chat-testを再利用し、比較条件を増やしすぎず一度に一つの仮説を検証します。
