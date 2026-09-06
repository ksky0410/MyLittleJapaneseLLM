# 実験118：長めの応答を一定割合で学ぶSFT

## 学習完了時の記録

## 評価の計画

Exp117と同じRunpod A40、同じdomain Token列、同じ評価config、同じ一般会話48例・医療162例・生成seedを使い、best checkpoint（step 3,500）を評価する。一般会話のEOS、平均生成長、Token overlap F1、医療のEOS、平均生成長、回答形式抽出、完全一致、Token overlap F1を比較し、生成本文の話題適合性と反復も確認する。評価JSON、全文TXT、評価ログは全件保存する。

## 評価結果と判断

Exp118 best（step 3,500）のdomain lossはFineWeb2 2.906245、general 4.039475、conversation 1.984228、medical 1.870503だった。Exp117 bestの2.903175、4.041337、1.992968、1.871806と比べ、FineWeb2は悪化したが、general・conversation・medicalは改善した。

一般会話48例はEOS 48/48、平均生成長10.208 tokens、Token overlap F1 0.234298だった。Exp117 bestのEOS 48/48、平均10.000 tokens、F1 0.246358よりF1が低下した。出力には「食べてみたい!笑」のように自然な短い反応もあるが、「お昼ごはん食べました？」に「洗濯はなさんだりしないですか?」、「皆さんダイエットとかなさってますか？」に「大阪で!」のように話題が外れる応答が残った。長文samplingによって一般会話の自然さが改善したとは判定しない。

医療162例はEOS 160/162、平均生成長45.944 tokens、Token overlap F1 0.338486、回答形式抽出162/162、正解の完全一致27/162（16.67%）だった。Exp117 bestと比べてF1は0.250300から大幅に上がったが、完全一致は31/162から27/162へ低下し、誤答理由を長く生成する例が増えた。したがって、医療F1の上昇も正確性の改善とは解釈しない。

長文samplingの仮説は、validation lossと医療の表面的なToken overlapには効果があったが、一般会話の話題適合性と医療完全一致には効果がなく、自然な日本語の主条件としては棄却する。Exp118は比較用の成果物として保存し、次はExp117 bestを基盤に戻す。SFTデータをさらに反復する前に、未使用FineWeb2日本語データを追加して事前学習を継続し、一般的な日本語の表現力と知識を増やす。

評価成果物は、[domain評価](../../artifacts/evaluations/exp118-best/domains.json)、[一般会話JSON](../../artifacts/evaluations/exp118-best/general-chat.json)、[一般会話全文](../../artifacts/evaluations/exp118-best/general-chat.txt)、[医療JSON](../../artifacts/evaluations/exp118-best/medical-chat.json)、[医療全文](../../artifacts/evaluations/exp118-best/medical-chat.txt)、[評価ログ](../../experiments/results/exp118/evaluation.log)に保存した。SHA-256はdomain JSON `53c55fb777500cad8a69d84d389a4dd8174e1d12f136ab027475fff145d350e7`、一般会話JSON `b218a7ce779fcb90736ccf69774781b94cb6df40644e44fa155e049bc36dec61`、一般会話TXT `7c040f21ec569e3eb1ce8bc34cd25737985c9c6dc5900224029839346cf577f9`、医療JSON `2950e3dfb319d8464bec4e6fabc168f76f93d752fb30947226faea5f53e06060`、医療TXT `111b17c68507e127b9bd393b8fa4c0b1b327053792f1677417cbe3840874d5e0`、評価ログ `906ca1640d8f5fe0b57d76a1a47c0e687833c8483bf3e9255fc63d05901b8da2`である。

step 4,000までNaN、OOM、shape errorなく完走した。step 4,000のvalidation lossは2.720873、perplexityは15.193575、経過時間は353.72秒だった。最良checkpointはstep 3,500で、validation lossは2.720545、perplexityは15.188603だった。Exp117 bestの2.728217より0.007672低く、validation上は改善した。

最良重み`best.pt`のSHA-256は`ec8d2a661a715c299ca13067c812da82b2778086bd5bb9e5ac8e0005d025da9c`である。最大GPUメモリ使用量はallocated 1,490,586,112 bytes、reserved 1,598,029,824 bytesだった。metrics 17点、生成サンプル17本、step 4,000 metadata、学習ログを保存した。固定prompt「今日は天気がいいですね。」へのstep 4,000出力は「こちらこそ、こんばんは!」で、文章は短く完結したが、天気への意味的な反応とは言いにくい。固定評価で自然さを判定する。

## 実施前の計画

### 目的

Exp117では、RPCのresponse token予算を増やし、MRMPを減らすことでFineWeb2、general、conversation、medicalのdomain lossはすべて改善した。しかし、固定一般会話48例のToken overlap F1はExp116の0.249142から0.246358へわずかに低下し、医療の完全一致も33/162から31/162へ低下した。会話データの量比だけでは自然な日本語の改善につながらないため、今回は同じExp117データの中から、24 Token以上の応答を各batchへ一定割合で供給する学習経路を試す。

### 仮説

Exp117の最終SFTデータは平均応答長14.48 Token、24 Token以上の例12.60%、32 Token以上の例3.99%で、通常のランダムsamplingでは短い相槌や定型応答が大半のbatchを占める。各batch 8例のうち2例を24 Token以上から選ぶ`long-response-ratio=0.25`にすると、長い応答の比率を実質的に高め、直前の話題を参照する文や複数文の生成を学びやすくできると予想する。

ただし、過去の20M実験では長文比率の改善がseed 42でしか再現せず、seed 123・777では固定chat F1が低下した。したがって、今回は改善を前提にせず、Exp117との差を「長文samplingの効果」として記録する。長くなっただけで話題逸脱や反復が増える場合は失敗と判断する。

### 比較条件

Exp117 bestを初期値とし、モデル、Tokenizer、SFTデータ、validation、rehearsal、学習率、4,000 step、評価方法を固定する。変更は`long-response-ratio=0.25`と`long-response-min-tokens=24`、seed 118、出力先だけである。

- 初期checkpoint：Exp117 best、step 3,500、SHA-256 `333456ca779477bba5191e1acce5414a9a48b4d7acb1336207f97a8da301ad20`
- モデル：50,207,616 parameters、12 layers、dimension 576、9 heads、context 256、RoPE、LayerNorm、SwiGLU、vocab 4096
- 学習データ：`artifacts/sft/issue1-conversation-rebalance-medical-answer-focus-v2/train.npz`、SHA-256 `30d4c6de43391fcfedfc46966067d09c0f9f86d41a04aa15060d64214bd09e26`
- データ分布：130,784例、response token 1,894,122、平均応答長14.48 Token、24 Token以上12.60%、32 Token以上3.99%
- validation：`artifacts/sft/issue1-general-medical-concat-v1/validation.npz`、SHA-256 `95b6729ea46821d247ced049a0f06eef607c2d5ceb8e76cbcb8d337bebd8ad35`
- rehearsal：`artifacts/tokens/mixed-ja-80-10-10-v2-train.bin`、ratio 0.2、SHA-256 `d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`
- Tokenizer：`artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model`、SHA-256 `5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`
- 設定：`configs/issue1-exp117-long-response-sft-runpod-4k.toml`

設定ファイルのSHA-256は`2f3ca624f38cd9a03e3c850a71cadd41bfb44fd2f1cc165b944cfe18269f3d76`である。

### 成功・失敗の判定

4,000 stepをNaN、OOM、shape errorなく完走し、250 stepごとのmetricsと生成全文を保存できれば実装上の成功とする。性能面では、Exp117 bestと比べて一般会話のF1または実際の話題適合性が改善し、平均生成長だけでなく具体的な応答が増えることを期待する。domain lossだけが改善する場合、あるいは出力が長くなるだけで文脈逸脱・反復・誤答が増える場合は自然さの改善と認定しない。

### 実行コマンド

```bash
cd /workspace/exp100
PYTHONUNBUFFERED=1 PYTHONPATH=scripts uv run python scripts/train_sft_torch.py \
  --config configs/issue1-exp117-long-response-sft-runpod-4k.toml \
  --base-checkpoint artifacts/checkpoints/issue1-exp116-conversation-rebalance-sft-runpod-4k/best.pt \
  --train-data artifacts/sft/issue1-conversation-rebalance-medical-answer-focus-v2/train.npz \
  --validation-data artifacts/sft/issue1-general-medical-concat-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-exp117-long-response-sft-runpod-4k \
  --samples-dir artifacts/samples/issue1-exp117-long-response-sft-runpod-4k \
  --device cuda --sample-template conversation --sample-speaker-a DA --sample-speaker-b DC \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin --rehearsal-ratio 0.2 \
  --long-response-ratio 0.25 --long-response-min-tokens 24
```

## 学習中の記録

### 2026-09-07：step 1〜1,750

Runpod A40上でExp117 bestから長文sampling条件のSFTを開始した。step 1のvalidation lossは2.728228、step 250は2.730192、step 500は2.729358、step 750は2.728031、step 1,000は2.726478、step 1,250は2.730539、step 1,500は2.727325、step 1,750は2.726550だった。step 1,000でExp117 bestの2.728217を一時的に下回り、step 1,250で反発した後、step 1,750で再び改善した。step 1,750のlearning rateは3.394e-6、経過時間は154.48秒である。

全stepで`long_response_ratio=0.25`、`long_response_min_tokens=24`がmetricsへ記録されている。NaN、OOM、shape errorは発生していない。step 0から1,750までの生成文はRunpod上に保存されており、完了後に全件回収する。

### 2026-09-07：step 2,000〜3,000

step 2,000のvalidation lossは2.725745、step 2,250は2.725349、step 2,500は2.724989、step 2,750は2.724544、step 3,000は2.723329だった。step 3,000時点でExp117 bestの2.728217を0.004887下回り、学習後半にかけて改善が続いている。step 3,000のlearning rateは1.227e-6、経過時間は265.87秒だった。NaN、OOM、shape errorは発生していない。

学習開始後は少なくとも1,000 step以内ごとにvalidation loss、perplexity、learning rate、経過時間、生成文、警告、設定変更を追記する。悪い生成や短すぎる生成も削除しない。
