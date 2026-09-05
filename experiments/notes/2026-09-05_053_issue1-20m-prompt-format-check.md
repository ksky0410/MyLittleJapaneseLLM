# 実験053：20M SFT checkpointのprompt形式切り分け

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験052ではratio 0.10と0.50の学習中サンプルが、step 500以降に入力の「今日なにしてた？」だけを残して終了しました。一方、同じcheckpointを実際の会話履歴と話者markerを含む固定48例で評価するとEOSは48/48で、ratio 0.50のToken overlap F1は0.2020でした。学習中サンプルのpromptがSFTで使った形式と一致していない可能性があるため、raw promptと`<|startofconversation|>`・話者markerを付けたconversation promptを同じcheckpointで比較します。

仮説は、raw promptでは会話SFTの入力分布から外れるため早期EOSや無関係な生成が増え、conversation promptでは応答らしい続きを生成しやすくなるというものです。これは新しい学習をせず、同じ重みへ同じ8個の固定promptを二つの形式で与える切り分け実験です。評価結果だけでなく、8個すべての生成本文をJSON/TXTへ保存します。prompt形式が原因なら、次の長時間SFTでは学習中サンプルにもconversation形式を使用します。

## 使用条件と再現方法

対象は実験052のratio 0.10とratio 0.50のbest checkpointで、モデルは各19,308,032 parameters、vocab 4,096、dim 384、10層、6 heads、context 256、RoPE、LayerNorm、SwiGLUです。checkpointのSHA-256はratio 0.10が`3b097acdceebdc25c3c42666e611b391571b3111fb244e24d0eca03415f6dc23`、ratio 0.50が`9920c1a78c1665c6ff93cc293b101824f565c1d582ea5fa66f0dd8f45c801b21`です。Tokenizer、config、seedは実験052と同じで、8 prompt、max new tokens 64、temperature 0.8、top-k 40、seed 42を使います。評価はローカルPyTorch CPUで行うため、重みの学習条件やColabのruntimeは変更しません。

実験開始時点のcommitは実験052成果物をpushした`8f10b90`です。新しい評価scriptは`scripts/evaluate_torch_prompt_set.py`、テストは`tests/test_evaluate_torch_prompt_set.py`です。実行前にこれらをcommitし、テストを通過させます。

## 実行コマンド

```bash
uv run pytest -q
uv run python scripts/evaluate_torch_prompt_set.py \
  --config configs/issue1-both-20m-sft-torch-colab-1k.toml \
  --checkpoint artifacts/checkpoints/issue1-both-20m-rehearsal-ratio010-colab-1k/best.pt \
  --template raw --device cpu --seed 42 --max-new-tokens 64 \
  --output artifacts/evaluations/issue1-both-20m-rehearsal-ratio010-prompt-raw.json \
  --text-output artifacts/evaluations/issue1-both-20m-rehearsal-ratio010-prompt-raw.txt
```

同じコマンドでratio 0.10をconversation、ratio 0.50をraw、ratio 0.50をconversationへ変更し、出力名も条件に合わせます。学習は実施しません。

## 成功条件

二つのcheckpointについてrawとconversationの計4評価が完了し、8 prompt分の生成本文、EOS数、平均completion Token数、カテゴリ別集計が保存されることです。checkpoint reload errorやpromptのToken化エラーが出た場合も、そのまま失敗として追記します。

## 結果と解釈

評価完了後に、rawとconversationのEOS・空出力・平均生成長・生成本文を比較し、学習中サンプルの早期終了がprompt形式で説明できるかを記録します。説明できない場合は、EOS lossの重み付け、response-only maskの構造、generation samplerの順に調べます。

## 次に試すこと

conversation形式が明確に良ければ、学習configのsample promptを会話形式へ変更し、ratio 0.50を3,000 step程度まで長く学習します。差が小さければ、SFT学習例の入力と固定評価の話者IDを揃える実験を追加します。
