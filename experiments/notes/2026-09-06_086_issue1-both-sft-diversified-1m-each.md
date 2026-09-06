# 実験086：RPC・MRMPのSFTデータを各1M response tokenへ増やす

## 開始前の計画

実施日は2026年9月6日、担当者はCodexです。実験085では、082と同じSFTデータを10,000 stepまで反復すると、5領域validation lossと48例chat-test全体F1が改善しました。一方、Issue #1固定プロンプトでは「こんにちは」「こんばんは」「よろしくお願いします」への偏りが残り、同じbalanced subsetを長く見るだけでは口語応答の多様性が増えないことが分かりました。

現在の`issue1-both-balanced-v1/train.npz`は、RPCとMRMPから各385k response tokenを選んだ64,423例です。元のtrain NPZにはRPC約5.13M、MRMP770,975 response tokenがあります。086では、元のtrain NPZからRPCとMRMPを可能な範囲で同じ770,975 response tokenへseed固定で抽出し、`concat_sft_npz.py`で連結します。当初は各1,000,000 tokenを予定しましたが、MRMPの入力総量が770,975 tokenしかないため成立せず、RPC各770,975へ計画を修正しました。sourceごとの予算を等しくし、MRMP全量と同量のRPCを使うことで、現在の約2倍のresponse token、より多い会話例を使います。validation NPZと評価セットは変更しません。

仮説は、085のように学習stepを増やして同じ例を再利用するより、同じ10,000 stepで未観測の会話例を増やす方が、Issue #1固定プロンプトの定型挨拶への収束を弱め、意味に沿った相づち・質問応答・口語表現を増やすことです。モデル、base、learning rate、seed、rehearsal ratio、EOS loss weight、学習stepは085と完全に揃えます。成功条件は、085より5領域lossを大きく悪化させず、chat-test全体F1またはshort・medium・longの複数層を改善し、Issue #1固定プロンプトの応答が入力別に分かれることです。出力が長くなるだけ、EOSだけが維持されるだけ、挨拶の種類が変わるだけなら成功としません。強いLLMの蒸留や外部教師データは使いません。

## 再現条件

実験開始時点のGit commitは`c630321`です。作業ツリーには別作業由来の`README.md`、`web/`、`scripts/serve_chat.py`、`src/my_little_japanese_llm/chat.py`、`tests/test_chat.py`の未コミット変更がありますが、086では変更せず、コミットへ混ぜません。設定は`configs/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-1m-each.toml`です。

抽出前の入力は`artifacts/sft/issue1-rpc-full-v1/train.npz`と`artifacts/sft/issue1-mrmp-full-v1/train.npz`です。RPCの入力は315,584例・5,132,071 response token、MRMPの入力は81,382例・770,975 response tokenです。修正版ではRPC target 770,975、seed 8601、MRMP target 770,975、seed 8602で抽出し、出力を`artifacts/sft/issue1-both-balanced-770k-each-v1/train.npz`へ連結します。validationは`artifacts/sft/issue1-both-full-v1/validation.npz`を使います。

085と同じ50,207,616 parameter、dim 576、12層、9 heads、context length 256、RoPE・LayerNorm・SwiGLU、batch size 8、10,000 step、eval/sample interval 100、checkpoint interval 500、learning rate 5e-5から5e-6、warmup 100、weight decay 0.01、seed 123、rehearsal ratio 0.20、EOS loss weight 0.50です。baseは081 best checkpointで、SHA-256は`1e09a7386a630133503c12052f7701216d505cb3bca8765cade38c879ba5e8cb`です。Tokenizerは`5bde054fb91da54cbf56673a6d25b630399d95ec331049e5fa2af1a8d60731e4`、rehearsal Token列は`d74a1820f09582f40538a42d34d8e3057261329dccd00df109991f36f8df8090`、Issue #1 promptは`b538af4f00668e60e712aa796b1de5d51e0f677c8b9a19bc4445d40a90929594`です。

ローカルでのデータ作成コマンドは次のとおりです。

```bash
uv run python scripts/select_sft_npz.py --input artifacts/sft/issue1-rpc-full-v1/train.npz --output artifacts/sft/issue1-rpc-770k-v1/train.npz --manifest artifacts/sft/issue1-rpc-770k-v1/manifest.json --target-response-tokens 770975 --seed 8601
uv run python scripts/select_sft_npz.py --input artifacts/sft/issue1-mrmp-full-v1/train.npz --output artifacts/sft/issue1-mrmp-770k-v1/train.npz --manifest artifacts/sft/issue1-mrmp-770k-v1/manifest.json --target-response-tokens 770975 --seed 8602
uv run python scripts/concat_sft_npz.py --input rpc=artifacts/sft/issue1-rpc-770k-v1/train.npz --input mrmp=artifacts/sft/issue1-mrmp-770k-v1/train.npz --output artifacts/sft/issue1-both-balanced-770k-each-v1/train.npz --manifest artifacts/sft/issue1-both-balanced-770k-each-v1/manifest.json
```

SFTのローカル再現コマンドは次のとおりです。

```bash
uv run python scripts/train_sft_torch.py \
  --config configs/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-1m-each.toml \
  --base-checkpoint artifacts/checkpoints/issue1-both-50m-pretrain-5m-5k/best.pt \
  --train-data artifacts/sft/issue1-both-balanced-770k-each-v1/train.npz \
  --validation-data artifacts/sft/issue1-both-full-v1/validation.npz \
  --output-dir artifacts/checkpoints/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each \
  --samples-dir artifacts/samples/issue1-both-50m-sft-from-5m-two-pass-seed123-10k-770k-each \
  --lr-schedule-steps 10000 --eos-loss-weight 0.5 \
  --rehearsal-tokens artifacts/tokens/mixed-ja-80-10-10-v2-train.bin \
  --rehearsal-ratio 0.20 --sample-template conversation \
  --sample-speaker-a DA --sample-speaker-b DC --device mps
```

## 実験中の記録

当初のRPC・MRMP各1,000,000 response token抽出を実行したところ、RPCは61,587例・1,000,005 tokenを作成できましたが、MRMPは入力総量770,975 tokenのため`ValueError: 入力のresponse Token数が不足しています: 770975 < 1000000`で停止しました。RPC 1M出力は削除せず、当初計画の未使用成果物として残します。この失敗を受け、MRMP全量の770,975 tokenに合わせてRPCも770,975 tokenへ抽出する条件へ変更します。

修正版の抽出はRPC 47,518例・770,981 response token、MRMP 81,382例・770,975 response tokenとなり、連結後は128,900例・1,541,956 response tokenです。修正版train NPZのSHA-256は`001dc022a998abc5756f641b199988112db77ff42903485ff7a6fd6bd0e028a3`、manifestは`8ba77285ffed2a3738a45f6adf2ad7350056eebf9e1aedc55ee5d42d38a6a382`です。設定ファイルのSHA-256は`919d1acee3929091b399dc8863d1906b4097ee90abea57baedcf0a7990f2f726`、Colab bootstrapは`b8f64a1c48c63810a7327af9b00c3ae8923a48912252f1c92c03604e9ea26ada`、成果物回収スクリプトは`53a27481110a9d5af689221b2d509ca56f0e6a3f36a184e3d850ea8a375e9c19`です。修正版をbundle化してからColab T4へ送ります。学習中はmetricsと生成文を100 stepごとに確認し、1,000 stepを超えてノート更新を空けません。データ抽出が失敗した場合、入力不足、例数、response token数、ハッシュ不一致、OOM、validation悪化、Issue #1固定promptの挨拶偏重を削除せず記録します。

## 実験終了後の結果と解釈

ここに、抽出後のデータ件数・response token数、学習実績、best checkpoint、5領域評価、48例chat-test、Issue #1固定prompt全文、各artifactのSHA-256を追記します。085との差は、モデル条件が同じでSFTデータの多様性だけを変えたpaired comparisonとして解釈します。

## 次に試すこと

086で改善する場合は、SFTデータを増やす主線を維持し、さらに一般日本語pretrainingを10Mから20Mへ増やす条件を試します。改善しない場合は、単純なsubset拡大ではなく、会話履歴の切り詰め方、短文・質問・相づち・長文の層化、pretrainingとSFTの順序を見直します。いずれも蒸留を使わず、データ量と品質を分けて検証します。
