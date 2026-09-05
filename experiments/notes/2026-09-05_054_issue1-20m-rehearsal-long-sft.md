# 実験054：20M rehearsal ratio 0.50の長時間SFT

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。実験052ではratio 0.10と0.50を1,000 step学習し、validationと固定chatの指標はratio 0.50がわずかに優勢でした。しかし、学習中のraw promptがSFT入力形式外だったため、step 500以降のサンプルだけではEOS崩壊を判断できませんでした。実験053ではconversation markerを付けると空出力がなくなりましたが、A/Bという未学習に近い話者IDのため挨拶へ偏りました。

054では実験052と同じ20M base checkpoint、Tokenizer、会話NPZ、rehearsal Token列、seed、optimizer、学習率を使い、ratio 0.50を1,000 stepから3,000 stepへ延長します。学習中サンプルにはtrain splitの先頭会話に実在する話者ID `DA` と `DC` を使い、`<|startofconversation|><|speaker:DA|>こんにちはー！<eos:3><|speaker:DC|>`という形式で保存します。

仮説は、ratio 0.50を長く学習すると会話validationと通常domain lossがさらに下がり、実在話者IDを使ったsample promptでも入力に対応する返答が増えるというものです。逆に、response-only SFTがEOSを強く学習している場合は、conversation形式でも早期終了が続くと予想します。054は052との単純な同条件比較ではなく、学習stepを増やした探索であるため、改善を一般的な最適設定とは断定しません。

## 使用条件と再現方法

モデルは19,308,032 parameters、dim 384、10層、6 heads、context 256、RoPE、LayerNorm、SwiGLUです。初期checkpointは`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`、SHA-256 `326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`です。train NPZ、validation NPZ、rehearsal Token列、Tokenizerは実験052と同じhashを使用します。設定は`configs/issue1-both-20m-rehearsal-ratio050-colab-3k.toml`です。

学習実行用scriptは`scripts/train_sft_torch.py`で、`--sample-template conversation --sample-speaker-a DA --sample-speaker-b DC`を追加します。Colab wrapperは`scripts/colab_bootstrap_054.py`、bundle連結は`scripts/colab_concat_054.py`、成果物回収archiveは`scripts/colab_package_054.py`です。新しいsample prompt処理とテストを含む基準commitをpushしてからbundleを作成し、bundleのbytesとSHA-256をこのノートへ追記してから学習を開始します。

基準commitは`e9a6e31`で、`configs/issue1-both-20m-rehearsal-ratio050-colab-3k.toml`のSHA-256は`21d71a92f3dca17f96334c81724ecc55fa21c1023991d4e849bd8169427c671c`、更新後の`train_sft_torch.py`は`7e81ffad0f713c0f916d174b918ffce54aad4cbf77fcd07cc295ca789230a85e`です。`/tmp/exp054_bundle.tar.gz`は271,721,074 bytes、SHA-256 `c133ce79bd086abe778a7430387d33a3b28fa8d028f1966b08a68821b4fc78b4`です。45MBのpart 5個と、残りを17MB、17MB、137KBへ分割した3個のpartを使い、連結scriptにはこのbytesとhashを固定しました。連結script更新後のSHA-256は`f658954bda14623c74e2860d747bf1e4b3a27afef215e8a3db29bd7da857a6c3`です。このhashをpushしてからColabへuploadします。

## 実行コマンド

```bash
uv run pytest -q
tar -czf /tmp/exp054_bundle.tar.gz \
  configs/issue1-both-20m-rehearsal-ratio050-colab-3k.toml \
  scripts/train_sft_torch.py scripts/train_torch.py scripts/_common.py \
  src artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.json \
  artifacts/sft/chat-v1-context256/train.npz \
  artifacts/sft/chat-v1-context256/validation.npz \
  artifacts/tokens/mixed-ja-80-10-10-v2-train.bin
split -b 45m -d -a 2 /tmp/exp054_bundle.tar.gz /tmp/exp054_bundle_part_
split -b 17m -d -a 1 /tmp/exp054_bundle_part_05 /tmp/exp054_last_part_
colab new --session exp054-20m-long-sft --gpu T4
colab exec --session exp054-20m-long-sft --timeout 120 --file scripts/colab_concat_054.py
colab exec --session exp054-20m-long-sft --timeout 3600 --file scripts/colab_bootstrap_054.py
colab exec --session exp054-20m-long-sft --timeout 120 --file scripts/colab_package_054.py
```

bundle連結後に元bundleのSHA-256を照合し、軽量archive、manifest、best checkpointを回収した後、Colab sessionを停止します。学習中の全生成TXT、metrics、checkpoint metadata、評価JSON/TXTはGitで追跡します。重い`.pt`はGitへ追加せず、hashだけをmetadataとノートへ残します。

## 成功条件

ratio 0.50が3,000 stepまで完走し、NaN、OOM、shape error、mask対象不足、checkpoint reload errorがないことです。step 0から3,000まで100 step間隔の生成TXT、500 step間隔のcheckpoint metadata、validation lossとtrain loss、runtime情報を保存します。学習後にratio 0.50のdomain評価と固定chat 48例を同じ評価設定で行い、052の1,000 step結果と比較します。

## 結果と解釈

学習完了後に、最終・best step、validation loss、general・conversation・medical・RealPersonaChat・MRMP loss、固定chatのEOS・生成長・F1、実在話者IDを使ったsample生成の内容を記録します。学習中の生成が悪い場合も削除せず、SFT入力形式、EOS、話者ID、学習量のどれが原因かを切り分けます。

## 次に試すこと

長時間学習で改善が見えれば、ratio 0.50を基準に短い応答と長い応答を分けたsampling、さらに大きいモデルのSFTへ進みます。改善が乏しければ、response-only maskでEOSを含める場合と含めない場合を比較し、その後にLoRA/QLoRAへ進みます。
