# 実験055：20M SFTのresponse末尾EOS loss ablation

## 開始前の計画

実施日は2026-09-05、担当者はCodexです。Issue #1の会話SFTを進めた049〜054では、話者境界を保ったresponse-only maskとrehearsalを使いました。054ではconversation形式のsample promptにより形式上の早期終了は改善しましたが、短い会話応答が挨拶へ寄りやすく、EOSの学習が強すぎる可能性が残っています。

055では、実験054と同じ20M base、ratio 0.50、会話NPZ、rehearsal Token列、seed、optimizer、学習率、3,000 stepを使い、response末尾EOSをSFT lossの対象から除く条件を追加します。054のEOS込み条件を比較対象として再利用し、変更をEOS maskだけに限定します。

仮説は、EOSを学習しない条件では短すぎるcompletionが減り、実在話者IDと履歴を使った生成の内容が続きやすくなるというものです。一方、EOSを除くと応答終了を学習できないため、長すぎる生成や会話境界の無視が増える可能性もあります。validation lossだけでなく、EOS数、平均生成長、固定48問、生成全文、5 domain loss、人手レビューを比較します。

## 使用条件と再現方法

モデルは19,308,032 parameters、dim 384、10層、6 heads、context 256、RoPE、LayerNorm、SwiGLUです。初期checkpointは実験050の`artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt`、SHA-256 `326e30b6b6480c76a0dc468d79f96aeb79e6d844a475d80b86caf27996a86751`です。設定は`configs/issue1-both-20m-rehearsal-ratio050-noeos-colab-3k.toml`で、学習中sampleは`DA`と`DC`の実在話者IDを使うconversation形式です。

新しいCLI引数`--exclude-eos-from-sft-loss`を`scripts/train_sft_torch.py`へ追加し、mask処理のテストを追加します。Colab wrapperは`scripts/colab_bootstrap_055.py`、bundle連結は`scripts/colab_concat_055.py`、archive化は`scripts/colab_package_055.py`です。学習開始前にコード、config、bundle hash、連結scriptをcommitしてpushします。

## 実行コマンド

```bash
uv run pytest -q
tar -czf /tmp/exp055_bundle.tar.gz \
  configs/issue1-both-20m-rehearsal-ratio050-noeos-colab-3k.toml \
  scripts/train_sft_torch.py scripts/train_torch.py scripts/_common.py \
  src artifacts/tokenizer/mixed-ja-80-10-10-v2-unigram.model \
  artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.pt \
  artifacts/checkpoints/issue1-both-20m-colab-2p5k/best.json \
  artifacts/sft/chat-v1-context256/train.npz \
  artifacts/sft/chat-v1-context256/validation.npz \
  artifacts/tokens/mixed-ja-80-10-10-v2-train.bin
split -b 45m -d -a 2 /tmp/exp055_bundle.tar.gz /tmp/exp055_bundle_part_
split -b 17m -d -a 1 /tmp/exp055_bundle_part_05 /tmp/exp055_last_part_
colab new --session exp055-20m-noeos-sft --gpu T4
colab exec --session exp055-20m-noeos-sft --timeout 120 --file scripts/colab_concat_055.py
colab exec --session exp055-20m-noeos-sft --timeout 3600 --file scripts/colab_bootstrap_055.py
colab exec --session exp055-20m-noeos-sft --timeout 120 --file scripts/colab_package_055.py
```

bundle連結時にbytesとSHA-256を検証し、学習後は軽量archive、manifest、best checkpointを回収してsessionを停止します。重い`.pt`はGitへ追加せず、metadataとノートにhashを残します。生成TXT、metrics、評価JSON/TXTはGitの追跡対象です。

## 成功条件

EOS除外条件が3,000 stepまで完走し、NaN、OOM、shape error、mask対象不足、checkpoint reload errorがないことです。step 0から3,000まで100 step間隔の生成TXT、500 step間隔のmetadata、5 domain評価、固定chat 48例を保存します。

## 結果と解釈

学習完了後にEOS込みの054とEOS除外の055を、SFT validation、通常domain、EOS・生成長、固定chat F1、生成の反復・長文化・話者境界の観点で比較します。EOS除外が良い場合も、評価データの重複と20Mモデルの制約を明記します。

## 次に試すこと

EOS除外で長さと自然さが改善すれば、EOSを別重みで学習する中間条件や、response長別のloss weightingを試します。改善しなければ、EOS込みratio 0.50を基準として50M級モデルやLoRA/QLoRAへ進みます。
