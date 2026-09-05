# 実験032：Google Colab CLIのGPUランタイム確認

## 開始前の計画

実施日時は2026-09-05、担当者はCodexです。利用可能になったGoogle Colab CLIを、将来の50M級以上の学習や長時間学習に使えるか確認します。現在の本体学習はApple Silicon向けMLX実装なので、ColabのGPU上でそのまま実行できるとは仮定しません。まず一時GPU VMの作成、認証、PyTorchの利用、CUDA GPUの認識、簡単な行列積だけを検証します。

今回の仮説は、Colab CLIからGPUランタイムを再現可能なスクリプトで起動でき、PyTorchとCUDAが利用できるなら、次の段階でPyTorch/CUDA版の学習バックエンドを追加する価値があるというものです。GPUが取れない、認証できない、またはPyTorchが使えない場合は、今回のMacBook MLX実装を継続し、Colab移行は保留します。

## 実行前の再現情報

ローカルのColab CLIは`0.6.0`です。`colab run`はGPU種別としてT4、L4、G4、H100、A100を受け付けます。CLIのヘルプ確認では、`colab run --gpu T4 SCRIPT`が一時VM上でスクリプトを実行し、終了後にVMを解放する仕様でした。確認時点でリモートのアクティブセッションはありませんでした。

検証スクリプトは`scripts/colab_probe.py`です。行列積は2048×2048、10回、PyTorchのCUDA device上で実行します。今回の操作はモデル学習ではなく、学習環境の確認です。Colabランタイムを保持せず、処理終了後に解放します。

実行予定コマンドは次のとおりです。

```bash
colab run --gpu T4 scripts/colab_probe.py
```

成功基準は、GPUランタイムが作成され、スクリプトが`torch_available: true`、`cuda_available: true`、GPU名、CUDAバージョン、処理時間、最大メモリ使用量を出力して終了することです。失敗した場合も、エラー内容と移行判断をこのノートへ追記します。

## 実験中の記録

2026-09-05、`colab run --gpu T4 scripts/colab_probe.py`を実行しました。Colab CLIは`run-e5b449`セッションを作成し、スクリプト終了後に正常解放しました。セッションはLinux x86_64、Python 3.13.15でした。実行中のアクティブセッションを残していません。

## 結果と解釈

GPUランタイムの検証は成功しました。PyTorchは`2.11.0+cu128`、CUDAは`12.8`、GPUは`Tesla T4`、compute capabilityは7.5、`nvidia-smi`上のメモリは15,360MiBでした。2048×2048の行列積10回は0.195305秒、最大割当メモリは75,628,544 bytes（約72.2MiB）でした。`torch_available: true`と`cuda_available: true`を確認できたため、50M級または長時間学習ではColab GPUを使う価値があります。

ただし、現在の学習コードはMLX専用であり、今回のT4でそのまま動くわけではありません。したがって、今回の結果は「Colabで学習が成功した」という意味ではなく、「PyTorch/CUDA版を追加すればColabで実行できる環境が確認できた」という意味です。MacBook側の20M学習は約1,408秒でしたが、行列積だけではモデル学習速度を推定できません。次の移行実験では、同じモデル構造・Token列・seed・学習条件をPyTorchへ実装し、stepあたりの時間、validation loss、生成結果、チェックポイントhashを比較します。

今回の出力には、GPU名、CUDA・PyTorchのバージョン、処理時間、メモリ使用量が含まれています。実行ログ自体はターミナルで取得し、スクリプトは`scripts/colab_probe.py`としてcommitします。

## 次に試すこと

GPUが利用できた場合は、MLXと同じdecoder-only TransformerをPyTorchへ移植し、まず同一20Mモデル・短いstep数でMacとColabのloss曲線と速度を比較します。その後、50M級や長いToken予算でColabを使うか判断します。GPUが利用できなかった場合は、MacBookでRoPEまたはWikipedia追加実験を優先します。
