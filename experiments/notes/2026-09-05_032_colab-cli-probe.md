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

未実施です。

## 結果と解釈

未実施です。

## 次に試すこと

GPUが利用できた場合は、MLXと同じdecoder-only TransformerをPyTorchへ移植し、まず同一20Mモデル・短いstep数でMacとColabのloss曲線と速度を比較します。その後、50M級や長いToken予算でColabを使うか判断します。GPUが利用できなかった場合は、MacBookでRoPEまたはWikipedia追加実験を優先します。
