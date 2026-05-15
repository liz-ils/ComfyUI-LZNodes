# ComfyUI-LZNodes

ComfyUI-LZNodes は [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 向けのカスタムノード集です。複雑になりがちなノードの配線を簡略化し、画像生成プロセスをより快適にするために設計されています。

制作者: Liz

[English Version](README.md)

## 主な機能

この拡張機能は、よくあるワークフローをひとまとめにする便利なノードを提供します。

*   **LZ パイプシステム (`LZPipePack` / `LZPipePackXL` / `LZPipeUnpack` / `LZPipeMerge` / `LZPipeInfo`)**: `MODEL`、`CLIP`、`VAE`、`CONDITIONING`、`LATENT`、および各種生成パラメータを1本の `lz_pipe` にまとめることで、ノード同士がスパゲッティのように絡まるのを防ぎます。
*   **スマートなローダー群**:
    *   `EZCheckpointLoader`: チェックポイントの読み込みと、ポジティブ/ネガティブプロンプトのエンコードを1つのノードで同時に行います。
    *   `LZSimpleCheckpointLoader`: モデルの各コンポーネントを取り出すことに特化した軽量なローダーです。
    *   `LZLoRAStacker`: 最大10個までのLoRAを簡単にスタック可能です。キャッシュ機能により高速な生成をサポートします。
    *   `LZAnimaLoader`: Animaモデル専用のローダーで、拡散モデル、テキストエンコーダー、VAEを個別に読み込みます。
*   **プロンプト管理**:
    *   `AdvancedPositivePrompt` / `AdvancedNegativePrompt`: 基本のプロンプト、アーティストタグ、クオリティタグなどを綺麗に分けて入力できます。内部で自動的に結合＆エンコードされます。
    *   `DualCLIPTextEncode`: ポジティブプロンプトとネガティブプロンプトを横並びで同時にエンコードするノードです。
    *   `LZPromptWeight`: プロンプトタグに重み構文 `(tag:weight)` を自動で付与します。
    *   `LZTagEditor`: 最大10個のタグを強度やON/OFFと共に視覚的に編集できます。
    *   `LZPromptReplaceSingle` / `LZPromptReplaceMulti` / `LZPromptReplaceString`: CSVファイルや改行区切りの候補リストからランダムに選択し、プロンプト内のプレースホルダーを動的に置換します。
*   **テキストユーティリティ**:
    *   `StringNode` / `StringConcatNode`: 基本的な文字列出力、および複数入力の結合ノードです。
    *   `LZTextPreview`: 文字列をノードUI上に直接表示します。
    *   `LZStringSanitize`: ファイル名に使用できない文字を除去・置換します。
    *   `LZStringSelect`: 複数の文字列入力からインデックスで1つを選択します。
*   **オールインワンのサンプリング (`LZKSamplerDecode`)**: KSamplerとVAEDecodeを1つのステップにまとめ、`lz_pipe` から必要なデータを直接引き出します。もしコンディショニングデータが接続されていなくても、入力されたテキストから内部で自動的にCLIPエンコードを行う賢い設計です。
*   **XY Plotシステム (`LZXYPlot` / `LZXYPlotSampler` / `LZXYSampler` / `LZXYGridOutput`)**: チェックポイント、LoRA、Sampler、Scheduler、プロンプトの変化をグリッドで比較するシステムです。置換キーや自動グリッド画像生成に対応しています。
*   **画像保存と詳細なログ記録**:
    *   `LZSaveImageAndLog`: 生成された画像（PNG/WEBP/JPG）の保存と同時に、シード値、ステップ数、さらにはチェックポイントのハッシュ値やプロンプトの全文まで、考えうる限りの詳細なメタデータを `.txt` ログファイルとして出力します。もちろん、PNGの不可視メタデータ領域（PNGInfo）への書き込みにも対応しています。
    *   `LZBatchSaveWithLabels`: XYラベルをファイル名やメタデータに埋め込んで一括保存し、自動的にバッチログも出力します。
    *   `LZAppendLogToCSV`: 生成パラメータをCSVファイルに追記し、スプレッドシートでの管理を可能にします。
    *   `LZLogReader`: `LZSaveImageAndLog` で出力した `.txt` ログを読み込み、パラメータをノード出力として復元します。
*   **モデルマージレシピ (`LZMergeRecipeRandom` / `LZMergeRecipeManual` / `LZMergeRecipeRandomAdvanced`)**: モデルマージツールと互換性のある19ブロック（IN00-IN08、M00、OUT00-OUT08）のマージレシピを生成します。
*   **プリセットLatent (`PresetEmptyLatentImage`)**: SDXLの一般的な解像度を使って、空のLatent画像を素早く作成できます。SDXL（4ch）とAnima（16ch）の両方の潜在形式に対応しています。

## ノード一覧

| カテゴリ | ノード名 |
|---|---|
| **Loaders** | EZCheckpointLoader, LZSimpleCheckpointLoader, LZLoRAStacker, LZAnimaLoader |
| **Prompt** | DualCLIPTextEncode, AdvancedPositivePrompt, AdvancedNegativePrompt, LZPromptWeight, LZTagEditor |
| **Text** | StringNode, StringConcatNode, LZTextPreview, LZStringSanitize, LZStringSelect, LZPromptReplaceSingle, LZPromptReplaceMulti, LZPromptReplaceString |
| **Latent** | PresetEmptyLatentImage |
| **Pipe** | LZPipePack, LZPipePackXL, LZPipeUnpack, LZPipeInfo, LZPipeMerge |
| **Sampling** | LZKSamplerDecode |
| **XY Plot** | LZXYPlot, LZXYPlotSampler, LZXYSampler, LZXYGridOutput |
| **IO** | LZSaveImageAndLog, LZBatchSaveWithLabels |
| **Log** | LZAppendLogToCSV, LZLogReader |
| **Merge** | LZMergeRecipeRandom, LZMergeRecipeManual, LZMergeRecipeRandomAdvanced |

## インストール方法

1. ComfyUIの `custom_nodes` ディレクトリに移動します。
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. このリポジトリをクローンします。
   ```bash
   git clone https://github.com/liz-ils/ComfyUI-LZNodes
   ```
3. ComfyUI を再起動してください。

## 使い方

インストール後、ComfyUIの新規ノード追加メニュー内の `MyCustomNodes` カテゴリ（例: `MyCustomNodes/Loaders`, `MyCustomNodes/Prompt`, `MyCustomNodes/Sampling` など）から呼び出すことができます。

*ヒント: `EZCheckpointLoader`を用意し、`LZPipePack` を通して `LZKSamplerDecode` に繋ぎ、最後に `LZSaveImageAndLog` で締めくくるのが、最も簡単で強力な使い方です。*

## ライセンス

このプロジェクトは MIT License の下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。
