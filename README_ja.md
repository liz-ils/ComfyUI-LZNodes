# ComfyUI-LZNodes

ComfyUI-LZNodes は [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 向けのカスタムノード集です。複雑になりがちなノードの配線を簡略化し、画像生成プロセスをより快適にするために設計されています。

[English Version](README.md)

## 主な機能

この拡張機能は、よくあるワークフローをひとまとめにする便利なノードを提供します。

*   **LZ パイプシステム (`LZPipePack` / `LZPipePackXL` / `LZPipeUnpack` / `LZPipeMerge` / `LZPipeInfo`)**: `MODEL`、`CLIP`、`VAE`、`CONDITIONING`、`LATENT`、および各種生成パラメータを1本の `lz_pipe` にまとめることで、ノード同士がスパゲッティのように絡まるのを防ぎます。
*   **スマートなローダー群**:
    *   `EZCheckpointLoader`: チェックポイントの読み込みと、ポジティブ/ネガティブプロンプトのエンコードを1つのノードで同時に行います。
    *   `LZSimpleCheckpointLoader`: モデルの各コンポーネントを取り出すことに特化した軽量なローダーです。
    *   `LZLoRAStacker`: 最大10個までのLoRAを簡単にスタック可能です。キャッシュ機能により高速な生成をサポートします。
    *   `LZAnimaLoader`: Animaモデル専用のローダーで、拡散モデル、テキストエンコーダー、VAEを個別に読み込みます。
    *   `LZKrea2Loader`: Krea 2 アーキテクチャ(12B SingleStreamDiT + Qwen3-VL-4B テキストエンコーダー + Qwen-Image VAE)専用のワンノードローダーです。lz_pipe 対応、`weight_dtype` 選択可能、`negative_mode="auto"` で未使用時のネガティブを自動ゼロ化します(Krea 2 Turbo は CFG 1.0 前提のため、公式テンプレートと同一の挙動)。ComfyUI v0.26.0 以降が必要です。
*   **Anima Artist Mixer**:
    *   `LZAnimaArtistNode`: 統合ノード。画師チェーンの分割・エンコードと cross-attention 注入を1ステップで実行。`(MODEL, CONDITIONING, STRING)` を出力し、`positive_text` を直接パイプシステムに接続可能。
    *   `LZAnimaArtistPack` / `LZAnimaArtistCrossAttn`: 分割・エンコードと cross-attn 注入を分離したノード（上級者向け）。`LZAnimaArtistCrossAttn` も `positive_text` (STRING) を出力。
    *   `LZAnimaArtistOptions`: ブロック範囲、fusionモード、EMA平滑化、低ランク注入、static capture、anchor Q を細かく設定するための高度なオプションノード。
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
    *   `LZSaveStringToCSV`: 文字列の行をCSVファイルに保存・追記します（ヘッダはファイル新規作成時のみ書き込まれます）。
*   **オールインワンのサンプリング (`LZKSamplerDecode`)**: KSamplerとVAEDecodeを1つのステップにまとめ、`lz_pipe` から必要なデータを直接引き出します。もしコンディショニングデータが接続されていなくても、入力されたテキストから内部で自動的にCLIPエンコードを行う賢い設計です。
*   **XY Plotシステム (`LZXYPlot` / `LZXYPlotSampler` / `LZXYSampler` / `LZXYGridOutput`)**: チェックポイント、拡散モデル単体(Krea 2 / Anima / Flux などのDiT)、LoRA、Sampler、Scheduler、プロンプトの変化をグリッドで比較するシステムです。置換キーや自動グリッド画像生成に対応しています。
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
| **Loaders** | EZCheckpointLoader, LZSimpleCheckpointLoader, LZLoRAStacker, LZAnimaLoader, LZKrea2Loader |
| **Anima** | LZAnimaArtistNode, LZAnimaArtistPack, LZAnimaArtistCrossAttn, LZAnimaArtistOptions |
| **Prompt** | DualCLIPTextEncode, AdvancedPositivePrompt, AdvancedNegativePrompt, LZPromptWeight, LZTagEditor |
| **Text** | StringNode, StringConcatNode, LZTextPreview, LZStringSanitize, LZStringSelect, LZSaveStringToCSV, LZPromptReplaceSingle, LZPromptReplaceMulti, LZPromptReplaceString |
| **Latent** | PresetEmptyLatentImage |
| **Pipe** | LZPipePack, LZPipePackXL, LZPipeUnpack, LZPipeInfo, LZPipeMerge |
| **Sampling** | LZKSamplerDecode |
| **XY Plot** | LZXYPlot, LZXYPlotSampler, LZXYSampler, LZXYGridOutput |
| **IO** | LZSaveImageAndLog, LZBatchSaveWithLabels |
| **Log** | LZAppendLogToCSV, LZLogReader |
| **Merge** | LZMergeRecipeRandom, LZMergeRecipeManual, LZMergeRecipeRandomAdvanced |

## Krea 2 クイックガイド

`LZKrea2Loader` が [Krea 2](https://huggingface.co/krea/Krea-2-Turbo) アーキテクチャにネイティブ対応しています(要 **ComfyUI v0.26.0 以降**)。

**モデルファイル**([Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) の ComfyUI 再パッケージ版を推奨):

| 種類 | ファイル例 | 配置先 |
|---|---|---|
| 拡散モデル(DiT) | `krea2_turbo_fp8_scaled.safetensors` | `models/diffusion_models` |
| テキストエンコーダー | `qwen3vl_4b_fp8_scaled.safetensors` | `models/text_encoders` |
| VAE | `qwen_image_vae.safetensors` | `models/vae` |
| スタイルLoRA(任意) | `krea2_darkbrush.safetensors` など | `models/loras` |

**推奨設定**(公式 [krea-2](https://github.com/krea-ai/krea-2) リポジトリより):

| バリアント | ステップ数 | CFG | Sampler | Scheduler |
|---|---|---|---|---|
| Krea 2 **Turbo** | 8 | 1.0 | euler | simple |
| Krea 2 **Raw** | 52 | 3.5 | euler | simple |

*   Krea 2 Turbo は蒸留モデルのため、CFG 1.0 ではネガティブプロンプトが機能しません。`negative_mode="auto"`(デフォルト)は positive をゼロ化したものをネガティブとして使用します(公式テンプレートの `ConditioningZeroOut` と同一挙動で、無駄なテキストエンコードを省略)。
*   サイズは 1K〜2K に対応。`PresetEmptyLatentImage`(16ch モード)か、本体の `ResolutionSelector` ノードを `width`/`height` 入力に接続して使用してください。
*   Krea 2 で XY Plot を使う場合は `diffusion_model` 軸を使用してください(`checkpoint` 軸はフルチェックポイント専用です)。
*   Turbo でネガティブプロンプトを使いたい場合は、[ComfyUI-krea2-negpip](https://github.com/blue-pen5805/ComfyUI-krea2-negpip) などのコミュニティノードと併用できます。

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

## サードパーティーノードについて
**Anima Artist Mixer** - `LZAnimaArtistNode`、`LZAnimaArtistPack`、`LZAnimaArtistCrossAttn`、`LZAnimaArtistOptions` は [Anima-Artist-Mixer](https://github.com/An1X3R/Anima-Artist-Mixer) (by An1X3R and 汐浮尘) のコードをベースに改変しています。
- MIT License に基づき使用しています：

```
MIT License

Copyright (c) 2026 An1X3R and 汐浮尘

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
