# ノードリファレンス

[READMEに戻る](../README_ja.md) | [English](nodes.md)

ComfyUI-LZNodes の全47ノードを、メニューカテゴリ（`MyCustomNodes` 配下）ごとにまとめています。「表示名」は ComfyUI の UI 上で表示・検索できる名前、「クラス名」はワークフローの JSON 内で使用される名前です。

## Loaders

| 表示名 | クラス名 |
|---|---|
| LZ Checkpoint Loader & Encode | `LZCheckpointLoader` |
| LZ Checkpoint Loader & Encode (Legacy) | `EZCheckpointLoader` |
| LZ Simple Checkpoint Loader | `LZSimpleCheckpointLoader` |
| LZ LoRA Stacker | `LZLoRAStacker` |
| LZ LoRA Stacker (Model Only) | `LZLoRAStackerModelOnly` |
| LZ LoRA Loader (Model Only) | `LZLoRALoaderModelOnly` |
| LZ Anima Loader | `LZAnimaLoader` |
| LZ Krea2 Loader | `LZKrea2Loader` |

- `LZCheckpointLoader`: チェックポイントの読み込みと、ポジティブ/ネガティブプロンプトのエンコードを1つのノードで行います。
- `EZCheckpointLoader`: `LZCheckpointLoader` の旧バージョンです。過去のワークフローがそのまま動作するよう残されています。
- `LZSimpleCheckpointLoader`: モデルの各コンポーネント（MODEL / CLIP / VAE）の取り出しに特化した軽量なローダーです。
- `LZLoRAStacker`: 最大10個までの LoRA をスタックできます。キャッシュ機能により高速な生成をサポートします。`lora_model` / `lora_weight`（カンマ連結のモデル名/重み、`lz_pipe` にも格納）を出力します。
- `LZLoRAStackerModelOnly` / `LZLoRALoaderModelOnly`: LoRA を MODEL のみに適用し、CLIP には触れないバリエーションです。
- `LZAnimaLoader`: Anima モデル専用のローダーで、拡散モデル、テキストエンコーダー、VAE を個別に読み込みます。
- `LZKrea2Loader`: Krea 2 アーキテクチャ（12B SingleStreamDiT + Qwen3-VL-4B テキストエンコーダー + Qwen-Image VAE）専用のワンノードローダーです。`lz_pipe` 対応、`weight_dtype` 選択可能。`negative_mode="auto"`（デフォルト）では、ネガティブプロンプト未使用時にポジティブのコンディショニングをゼロ化してネガティブとして使用します（Krea 2 Turbo は CFG 1.0 前提のため、公式テンプレートと同一の挙動）。ComfyUI v0.26.0 以降が必要です。詳細は [Krea 2 クイックガイド](krea2_ja.md) を参照してください。

## Anima

| 表示名 | クラス名 |
|---|---|
| LZ Anima Artist Node | `LZAnimaArtistNode` |
| LZ Anima Artist Pack (Split + Encode) | `LZAnimaArtistPack` |
| LZ Anima Artist Cross-Attn (v2) | `LZAnimaArtistCrossAttn` |
| LZ Anima Artist Options (Advanced) | `LZAnimaArtistOptions` |
| LZ Artist Chain Text | `LZArtistChainText` |

- `LZAnimaArtistNode`: 統合ノード。アーティストチェーンの分割・エンコードと cross-attention 注入を1ステップで実行します。`(MODEL, CONDITIONING, STRING)` を出力し、STRING はベースプロンプトのテキスト（`positive_text`）なので、パイプシステムに直接接続できます。
- `LZAnimaArtistPack` / `LZAnimaArtistCrossAttn`: 分割・エンコードと cross-attn 注入を分離したノード（上級者向け）。`LZAnimaArtistCrossAttn` も `positive_text`（STRING）を出力します。
- `LZAnimaArtistOptions`: ブロック範囲、fusion モード、EMA 平滑化、低ランク注入、static capture、anchor Q を細かく設定するための高度なオプションノードです。
- `LZArtistChainText`: `artist_chain` 入力向けの補助ノード（UI 上では緑色で表示）。改行区切りのアーティスト一覧をカンマ連結した `artist_tag` に変換し、元テキストと `lz_pipe` もそのまま出力します。

## Prompt

| 表示名 | クラス名 |
|---|---|
| LZ Advanced Positive Prompt | `AdvancedPositivePrompt` |
| LZ Advanced Negative Prompt | `AdvancedNegativePrompt` |

- `AdvancedPositivePrompt` / `AdvancedNegativePrompt`: 基本プロンプト、アーティストタグ、クオリティタグを分けて入力できます。内部で自動的に結合とエンコードが行われます。

## Conditioning

| 表示名 | クラス名 |
|---|---|
| LZ Dual CLIP Text Encode | `DualCLIPTextEncode` |
| LZ CLIP Text Encode | `LZCLIPTextEncode` |

- `DualCLIPTextEncode`: ポジティブプロンプトとネガティブプロンプトを1つのノードで同時にエンコードします。
- `LZCLIPTextEncode`: 単一のテキストをエンコードし、`(CONDITIONING, STRING)` を出力します。

## Text

| 表示名 | クラス名 |
|---|---|
| LZ String Output | `StringNode` |
| LZ String Concat | `StringConcatNode` |
| LZ Text Preview | `LZTextPreview` |
| LZ String Sanitize | `LZStringSanitize` |
| LZ String Select | `LZStringSelect` |
| LZ Save String to CSV | `LZSaveStringToCSV` |
| LZ Prompt Weight | `LZPromptWeight` |
| LZ Tag Editor | `LZTagEditor` |
| LZ Prompt Replace Single | `LZPromptReplaceSingle` |
| LZ Prompt Replace Multi | `LZPromptReplaceMulti` |
| LZ Prompt Replace String | `LZPromptReplaceString` |
| LZ Prompt Pick (Multi-source) | `LZPromptPick` |

- `StringNode` / `StringConcatNode`: 基本的な文字列出力と、複数入力の結合ノードです。
- `LZTextPreview`: 文字列をノード UI 上に直接表示します。
- `LZStringSanitize`: ファイル名に使用できない文字を除去・置換します。
- `LZStringSelect`: 複数の文字列入力からインデックスで1つを選択します。
- `LZSaveStringToCSV`: 文字列の行を CSV ファイルに保存・追記します（ヘッダーはファイル新規作成時のみ書き込まれます）。
- `LZPromptWeight`: プロンプトタグに重み構文 `(tag:weight)` を自動で付与します。
- `LZTagEditor`: 最大10個のタグを、強度や ON/OFF とともに視覚的に編集できます。
- `LZPromptReplaceSingle` / `LZPromptReplaceMulti` / `LZPromptReplaceString`: txt/csv/json ファイルや直書きの候補リストからランダムに選択し、プロンプト内のプレースホルダーを動的に置換します。seed 対応（`-1` は毎回ランダム）。ファイルの配置場所は[ファイルの配置場所](#ファイルの配置場所)を参照してください。
- `LZPromptPick`: 最大2ファイルと直書き候補から複数個をピックします。ランダム強度（`weight_min` / `weight_max`）、重複なし（`unique`、デフォルト ON）、seed 制御に対応します。

## Latent

| 表示名 | クラス名 |
|---|---|
| LZ Preset Empty Latent Image | `PresetEmptyLatentImage` |

- `PresetEmptyLatentImage`: SDXL の一般的な解像度で空の Latent 画像を素早く作成できます。SDXL（4ch）と Anima（16ch）の両方の潜在形式に対応しています。

## Pipe

| 表示名 | クラス名 |
|---|---|
| LZ Pipe Pack | `LZPipePack` |
| LZ Pipe Pack XL | `LZPipePackXL` |
| LZ Pipe Unpack | `LZPipeUnpack` |
| LZ Pipe Info | `LZPipeInfo` |
| LZ Pipe Merge | `LZPipeMerge` |

- パイプ系ノードは、`MODEL`、`CLIP`、`VAE`、`CONDITIONING`、`LATENT`、生成パラメータを1本の `lz_pipe` にまとめます。Pack がバンドルを作成し、Unpack が中身を取り出し、Merge が2本のパイプを結合し、Info がパイプの内容を表示します。

## Sampling

| 表示名 | クラス名 |
|---|---|
| LZ KSampler & Decode | `LZKSamplerDecode` |

- `LZKSamplerDecode`: KSampler と VAE Decode を1つのノードにまとめ、`lz_pipe` から必要なデータを直接読み取ります。コンディショニングが接続されていない場合は、パイプ内の CLIP を使って入力テキストを内部でエンコードすることもできます。

## XY

| 表示名 | クラス名 |
|---|---|
| LZ XY Plot | `LZXYPlot` |
| LZ XY Plot & Sampler | `LZXYPlotSampler` |
| LZ XY Sampler | `LZXYSampler` |
| LZ XY Grid Output | `LZXYGridOutput` |

- XY Plot 系ノードは、チェックポイント、拡散モデル単体（Krea 2 / Anima / Flux などの DiT）、LoRA、Sampler、Scheduler、プロンプトの変化をグリッドで比較します。置換キーと、ラベル付きグリッド画像の自動生成に対応しています。

## IO

| 表示名 | クラス名 |
|---|---|
| LZ Save Image & Log | `LZSaveImageAndLog` |
| LZ Batch Save With Labels | `LZBatchSaveWithLabels` |

- `LZSaveImageAndLog`: 生成画像（PNG / WEBP / JPG）の保存と同時に、シード値、ステップ数、チェックポイントのハッシュ値、プロンプト全文などを含む `.txt` ログファイルを出力します。PNG の場合は同じ情報を PNG メタデータ（`parameters`）にも埋め込みます。なお、メタデータへの埋め込みは PNG のみで、`.txt` ログは全形式で出力されます。
- `LZBatchSaveWithLabels`: XY ラベルをファイル名とメタデータに埋め込んで一括保存し、バッチログも自動で出力します。

## Log

| 表示名 | クラス名 |
|---|---|
| LZ Append Log to CSV | `LZAppendLogToCSV` |
| LZ Log Reader | `LZLogReader` |

- `LZAppendLogToCSV`: 生成パラメータを CSV ファイルに追記し、スプレッドシートでの管理を可能にします。
- `LZLogReader`: `LZSaveImageAndLog` が出力した `.txt` ログを読み込み、パラメータをノード出力として復元します。

## Merge

| 表示名 | クラス名 |
|---|---|
| LZ Merge Recipe Random | `LZMergeRecipeRandom` |
| LZ Merge Recipe Manual | `LZMergeRecipeManual` |
| LZ Merge Recipe Random Advanced | `LZMergeRecipeRandomAdvanced` |

- マージレシピ系ノードは、モデルマージツールと互換性のある19ブロック（IN00–IN08、M00、OUT00–OUT08）のマージレシピを生成します。入出力の詳細は [マージレシピノード](merge-recipe_ja.md) を参照してください。

## ファイルの配置場所

- **プロンプト候補ファイル**（`LZPromptReplaceSingle` / `LZPromptReplaceMulti` / `LZPromptReplaceString` / `LZPromptPick`）: `.txt` / `.csv` / `.json` ファイルを、ComfyUI の出力ディレクトリ内の `prompt_csv` フォルダ（`ComfyUI/output/prompt_csv/`）に置いてください。フォルダは初回使用時に自動で作成されます。対応するファイルサイズは最大 2 MB です。
- **保存画像とログ**（`LZSaveImageAndLog` / `LZBatchSaveWithLabels`）: 本体の Save Image ノードと同様に、ComfyUI の出力ディレクトリに保存されます。
