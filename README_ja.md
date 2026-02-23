# ComfyUI-LZNodes

ComfyUI-LZNodes は [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 向けのカスタムノード集です。複雑になりがちなノードの配線を簡略化し、画像生成プロセスをより快適にするために設計されています。

制作者: Liz

[English Version](README.md)

## 主な機能

この拡張機能は、よくあるワークフローをひとまとめにする便利なノードを提供します。

*   **LZ パイプシステム (`LZPipePack` / `LZPipeUnpack`)**: `MODEL`、`CLIP`、`VAE`、`CONDITIONING`、`LATENT`、および各種生成パラメータを1本の `lz_pipe` にまとめることで、ノード同士がスパゲッティのように絡まるのを防ぎます。
*   **スマートなローダー群**:
    *   `EZCheckpointLoader`: チェックポイントの読み込みと、ポジティブ/ネガティブプロンプトのエンコードを1つのノードで同時に行います。
    *   `LZSimpleCheckpointLoader`: モデルの各コンポーネントを取り出すことに特化した軽量なローダーです。
    *   `LZLoRAStacker`: 最大10個までのLoRAを簡単にスタック可能です。キャッシュ機能により高速な生成をサポートします。
*   **プロンプト管理**:
    *   `AdvancedPositivePrompt` / `AdvancedNegativePrompt`: 基本のプロンプト、アーティストタグ、クオリティタグなどを綺麗に分けて入力できます。内部で自動的に結合＆エンコードされます。
    *   `DualCLIPTextEncode`: ポジティブプロンプトとネガティブプロンプトを横並びで同時にエンコードするノードです。
*   **オールインワンのサンプリング (`LZKSamplerDecode`)**: KSamplerとVAEDecodeを1つのステップにまとめ、`lz_pipe` から必要なデータを直接引き出します。もしコンディショニングデータが接続されていなくても、入力されたテキストから内部で自動的にCLIPエンコードを行う賢い設計です。
*   **画像保存と詳細なログ記録 (`LZSaveImageAndLog`)**: 生成された画像（PNG/WEBP/JPG）の保存と同時に、シード値、ステップ数、さらにはチェックポイントのハッシュ値やプロンプトの全文まで、考えうる限りの詳細なメタデータを `.txt` ログファイルとして出力します。もちろん、PNGの不可視メタデータ領域（PNGInfo）への書き込みにも対応しています。

## インストール方法

1. ComfyUIの `custom_nodes` ディレクトリに移動します。
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. このリポジトリをクローンします。
   ```bash
   git clone <repository_url> ComfyUI-LZNodes
   ```
3. ComfyUI を再起動してください。

## 使い方

インストール後、ComfyUIの新規ノード追加メニュー内の `MyCustomNodes` カテゴリ（例: `MyCustomNodes/Loaders`, `MyCustomNodes/Prompt`, `MyCustomNodes/Sampling` など）から呼び出すことができます。

*ヒント: `EZCheckpointLoader`を用意し、`LZPipePack` を通して `LZKSamplerDecode` に繋ぎ、最後に `LZSaveImageAndLog` で締めくくるのが、最も簡単で強力な使い方です。*

## ライセンス

このプロジェクトは MIT License の下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。
