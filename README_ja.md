# ComfyUI-LZNodes

[![Release](https://img.shields.io/github/v/release/liz-ils/ComfyUI-LZNodes?style=flat-square)](https://github.com/liz-ils/ComfyUI-LZNodes/releases)
[![License: MIT](https://img.shields.io/github/license/liz-ils/ComfyUI-LZNodes?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue?style=flat-square)

[English](README.md) | **日本語**

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) 向けのカスタムノード集（全47ノード）です。モデルやパラメータを1本の接続にまとめるパイプシステム、Checkpoint / LoRA / Anima / Krea 2 用のスマートローダー、プロンプト管理、XY Plot による比較、生成ログ付きの画像保存など、複雑になりがちなノード配線を簡略化し、画像生成を快適にするためのノードを収録しています。

## 動作要件

- ComfyUI — 最新版を推奨。Krea 2 用ノードのみ **v0.26.0 以降**が必要です。
- Python 3.10 以降（ComfyUI が使用する Python で問題ありません）。
- 追加パッケージ不要。本パックは ComfyUI に同梱されるライブラリ（PyTorch、NumPy、Pillow）のみを使用します。

## インストール

**ComfyUI-Manager（推奨）:** 「Install Custom Nodes」で **ComfyUI-LZNodes** を検索してインストールし、ComfyUI を再起動してください。Comfy Registry にも公開されています。

**git clone:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/liz-ils/ComfyUI-LZNodes
```

その後、ComfyUI を再起動してください。更新は Manager の Update 機能を使うか、`ComfyUI-LZNodes` フォルダ内で `git pull` を実行して再起動してください。

## ドキュメント

- [ノードリファレンス](docs/nodes_ja.md) — 全47ノードの一覧と、表示名・カテゴリ・各ノードの解説。
- [Krea 2 クイックガイド](docs/krea2_ja.md) — `LZKrea2Loader` 用のモデルファイル配置と推奨設定。
- [マージレシピノード](docs/merge-recipe_ja.md) — マージレシピノードの入出力。
- [サードパーティライセンス](docs/third-party-licenses_ja.md) — 収録しているサードパーティ製コードのライセンス表記。

## ライセンス

MIT License の下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。バージョン履歴は [Releases](https://github.com/liz-ils/ComfyUI-LZNodes/releases) を参照してください。

Anima Artist Mixer 系ノードは [Anima-Artist-Mixer](https://github.com/An1X3R/Anima-Artist-Mixer)（by An1X3R and 汐浮尘、MIT License）をベースにしています。詳細は [サードパーティライセンス](docs/third-party-licenses_ja.md) を参照してください。
