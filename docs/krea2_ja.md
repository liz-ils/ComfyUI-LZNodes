# Krea 2 クイックガイド

[READMEに戻る](../README_ja.md) | [English](krea2.md)

`LZKrea2Loader` は [Krea 2](https://huggingface.co/krea/Krea-2-Turbo) アーキテクチャにネイティブ対応しています（要 **ComfyUI v0.26.0 以降**）。

**モデルファイル**（[Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) の ComfyUI 再パッケージ版を推奨）:

| 種類 | ファイル例 | 配置先 |
|---|---|---|
| 拡散モデル（DiT） | `krea2_turbo_fp8_scaled.safetensors` | `models/diffusion_models` |
| テキストエンコーダー | `qwen3vl_4b_fp8_scaled.safetensors` | `models/text_encoders` |
| VAE | `qwen_image_vae.safetensors` | `models/vae` |
| スタイル LoRA（任意） | `krea2_darkbrush.safetensors` など | `models/loras` |

**推奨設定**（公式 [krea-2](https://github.com/krea-ai/krea-2) リポジトリより）:

| バリアント | ステップ数 | CFG | Sampler | Scheduler |
|---|---|---|---|---|
| Krea 2 **Turbo** | 8 | 1.0 | euler | simple |
| Krea 2 **Raw** | 52 | 3.5 | euler | simple |

- Krea 2 Turbo は蒸留モデルのため、CFG 1.0 ではネガティブプロンプトが機能しません。`negative_mode="auto"`（デフォルト）は、ポジティブのコンディショニングをゼロ化したものをネガティブとして使用します（公式テンプレートの `ConditioningZeroOut` と同一の挙動で、不要なテキストエンコードを省略できます）。
- サイズは 1K〜2K に対応します。`PresetEmptyLatentImage`（16ch モード）か、本体の `ResolutionSelector` ノードを `width` / `height` 入力に接続して使用してください。
- Krea 2 で XY Plot を使う場合は `diffusion_model` 軸を使用してください（`checkpoint` 軸はフルチェックポイント専用です）。
- Turbo でネガティブプロンプトを使いたい場合は、[ComfyUI-krea2-negpip](https://github.com/blue-pen5805/ComfyUI-krea2-negpip) などのコミュニティノードと併用できます。
