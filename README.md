# ComfyUI-LZNodes

[![Release](https://img.shields.io/github/v/release/liz-ils/ComfyUI-LZNodes?style=flat-square)](https://github.com/liz-ils/ComfyUI-LZNodes/releases)
[![License: MIT](https://img.shields.io/github/license/liz-ils/ComfyUI-LZNodes?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue?style=flat-square)

**English** | [日本語](README_ja.md)

A collection of 47 custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that simplifies complex node wiring and streamlines image generation: a pipe system that bundles models and parameters into a single connection, smart Checkpoint / LoRA / Anima / Krea 2 loaders, prompt management, XY plot comparison, and image saving with detailed generation logs.

## Requirements

- ComfyUI — a recent version is recommended. The Krea 2 nodes require **v0.26.0 or later**.
- Python 3.10 or later (the Python used by ComfyUI is fine).
- No additional packages: this pack only uses libraries that ship with ComfyUI (PyTorch, NumPy, Pillow).

## Installation

**ComfyUI-Manager (recommended):** search for **ComfyUI-LZNodes** in "Install Custom Nodes", install it, and restart ComfyUI. The pack is also published on the Comfy Registry.

**git clone:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/liz-ils/ComfyUI-LZNodes
```

Then restart ComfyUI. To update, use the Manager's Update function, or run `git pull` in the `ComfyUI-LZNodes` folder and restart.

## Documentation

- [Node Reference](docs/nodes.md) — all 47 nodes: the full list, display names, categories, and what each node does.
- [Krea 2 Quick Guide](docs/krea2.md) — model file placement and recommended settings for `LZKrea2Loader`.
- [Merge Recipe Nodes](docs/merge-recipe.md) — inputs and outputs of the merge recipe nodes.
- [Third-Party Licenses](docs/third-party-licenses.md) — license notices for included third-party code.

## License

MIT License — see the [LICENSE](LICENSE) file. Version history is on the [Releases](https://github.com/liz-ils/ComfyUI-LZNodes/releases) page.

The Anima Artist Mixer nodes are based on [Anima-Artist-Mixer](https://github.com/An1X3R/Anima-Artist-Mixer) by An1X3R and 汐浮尘 (MIT License). See [Third-Party Licenses](docs/third-party-licenses.md).
