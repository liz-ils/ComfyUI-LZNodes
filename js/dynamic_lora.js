// js/dynamic_lora.js
// LZ LoRA Stacker / LZ LoRA Stacker (Model Only) のスロット表示を lora_count で制御する。
// 旧方式(hidden_text + ボタン直隠し)では新旧フロントエンドで個数調整が効かないため、
// count 駆動 + type:"hidden" 方式に統一した。

import { app } from "../../../scripts/app.js";

const TARGETS = ["LZLoRAStacker", "LZLoRAStackerModelOnly"];
const MAX_LORAS = 10;

function getCount(node) {
    const w = node.widgets?.find((w) => w.name === "lora_count");
    let v = Number(w?.value ?? 1);
    if (!Number.isFinite(v)) v = 1;
    return Math.max(1, Math.min(MAX_LORAS, Math.round(v)));
}

function slotWidgets(node, i) {
    const loraW = node.widgets?.find((w) => w.name === `lora_${i}`);
    const mw = node.widgets?.find((w) => w.name === `model_weight_${i}`);
    const cw = node.widgets?.find((w) => w.name === `clip_weight_${i}`);
    const sm = node.widgets?.find((w) => w.name === `strength_model_${i}`);
    return { loraW, mw, cw, sm };
}

function hideWidget(w) {
    if (!w || w.type === "hidden") return;
    if (w.origType === undefined) {
        w.origType = w.type;
        w.origComputeSize = w.computeSize;
    }
    w.type = "hidden";
    w.computeSize = () => [0, -4];
}

function showWidget(w, fallbackType) {
    if (!w || w.type !== "hidden") return;
    w.type = w.origType || fallbackType || "combo";
    w.computeSize = w.origComputeSize || undefined;
}

function updateVisibility(node) {
    if (!node.widgets) return;
    const count = getCount(node);
    for (let i = 1; i <= MAX_LORAS; i++) {
        const { loraW, mw, cw, sm } = slotWidgets(node, i);
        const show = i <= count;
        if (show) {
            showWidget(loraW, "combo");
            showWidget(mw, "number");
            showWidget(cw, "number");
            showWidget(sm, "number");
        } else {
            hideWidget(loraW);
            hideWidget(mw);
            hideWidget(cw);
            hideWidget(sm);
        }
    }
    try {
        const size = node.computeSize();
        size[0] = Math.max(size[0], node.size[0]);
        node.setSize(size);
    } catch (e) { /* ignore */ }
    if (app?.graph) app.graph.setDirtyCanvas(true, true);
}

// 旧ワークフロー互換: count より後ろのスロットに値があれば count を引き上げる
function backfillCount(node) {
    if (!node.widgets) return;
    const countW = node.widgets.find((w) => w.name === "lora_count");
    let count = getCount(node);
    for (let i = MAX_LORAS; i > count; i--) {
        const { loraW } = slotWidgets(node, i);
        if (loraW && loraW.value !== undefined && loraW.value !== null && loraW.value !== "None" && loraW.value !== "") {
            count = i;
            break;
        }
    }
    if (countW && Number(countW.value) !== count) {
        countW.value = count;
    }
}

app.registerExtension({
    name: "LZ.DynamicLoRAStacker",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!TARGETS.includes(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            this.addWidget("button", "➕ Add LoRA", "add_button", () => {
                const w = this.widgets?.find((w) => w.name === "lora_count");
                if (!w) {
                    // count が無い旧ノードのフォールバック: 最初の隠しスロットを表示
                    updateVisibility(this);
                    return;
                }
                w.value = Math.min(MAX_LORAS, getCount(this) + 1);
                if (w.callback) w.callback(w.value);
                updateVisibility(this);
            });

            this.addWidget("button", "➖ Remove Last", "remove_button", () => {
                const w = this.widgets?.find((w) => w.name === "lora_count");
                if (!w) return;
                const next = Math.max(1, getCount(this) - 1);
                // 末尾スロットの値をリセットしてから減らす
                const { loraW, mw, cw, sm } = slotWidgets(this, getCount(this));
                if (loraW) loraW.value = "None";
                if (mw) mw.value = 1.0;
                if (cw) cw.value = 1.0;
                if (sm) sm.value = 1.0;
                w.value = next;
                if (w.callback) w.callback(w.value);
                updateVisibility(this);
            });

            const countW = this.widgets?.find((w) => w.name === "lora_count");
            if (countW) {
                const prev = countW.callback;
                countW.callback = (v) => {
                    if (prev) prev(v);
                    updateVisibility(this);
                };
            }

            requestAnimationFrame(() => updateVisibility(this));
            setTimeout(() => updateVisibility(this), 50);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            // 読み込み直後は保存値を優先し、必要なら count を引き上げる
            setTimeout(() => {
                backfillCount(this);
                updateVisibility(this);
            }, 20);
            setTimeout(() => {
                backfillCount(this);
                updateVisibility(this);
            }, 300);
            return r;
        };
    },
});
