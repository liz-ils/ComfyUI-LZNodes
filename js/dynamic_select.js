// js/dynamic_select.js

import { app } from "../../../scripts/app.js";

function updateVisibilityByCount(node) {
    if (!node.widgets) return;
    const countW = node.widgets.find(w => w.name === "count");
    const count = Math.max(1, Math.min(10, Number(countW?.value ?? 2)));

    const textWidgets = node.widgets.filter(w => w.name && w.name.startsWith("text"));
    for (let i = 0; i < textWidgets.length; i++) {
        const w = textWidgets[i];
        const index = Number(w.name.replace("text", ""));
        const shouldShow = index <= count;
        if (!shouldShow) {
            if (w.type !== "hidden") {
                w.origType = w.type;
                w.origComputeSize = w.computeSize;
                w.type = "hidden";
                w.computeSize = () => [0, -4];
            }
        } else {
            if (w.type === "hidden") {
                w.type = w.origType || "customtext";
                w.computeSize = w.origComputeSize || undefined;
            }
        }
    }

    // ノードサイズを再計算（幅は維持して高さのみ更新）
    const newSize = node.computeSize();
    node.setSize([node.size[0], newSize[1]]);

    // select_index の値を count に収める
    const selW = node.widgets.find(w => w.name === "select_index");
    if (selW) {
        let v = Number(selW.value ?? 1);
        if (!Number.isFinite(v)) v = 1;
        v = Math.max(1, Math.min(count, v));
        if (v !== selW.value) {
            selW.value = v;
            if (app?.graph) app.graph.setDirtyCanvas(true, true);
        }
    }
}

app.registerExtension({
    name: "LZ.DynamicStringSelect",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LZStringSelect") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                // 初期表示で即座に不要ウィジェットを非表示化
                updateVisibilityByCount(this);

                // count 変更時に UI 更新（単発）
                const countW = this.widgets?.find(w => w.name === "count");
                if (countW) {
                    const prevCallback = countW.callback;
                    countW.callback = (v) => {
                        if (prevCallback) prevCallback(v);
                        updateVisibilityByCount(this);
                        if (app?.graph) app.graph.setDirtyCanvas(true, true);
                    };
                }
                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                // 設定反映時も一度だけ反映
                updateVisibilityByCount(this);
                return r;
            };
        }
    }
});
