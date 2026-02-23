// js/dynamic_concat.js

import { app } from "../../scripts/app.js";

// ウィジェットの表示・非表示を正しく計算して更新する共通関数
function updateWidgets(node) {
    if (!node.widgets) return;

    let textWidgets = node.widgets.filter(w => w.name && w.name.startsWith("text"));
    if (textWidgets.length === 0) return;

    let keepVisible = 1;
    for (let i = 0; i < textWidgets.length; i++) {
        let w = textWidgets[i];
        let hasValue = w.value && w.value !== "";
        let isInput = node.inputs && node.inputs.some(inp => inp.name === w.name);
        
        if (hasValue || isInput) {
            keepVisible = i + 1;
        }
    }

    // 隠す処理と表示する処理
    for (let i = 0; i < textWidgets.length; i++) {
        let w = textWidgets[i];
        if (i >= keepVisible) {
            if (w.type !== "hidden_text") {
                w.origType = w.type;
                w.type = "hidden_text";
                w.computeSize = () => [0, -4];
            }
        } else {
            if (w.type === "hidden_text") {
                w.type = w.origType || "customtext";
                w.computeSize = undefined;
            }
        }
    }

    const size = node.computeSize();
    size[0] = Math.max(size[0], node.size[0]);
    node.setSize(size);
}

app.registerExtension({
    name: "LZ.DynamicStringConcat",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "StringConcatNode") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                this.addWidget("button", "➕ Add Text", "add_button", () => {
                    if (!this.widgets) return;
                    let textWidgets = this.widgets.filter(w => w.name && w.name.startsWith("text"));
                    let hiddenWidget = textWidgets.find(w => w.type === "hidden_text");
                    if (hiddenWidget) {
                        hiddenWidget.type = hiddenWidget.origType || "customtext";
                        hiddenWidget.computeSize = undefined;
                        this.setSize(this.computeSize());
                    }
                });

                this.addWidget("button", "➖ Remove Last", "remove_button", () => {
                    if (!this.widgets) return;
                    let textWidgets = this.widgets.filter(w => w.name && w.name.startsWith("text"));
                    let visibleWidgets = textWidgets.filter(w => w.type !== "hidden_text");
                    if (visibleWidgets.length > 1) { // 最低1つは残す
                        let lastW = visibleWidgets[visibleWidgets.length - 1];
                        lastW.origType = lastW.type;
                        lastW.type = "hidden_text";
                        lastW.value = ""; 
                        lastW.computeSize = () => [0, -4];
                        this.setSize(this.computeSize());
                    }
                });

                updateWidgets(this);
                requestAnimationFrame(() => updateWidgets(this));
                setTimeout(() => updateWidgets(this), 100);

                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                
                updateWidgets(this);
                requestAnimationFrame(() => updateWidgets(this));
                setTimeout(() => updateWidgets(this), 100);
                setTimeout(() => updateWidgets(this), 500);
                
                return r;
            };
            
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                updateWidgets(this);
                return r;
            };
        }
    }
});