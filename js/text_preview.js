// js/text_preview.js
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "LZ.TextPreview",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LZTextPreview") {
            const onExecuted = nodeType.prototype.onExecuted;
            
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }
                
                if (message && message.text) {
                    if (this.widgets) {
                        const index = this.widgets.findIndex((w) => w.name === "text_preview");
                        if (index !== -1) {
                            for (let i = index; i < this.widgets.length; i++) {
                                this.widgets[i].onRemove?.();
                            }
                            this.widgets.length = index;
                        }
                    }
                    
                    const w = ComfyWidgets["STRING"](this, "text_preview", ["STRING", { multiline: true }], app).widget;
                    
                    w.inputEl.readOnly = true;
                    w.inputEl.style.opacity = 0.7;
                    
                    w.value = message.text.join("");
                    
                    if (this.onResize) {
                        this.onResize(this.size);
                    }
                }
            };
        }
    }
});