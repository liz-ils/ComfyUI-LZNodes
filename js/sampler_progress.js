// js/sampler_progress.js
import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "LZ.SamplerProgress",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LZKSamplerDecode") {
            nodeType.prototype.onNodeCreated = function() {
                const progressWidget = this.addWidget("text", "progress", "", () => {});
                progressWidget.inputEl.readOnly = true;
                progressWidget.inputEl.style.textAlign = "center";
                progressWidget.inputEl.style.background = "transparent";
                progressWidget.inputEl.style.border = "none";
                progressWidget.inputEl.style.color = "#888";
                progressWidget.inputEl.style.fontSize = "12px";
            };

            const originalOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (originalOnExecuted) {
                    originalOnExecuted.apply(this, arguments);
                }
                
                const widgets = this.widgets;
                if (widgets) {
                    const progressWidget = widgets.find(w => w.name === "progress");
                    if (progressWidget) {
                        progressWidget.value = "Generation complete";
                    }
                }
            };
        }
    }
});
