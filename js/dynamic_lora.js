// js/dynamic_lora.js

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "LZ.DynamicLoRAStacker",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LZLoRAStacker") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // ➕ボタンの処理
                this.addWidget("button", "➕ Add LoRA", "add_button", () => {
                    if (!this.widgets) return;
                    for (let i = 1; i <= 10; i++) {
                        let loraW = this.widgets.find(w => w.name === `lora_${i}`);
                        if (loraW && loraW.type === "hidden_text") {
                            let mw = this.widgets.find(w => w.name === `model_weight_${i}`);
                            let cw = this.widgets.find(w => w.name === `clip_weight_${i}`);
                            
                            loraW.type = loraW.origType || "combo";
                            if(mw) mw.type = mw.origType || "number";
                            if(cw) cw.type = cw.origType || "number";
                            
                            loraW.computeSize = undefined;
                            if(mw) mw.computeSize = undefined;
                            if(cw) cw.computeSize = undefined;
                            
                            this.setSize(this.computeSize());
                            break;
                        }
                    }
                });

                this.addWidget("button", "➖ Remove Last", "remove_button", () => {
                    if (!this.widgets) return;
                    for (let i = 10; i >= 2; i--) {
                        let loraW = this.widgets.find(w => w.name === `lora_${i}`);
                        if (loraW && loraW.type !== "hidden_text") {
                            let mw = this.widgets.find(w => w.name === `model_weight_${i}`);
                            let cw = this.widgets.find(w => w.name === `clip_weight_${i}`);
                            
                            if (!loraW.origType) loraW.origType = loraW.type;
                            if (mw && !mw.origType) mw.origType = mw.type;
                            if (cw && !cw.origType) cw.origType = cw.type;
                            
                            loraW.type = "hidden_text";
                            loraW.value = "None";
                            if(mw) { mw.type = "hidden_text"; mw.value = 1.0; }
                            if(cw) { cw.type = "hidden_text"; cw.value = 1.0; }
                            
                            loraW.computeSize = () => [0, -4];
                            if(mw) mw.computeSize = () => [0, -4];
                            if(cw) cw.computeSize = () => [0, -4];
                            
                            this.setSize(this.computeSize());
                            break;
                        }
                    }
                });

                setTimeout(() => {
                    if (this.widgets) {
                        for (let i = 2; i <= 10; i++) {
                            let loraW = this.widgets.find(w => w.name === `lora_${i}`);
                            let mw = this.widgets.find(w => w.name === `model_weight_${i}`);
                            let cw = this.widgets.find(w => w.name === `clip_weight_${i}`);
                            
                            if (loraW) { loraW.origType = loraW.type; loraW.type = "hidden_text"; loraW.computeSize = () => [0, -4]; }
                            if (mw) { mw.origType = mw.type; mw.type = "hidden_text"; mw.computeSize = () => [0, -4]; }
                            if (cw) { cw.origType = cw.type; cw.type = "hidden_text"; cw.computeSize = () => [0, -4]; }
                        }
                        this.setSize(this.computeSize());
                    }
                }, 10);
                return r;
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                setTimeout(() => {
                    if (!this.widgets) return;
                    let keepVisible = 1;
                    for (let i = 1; i <= 10; i++) {
                        let loraW = this.widgets.find(w => w.name === `lora_${i}`);
                        if (loraW && loraW.value !== "None") {
                            keepVisible = i;
                        }
                    }
                    for (let i = 1; i <= 10; i++) {
                        let loraW = this.widgets.find(w => w.name === `lora_${i}`);
                        let mw = this.widgets.find(w => w.name === `model_weight_${i}`);
                        let cw = this.widgets.find(w => w.name === `clip_weight_${i}`);
                        
                        if (i > keepVisible) {
                            if (loraW && loraW.type !== "hidden_text") { loraW.origType = loraW.type; loraW.type = "hidden_text"; loraW.computeSize = () => [0, -4]; }
                            if (mw && mw.type !== "hidden_text") { mw.origType = mw.type; mw.type = "hidden_text"; mw.computeSize = () => [0, -4]; }
                            if (cw && cw.type !== "hidden_text") { cw.origType = cw.type; cw.type = "hidden_text"; cw.computeSize = () => [0, -4]; }
                        }
                    }
                    this.setSize(this.computeSize());
                }, 10);
                return r;
            };
        }
    }
});