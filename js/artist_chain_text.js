// js/artist_chain_text.js
// LZ Artist Chain Text: green node color.

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "LZ.ArtistChainText",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LZArtistChainText") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                this.color = "#355335";
                this.bgcolor = "#253c25";
                return r;
            };
        }
    }
});
