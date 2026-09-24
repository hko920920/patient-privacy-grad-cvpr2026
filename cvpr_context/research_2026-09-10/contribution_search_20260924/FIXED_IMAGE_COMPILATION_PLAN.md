# Fixed-image compilation feasibility probe

Written after the direct public transport result and before the compilation outcomes.

Question: can transported readouts be carried by the existing 128 PNGs by changing only soft labels, and does a label packet fitted on one development receiver transfer to the other? This is exploratory; both receiver families have already been used in development.

No Q/raw private statistics, no new DP summary, no image optimization, no new image bank, no new encoder forward, no Expert/Reserved. Reuse both feature-DP banks, all outcomes retained.

For each bank and calibration receiver (DenseNet and ResNet18, both directions fixed in advance), solve a bounded linear least squares problem for 128 soft labels in [-1,1]. The objective matches the public-P scores of the transported classifier when a ridge0.1 readout is trained on the fixed PNG features. Use public patient/class-balanced weights and a fixed label regularizer 0.001*trace(A'WA)/128 toward the original labels. No V loss, no hyperparameter selection. Evaluate the resulting SAME soft-label packet on both receivers with the original ridge0.1 convention. Keep public-fit residuals and all V endpoints.

This is not a claim of novelty for soft labels or a successful 200-step synthetic-image method. It measures whether fixed-image label correction alone can preserve the newly recovered function and cross-model transfer.

