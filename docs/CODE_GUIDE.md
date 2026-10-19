# Code guide (for the viva)

## Pipeline in one picture

    collect_gee.py  -> Sentinel-2 chips + ERA5-Land weather from Google Earth Engine
    add_history.py  -> adds fire-history features from FIRMS (fire_history.csv)
    real_dataset_matched.npz  -> arrays: img, seq, y, lat, lon, year, hist ...
    train.py        -> load/split data, build the network, train, evaluate
    paper_results.py-> trains 10 model variants x several seeds, writes tables 1-7, figures 1-9, saved model
    run_demo.py     -> loads the saved model, re-evaluates, writes the report and the map
    make_map.py     -> builds the interactive risk map (folium)

## train.py (core)

| Function / class | What it does |
|---|---|
| `interpolate_gaps(X)` | fills missing days (cloud gaps) in the time series by interpolation |
| `harvest_features(seq)` | derives 5 harvest-timing numbers from the NDVI curve (e.g. when it drops) |
| `load_and_split(path)` | loads the .npz, builds the features and splits train/val/test. With `STUBBLE_SPLIT=random` it groups fields into 0.1-degree location cells (`STUBBLE_CELL`) and splits cells 70/15/15 so nearby fields never leak across splits |
| `ImageBranch` | CNN: 3 blocks of Conv-BatchNorm-ReLU-MaxPool, then global average pooling |
| `SeqBranch` | LSTM (64 hidden units) over the daily series |
| `Net` | combines image (CNN), sequence (LSTM) and static (fire history + harvest) features; head = Dropout(0.2), Linear(144,64), ReLU, Dropout(0.5), Linear(64,1) |
| `augment(img)` | random flips/rotations of the image chip (training only) |
| `train_model(...)` | AdamW (weight decay 1e-2), BCE loss with `pos_weight` for class imbalance, Gaussian noise 0.05 on the series, early stopping on validation PR-AUC |
| `predict(model, part)` | returns risk probabilities |
| `best_threshold(y, p)` | picks the cut-off that maximises F1 on the validation set |
| `permutation_importance(...)` | scrambles one input at a time and measures the drop in PR-AUC |
| `evaluate(...)` | PR-AUC, ROC-AUC, accuracy, precision, recall, F1, top-share precision/recall/lift |
| `main()` | runs the full experiment |

## Why these design choices

- **PR-AUC as main metric**: only about 29-35% of fields burn, so accuracy alone is misleading.
- **pos_weight**: tells the loss that missing a burn is worse than a false alarm.
- **Dropout, weight decay, noise, augmentation, early stopping**: the first version overfit (train 0.93 vs val 0.70 PR-AUC).
- **Location-grouped split**: fields in the same ~11 km cell share weather, so splitting fields at random would leak information.
- **Validation-chosen threshold**: the test set is never used to tune anything.
- **Bootstrap CIs and paired differences**: show whether a gain between two models is real or noise.

## Likely questions

- *Why CNN + LSTM?* The image shows land state (space), the series shows change over time.
- *Is it overfitting?* Reduced with regularisation; see table of train/val/test scores in the report.
- *Why is accuracy about the same as Random Forest?* Honest answer: they are statistically tied; the deep model's value here is multi-modal input and the image/sequence pipeline.
- *Does pooling years inflate results?* Yes for forecasting; the year-based test (about 0.67) is the stricter figure.
