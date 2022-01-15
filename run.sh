DATASET=$1

if [[ "$DATASET" == "swat" ]]; then
    WINDOW_SIZE=50
    HORIZON=1
    STRIDE=1
    BATCH_SIZE=64
    EMBED_DIM=32
    TOPK=5
    EPOCHS=50
    VAL_SPLIT=0.2
    EARLY_STOPPING=10
    SMOOTHING=1
    SMOOTHING_METHOD="exp"
    THRESHOLDING="best"
    TRANSFORM="median"
    TARGET_TRANSFORM="median"
    NORMALIZE="True"
     
elif [[ "$DATASET" == "wadi" ]]; then
    WINDOW_SIZE=50
    STRIDE=1
    HORIZON=1
    BATCH_SIZE=64
    EMBED_DIM=32
    TOPK=8
    EPOCHS=50
    VAL_SPLIT=0.1
    EARLY_STOPPING=10
    SMOOTHING=1
    SMOOTHING_METHOD="exp"
    THRESHOLDING="best"
    TRANSFORM="median"
    TARGET_TRANSFORM="median"
    NORMALIZE="True"

elif [[ "$DATASET" == "demo" ]]; then
    WINDOW_SIZE=25
    STRIDE=1
    HORIZON=1
    BATCH_SIZE=32
    EMBED_DIM=16
    TOPK=3
    EPOCHS=50
    VAL_SPLIT=0
    EARLY_STOPPING=20
    SMOOTHING=1
    SMOOTHING_METHOD="mean"
    THRESHOLDING="best"
    TRANSFORM="none"
    TARGET_TRANSFORM="none"
    NORMALIZE="False"
fi

python main.py \
    -dataset $DATASET \
    -window_size $WINDOW_SIZE \
    -horizon $HORIZON \
    -stride $STRIDE \
    -val_split $VAL_SPLIT \
    -batch_size $BATCH_SIZE \
    -embed_dim $EMBED_DIM \
    -topk $TOPK \
    -epochs $EPOCHS \
    -early_stopping $EARLY_STOPPING \
    -smoothing $SMOOTHING \
    -smoothing_method $SMOOTHING_METHOD \
    -thresholding $THRESHOLDING \
    -normalize $NORMALIZE \
