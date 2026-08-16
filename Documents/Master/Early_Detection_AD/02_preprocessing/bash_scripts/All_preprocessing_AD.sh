#!/usr/bin/env bash

INPUT_DIR="/home/enooo/Documents/Master/data/Classes/AD"

OUTPUT_DIR="/home/enooo/Documents/Master/Early_Detection_AD/01_data/processed/classes/AD"

mkdir -p "$OUTPUT_DIR"

MNI_TEMPLATE="/home/enooo/fsl/data/standard/MNI152_T1_1mm.nii.gz"

for INPUT in "$INPUT_DIR"/*.nii.gz
do
    [ -f "$INPUT" ] || continue

    SUBJECT=$(basename "$INPUT" .nii.gz)

    echo "=================================="
    echo "Processing: $SUBJECT"
    echo "=================================="

    WORKDIR="$OUTPUT_DIR/tmp_$SUBJECT"
    mkdir -p "$WORKDIR"

    # -----------------------
    # Step 1: N4 Bias Correction
    # -----------------------
    N4BiasFieldCorrection \
        -i "$INPUT" \
        -o "$WORKDIR/n4.nii.gz"

    # -----------------------
    # Step 2: Registration
    # -----------------------
    antsRegistrationSyN.sh \
        -d 3 \
        -f "$MNI_TEMPLATE" \
        -m "$WORKDIR/n4.nii.gz" \
        -o "$WORKDIR/reg_" \
        -t r \
        -n 7

    REG_IMG="$WORKDIR/reg_Warped.nii.gz"

    if [ ! -f "$REG_IMG" ]; then
        echo "Registration failed for $SUBJECT"
        rm -rf "$WORKDIR"
        continue
    fi

    # -----------------------
    # Step 3: Brain Extraction
    # -----------------------
    bet \
        "$REG_IMG" \
        "$WORKDIR/brain.nii.gz" \
        -R -f 0.5 -g 0 -m

    # -----------------------
    # Step 4: Intensity Normalization
    # -----------------------
    fslmaths \
        "$WORKDIR/brain.nii.gz" \
        -inm 1 \
        "$OUTPUT_DIR/${SUBJECT}.nii.gz"

    # -----------------------
    # Cleanup
    # -----------------------
    rm -rf "$WORKDIR"

    echo "Finished: $SUBJECT"

done

echo "AD preprocessing complete."