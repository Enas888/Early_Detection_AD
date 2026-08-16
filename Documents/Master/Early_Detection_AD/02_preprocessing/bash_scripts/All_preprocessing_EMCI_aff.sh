#!/usr/bin/env bash

# ==========================================================
# Fast Affine MRI Preprocessing (EMCI)
# ==========================================================

# -----------------------
# Use all CPU cores
# -----------------------
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
export OMP_NUM_THREADS=$(nproc)

# -----------------------
# Paths
# -----------------------
CSV="/home/enooo/Documents/Master/Early_Detection_AD/01_data/csv/master_metadata_3t.csv"

INPUT_DIR="/home/enooo/Documents/Master/Early_Detection_AD/01_data/raw/Classes/EMCI"

OUTPUT_DIR="/home/enooo/Documents/Master/Early_Detection_AD/01_data/processed/classes/affine/EMCI"

MNI_TEMPLATE="/home/enooo/fsl/data/standard/MNI152_T1_1mm.nii.gz"

mkdir -p "$OUTPUT_DIR"

# ==========================================================
# Process only EMCI subjects
# ==========================================================

tail -n +2 "$CSV" | while IFS=',' read -r SUBJECT IMAGE_ID CLASS REST
do
    # Remove Windows carriage return if present
    CLASS=$(echo "$CLASS" | tr -d '\r')

    # Only EMCI
    [ "$CLASS" != "EMCI" ] && continue

    INPUT="${INPUT_DIR}/${SUBJECT}_I${IMAGE_ID}.nii.gz"

    if [ ! -f "$INPUT" ]; then
        echo "Missing file: $(basename "$INPUT")"
        continue
    fi

    OUTPUT_NAME="${SUBJECT}_${IMAGE_ID}"

    echo "=========================================="
    echo "Processing: ${OUTPUT_NAME}"
    echo "=========================================="

    WORKDIR="${OUTPUT_DIR}/tmp_${OUTPUT_NAME}"
    mkdir -p "$WORKDIR"

    # ------------------------------------------------------
    # Step 1 : N4 Bias Field Correction
    # ------------------------------------------------------
    N4BiasFieldCorrection \
        -i "$INPUT" \
        -o "$WORKDIR/n4.nii.gz"

    # ------------------------------------------------------
    # Step 2 : Affine Registration
    # ------------------------------------------------------
    antsRegistrationSyN.sh \
        -d 3 \
        -f "$MNI_TEMPLATE" \
        -m "$WORKDIR/n4.nii.gz" \
        -o "$WORKDIR/reg_" \
        -t a \
        -n $(nproc)

    REG_IMG="$WORKDIR/reg_Warped.nii.gz"

    if [ ! -f "$REG_IMG" ]; then
        echo "Registration failed: ${OUTPUT_NAME}"
        rm -rf "$WORKDIR"
        continue
    fi

    # ------------------------------------------------------
    # Step 3 : Brain Extraction
    # ------------------------------------------------------
    bet \
        "$REG_IMG" \
        "$WORKDIR/brain.nii.gz" \
        -R -f 0.5 -g 0 -m

    # ------------------------------------------------------
    # Step 4 : Intensity Normalization
    # ------------------------------------------------------
    fslmaths \
        "$WORKDIR/brain.nii.gz" \
        -inm 1 \
        "${OUTPUT_DIR}/${OUTPUT_NAME}.nii.gz"

    rm -rf "$WORKDIR"

    echo "Finished: ${OUTPUT_NAME}"
    echo
done

echo "=========================================="
echo "Affine preprocessing complete."
echo "Results saved to:"
echo "$OUTPUT_DIR"
echo "=========================================="