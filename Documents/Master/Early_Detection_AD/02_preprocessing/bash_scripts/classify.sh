#!/usr/bin/env bash

ROOT="$HOME/Documents/Master/data/new/ID_5548_Cohort_Analysis_MRI_9/ADNI"
STUDY="$HOME/Documents/Master/data/new/ID_5548_Cohort_Analysis_Study_Entry_05Jun2026.csv"
OUT="$HOME/Documents/Master/data/Classes"

mkdir -p "$OUT"/{AD,CN,MCI,EMCI,LMCI}

# =========================
# Load study CSV correctly
# =========================
declare -A GROUP_MAP

while IFS=',' read -r subject_id age visit date group
do
    # skip header or empty lines
    [[ "$subject_id" == "subject_id" || -z "$subject_id" ]] && continue

    # clean ALL noise
    subject_id=$(echo "$subject_id" | tr -d '"\r ')

    group=$(echo "$group" | tr -d '"\r ')

    GROUP_MAP["$subject_id"]="$group"

done < <(tail -n +2 "$STUDY")

echo "Loaded ${#GROUP_MAP[@]} subjects"

# debug check (VERY IMPORTANT)
echo "Sample key: $(printf '%s\n' "${!GROUP_MAP[@]}" | head -1)"

# =========================
# Process images
# =========================
find "$ROOT" -type d -name "I*" | while read -r imgdir
do
    image_id=$(basename "$imgdir" | sed 's/^I//')

    subject=$(echo "$imgdir" | grep -oE '[0-9]{3}_S_[0-9]{4}' | head -1)

    if [[ -z "$subject" ]]; then
        echo "❌ No subject found in path: $imgdir"
        continue
    fi

    group="${GROUP_MAP[$subject]}"

    if [[ -z "$group" ]]; then
        echo "❌ No diagnosis for subject: $subject"
        continue
    fi

    echo "✔ I$image_id → $subject → $group"

    dcm2niix \
    -z y \
    -f "${subject}_I${image_id}" \
    -o "$OUT/$group" \
    "$imgdir"

# =========================
# REMOVE JSON FILES (cleanup)
# =========================
find "$OUT/$group" -maxdepth 1 -type f -name "${subject}_I${image_id}*.json" -delete

done

echo "DONE"