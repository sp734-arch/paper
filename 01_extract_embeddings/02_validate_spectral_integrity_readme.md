## Data Quality Validation Utility (Optional)

After extraction, validate the spectral integrity to ensure recordings are full-bandwidth:
```bash
python validate_spectral_integrity.py \
    --dataset-path ./features_audit/density_50/features_english \
    --dataset-name "English (Telephone)" \
    --output validation_results.csv
```

**What this checks:**
- Detects 8kHz bandwidth limitations (PSTN telephony artifacts)
- Measures high-frequency energy in HeAR dimensions 32-511
- Flags embeddings with suspiciously low high-frequency content

**When to run:**
- After adding a new dataset
- When metadata mentions "telephone" or "phone" recordings
- To verify consistent spectral coverage across cohorts

**Expected result:**
- ✅ 100% pass rate = full-spectrum recordings (modern smartphones)
- ❌ <100% pass rate = possible legacy telephony codecs (investigate)

**Paper context:**
The English training corpus (Prior et al., 2023) is labeled "participant telephones"
but showed 100% pass rate on this validation, indicating modern smartphone recordings
with full bandwidth rather than legacy PSTN (8kHz) telephony.
```

