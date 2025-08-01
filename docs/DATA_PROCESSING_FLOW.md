# Data Processing Flow: 3 Sources to Final Dataset

## Overview
The system processes data from 3 different sources and combines them into a unified, labeled dataset for training the gesture recognition model.

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                      │
├─────────────────┬─────────────────┬─────────────────────────────────────────┤
│   SOURCE 1      │   SOURCE 2      │   SOURCE 3                              │
│   RAW DATA      │   SYNTHETIC     │   EXTERNAL DATASETS                     │
│                 │   DATA          │                                         │
└─────────────────┴─────────────────┴─────────────────────────────────────────┘
         │                   │                           │
         ▼                   ▼                           ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────────────────┐
│  PROCESSING     │ │  GENERATION     │ │  INTEGRATION                        │
│  • Load JSON    │ │  • Templates    │ │  • Format conversion                │
│  • Augment      │ │  • Hand anatomy │ │  • Label mapping                    │
│  • Transform    │ │  • Variations   │ │  • Quality check                    │
└─────────────────┘ └─────────────────┘ └─────────────────────────────────────┘
         │                   │                           │
         ▼                   ▼                           ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────────────────┐
│  OUTPUT         │ │  OUTPUT         │ │  OUTPUT                              │
│  • Augmented    │ │  • Synthetic    │ │  • Converted                        │
│  • Labeled      │ │  • Labeled      │ │  • Labeled                          │
│  • Metadata     │ │  • Metadata     │ │  • Metadata                         │
└─────────────────┘ └─────────────────┘ └─────────────────────────────────────┘
         │                   │                           │
         └───────────────────┼───────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA COMBINATION                                     │
│  • Merge all sources                                                        │
│  • Standardize format                                                       │
│  • Validate labels                                                          │
│  • Add source tracking                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FINAL DATASET                                        │
│  • Unified format (JSON)                                                    │
│  • Consistent labels                                                        │
│  • Source metadata                                                          │
│  • Quality indicators                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Processing Steps

### Source 1: Raw Data Processing

**Input:** Existing JSON files in `data/raw/`
```json
{
  "gesture": "fist",
  "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.3}, ...],
  "sample_id": "sample_0000"
}
```

**Processing:**
1. **Load** existing samples
2. **Augment** with transformations:
   - Random rotation (±15°)
   - Random scaling (0.9x-1.1x)
   - Random translation
   - Gaussian noise
   - Finger position variations
3. **Label:** Keep original gesture labels
4. **Metadata:** Add `source: 'raw'`, `augmented: true`

**Output:**
```json
{
  "gesture": "fist",
  "landmarks": [{"x": 0.12, "y": 0.18, "z": 0.31}, ...],
  "source": "raw",
  "augmented": true,
  "original_sample": "sample_0000"
}
```

### Source 2: Synthetic Data Generation

**Input:** Gesture templates
```python
gesture_templates = {
    'fist': {
        'finger_extensions': [0.1, 0.1, 0.1, 0.1, 0.1],
        'description': 'All fingers closed'
    }
}
```

**Processing:**
1. **Generate** base hand landmarks (21 points)
2. **Apply** gesture-specific templates
3. **Add** randomness for variation
4. **Label:** Based on template definition
5. **Metadata:** Add `source: 'synthetic'`, `template: {...}`

**Output:**
```json
{
  "gesture": "fist",
  "landmarks": [{"x": 0.0, "y": 0.0, "z": 0.0}, ...],
  "source": "synthetic",
  "template": {"finger_extensions": [0.1, 0.1, 0.1, 0.1, 0.1]},
  "synthetic_id": "synthetic_fist_001"
}
```

### Source 3: External Dataset Integration

**Input:** Various formats (CSV, JSON, XML)
```csv
id,gesture_label,x1,y1,z1,x2,y2,z2,...
external_001,fist,0.1,0.2,0.3,0.4,0.5,0.6,...
```

**Processing:**
1. **Load** external dataset
2. **Convert** format to standard landmarks
3. **Map** external labels to our classes
4. **Validate** data quality
5. **Metadata:** Add `source: 'external'`, `confidence: 0.95`

**Output:**
```json
{
  "gesture": "fist",
  "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.3}, ...],
  "source": "external",
  "original_id": "external_001",
  "confidence": 0.95
}
```

## Labeling Strategy

### Consistent Labeling Across Sources

All sources use the same gesture labels:
- `clap`
- `fist`
- `point`
- `thumbs_up`
- `wave`
- `peace`
- `open_hand`

### Source Tracking

Each sample includes source metadata:
```json
{
  "source": "raw|synthetic|external",
  "metadata": {
    "original_id": "sample_0000",
    "confidence": 0.95,
    "augmented": true,
    "synthetic": false
  }
}
```

### Quality Indicators

- **Confidence:** External dataset confidence scores
- **Augmented:** Flag for augmented samples
- **Synthetic:** Flag for synthetic samples
- **Original ID:** Track data lineage

## Data Validation

### Format Validation
- ✅ 21 landmark points per sample
- ✅ Consistent coordinate system
- ✅ Valid gesture labels
- ✅ Required metadata fields

### Quality Checks
- ✅ Landmark coordinate ranges
- ✅ Gesture consistency
- ✅ Source attribution
- ✅ Timestamp validation

## Final Dataset Structure

```json
{
  "id": "combined_0001",
  "gesture": "fist",
  "landmarks": [
    {"x": 0.0, "y": 0.0, "z": 0.0},  // Wrist
    {"x": 0.1, "y": 0.0, "z": 0.0},  // Thumb base
    // ... 19 more landmarks
  ],
  "timestamp": "2024-01-15T10:30:00Z",
  "source": "raw",
  "metadata": {
    "original_id": "sample_0000",
    "confidence": 1.0,
    "augmented": true,
    "synthetic": false
  }
}
```

## Benefits of This Approach

1. **Data Diversity:** Multiple sources provide varied samples
2. **Quality Control:** Source tracking enables quality assessment
3. **Scalability:** Easy to add new data sources
4. **Reproducibility:** Clear data lineage and processing steps
5. **Flexibility:** Supports different input formats

## Usage Example

```python
# Run the data integration demo
python scripts/data_integration_demo.py

# This will show:
# 1. How each source is processed
# 2. Labeling consistency across sources
# 3. Final combined dataset structure
# 4. Data distribution statistics
``` 