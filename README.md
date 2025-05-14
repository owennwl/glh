# GLH: Google Location History Analysis

Code used in CU-COVID19 smartphone mobility data analysis.

## Overview

This repository contains code for analyzing relationships between individual mobility data from Google Location History (GLH), population mobility from Google Community Mobility Reports, and mental health outcomes during the COVID-19 pandemic.

## Repository Contents

### Data Processing
- **preprocess_glh.py**: Converts raw Google Location History geolocation data into structured format
- **mobility_indices.py**: Transforms preprocessed location data into mobility indices for analysis

### Analysis
- **concurrent_analysis_mplus.inp**: Mplus code for analyzing concurrent relationships between:
  - Population mobility (Google Community Mobility Reports residential stay)
  - Individual mobility (Google Location History)
  - Mental health outcomes
  
- **longitudinal_analysis.py**: Python code for analyzing relationships between individual mobility and mental health outcomes over time
