# heart rate physiological measure
ML-based physiological biometric authentication using heart rate, built on open Scientific Data datasets, coded in Python.

# Overview 
- This research project explores anomaly detection in a single physiological measure (heart rate) over 1 month, with data collected from 49 individuals. Wearable data were collected using smartwatches and were used here to test whether a single measure is sufficient to flag strangers. However, the findings suggest the need for a multi-fusion system to create a digital-safe concept built upon biometrics.

# Results 
Evaluation metric:
- (FRR) False Rejection Rate.
- (FAR) False Acceptance Rate.
  
key result: 
- Person A: 10% flagged, 90% acceptance FRR.
- Person B: 16.3% flagged, 83.7% FAR.
- Person C: 9.7% flagged, 90.3% FAR.
- Person D: 22.5% flagged, 77.5% FAR.
- Person F: 57.8% flagged, 42.2% FAR.
 
# Details on Dataset Used
Data source:
1. Baigutanova, A., Park, S., Constantinides, M., Lee, S. W., Quercia, D., & Cha, M. (2025). A continuous real-world dataset comprising wearable-based heart rate variability alongside sleep diaries. Scientific data, 12(1), 1474. https://doi.org/10.1038/s41597-025-05801-3

2. Baigutanova, Aitolkyn; Park, Sungkyu; Constantinides, Marios; Lee, Sang Won; Quercia, Daniele; Cha, Meeyoung (2025). In-situ wearable-based dataset of continuous heart rate variability monitoring accompanied by sleep diaries. figshare. Dataset. https://doi.org/10.6084/m9.figshare.28509740.v1

Brief Description of The Dataset:
- This dataset is based on continuous heart rate measurements collected from 49 healthy individuals over 4 weeks (mean age: 28.35±5.87, including 51% females). The recordings were sampled every 100 ms (10 Hz), accounting for short-term HRV computation for each 5-minute segment of raw data.

Data Handling:
- For this ML model, data was used raw without deep cleaning. While charging watches, gaps and sensor dropouts are noticeable, showing that real-world biometric collection has limitations. Therefore, data cleaning was not carried out, accounting for realistic assumptions and deployment in later stages; instead, a quality threshold was introduced to exclude incorrect readings.

- Data was split so that person A served as the baseline, simulating an owner of hardware with 1-month heart rate data depicting unique patterns. The remaining 48 people were used as a test group against person A, taking their instances of touch, which is equivalent to about 10 heart rate readings per person. For this set, 5 people were tested against person A: persons B, C, D, and F. These individuals were filtered by their device ID, missingness_score threshold, and extraction of HR values.

Model Training:
- The model training was carried out using One-Class-SVM; the boundary is around person A's 1-month pattern.
  
- The parameters were readjusted after the first attempt to get better results; the initial model (nu=0.05,gamma='auto') flagged approximately 50% of person A's own data as anomalies the final attempt, gamma was set to scale for a broader decision boundary, and nu was adjusted to 0.1 to allow 10% anomalies in the baseline. The final model flagged 10% of person A's data, achieving 73% detection of stranger readings.

# Summary & Next Steps
Impact: This suggests the need for a multi-fusion system for identity authorization. Therefore, avoid relying on a single measure.

Next step: 
- Test the other singular physiological measures and expand algorithm selection for testing; make use of isolation forest and autoencoder.

- Create synthetic data for a multi fusion model to test against single measures and validate the feasibility of creating a digital safe. 






  
