# instrument-classifier
Uses machine learning to classify audio samples by musical instrument.

### Data Sources

[University of Iowa Electronic Music Studios Musical Instrument Sample Database](http://theremin.music.uiowa.edu/MIS-Pitches-2012)

[UK Philharmonia Orchestra Sound Samples](https://www.philharmonia.co.uk/explore/sound_samples)

### Data Technologies Used

Python

  - BeautifulSoup for web scraping
  - pandas for data wrangling
  - librosa for audio processing
  - matplotlib for visualization

AWS

 - S3 for raw audio samples
 - RDS (MySQL) for signal-processed data

### Under the Hood

Onset detection is used to isolate the time of the attack. The spectral content in four audio frames near the attack is measured across 28 dimensions, giving a 112 dimensional representation of each sample.

This resulting representation is then classified using XGBoost with 500 estimators.

The classifier is trained on 24 distinct instrument types representing common orchestral music.

### Results

We obtain an 83% accuracy rate on test data, which is a 4.5 times improvement over the baseline model.
