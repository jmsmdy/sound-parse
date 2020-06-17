# sound-shift

Automated music transcription using custom TensorFlow 2 deep learning.

### Generating Training Data

Random "compositions" were created by generating piano roll notation, then using this piano roll notation to generate corresponding audio files using samples from [University of Iowa Electronic Music Studios Musical Instrument Sample Database](http://theremin.music.uiowa.edu/MIS-Pitches-2012)

### Network Architecture

Input audio is divided into overlapping 23ms segments, and the Discrete Fourier Transforms of these segments are fed into an LSTM, which makes predictions for which notes are playing. 

### Why do Fourier Preprocessing?

Although in principle a neural network could learn to isolate pitches without Fourier pre-processing, a spectral representation is more amenable to it, because it turns the problem into a peak-finding exercise. The following images of a (log transformed) STFT and power spectrogram show how the fundamental frequency can be more easily isolated in a spectral representation:

![fourier](fourier.png)
![spectrogram](spectrogram.png)

### Limitations

Due to concerns with accurately detecting extended notes and different instruments, we restricted to transcribing marimba, as their percussive attacks are easier to identify (as discovered by previous work), and among pitched percussion instruments we found pitch detection performed best on marimba. The following plot of the first two principal components of samples colored by instrument shows why we chose marimba as our instrument of choice to transcribe instead of opting for instrument detection:

![PCA](PCA.png)


We hope in the future to expand the features we can extract / parse from an input audio file.

### Training

Training was completed over the course of a few hours on a local machine. The loss is based on "cosine similarity", which experimentally performed better than mean squared error or note-wise binary cross-entropy. 


### Reuslts

After training, the network was able to accurately reconstruct the original piano roll notation, with only slight errors in note onset. 

![results](results.jpg)

Caveat: this should be taken with a grain of salt, since test data was constructed using the same samples as training data, so it is not clear how well this would generalize.
