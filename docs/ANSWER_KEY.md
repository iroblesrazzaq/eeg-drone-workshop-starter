# Instructor answer key

1. Imagined movement of both hands versus both feet. The features describe power in two frequency bands for each of three electrodes. They discard phase, precise waveform timing, and activity captured outside those channels and bands.
2. Nearby overlapping windows share EEG samples and artifacts. Splitting first keeps those samples out of both sides. Holding out a person tests transfer across people. The workshop evaluates transfer to another recording from the same person.
3. Training fits the scaler and model. Validation selects C. Test estimates performance after that choice. The drone segment is reserved for replay. Using the test score to pick a model makes that split another validation set.
4. With the correct label hands, probabilities of 0.51 and 0.99 both classify correctly, but the second prediction has lower log loss. A very confident wrong prediction is penalized heavily. Pick C at the lowest validation loss.
5. The only fitted classes are hands and feet. The game reads the recorded cue timing to allow motion during task windows and resets the probability history when the task ends. An uncued system would need its own rest or intent detection, or explicit arming.
6. More frequent predictions shorten the time between decisions but their windows overlap more. A longer average generally reduces jitter and delays changes. With the same number of predictions, a faster cadence also makes the average cover less time. The two-second history and the onset delay still apply.
7. For example, averaging 0.52, 0.65, and 0.78 gives 0.65. With a 0.60 upper threshold, that commands up. A wider hover range needs a stronger probability to move, which may reduce unwanted movement and suppress some intended commands.
