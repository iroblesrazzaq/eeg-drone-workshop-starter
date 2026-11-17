# Workshop exit questions

Use your plots or flight log to support your answers where you can.

1. What are the model's two classes? What do the six features summarize, and what information do they discard?
2. Why do we split the recordings before making overlapping windows? What would holding out an entire person let us test?
3. What is the job of each split? What changes if you choose a model after looking at the test score?
4. How could two classifiers have equal accuracy but different log loss? Which plot guides your choice of C?
5. Why can a hands-versus-feet model issue a confident prediction while a person is resting? How does our game handle rest?
6. You classify every 250 ms using two seconds of EEG and average the last three predictions. What happens if you classify twice as often or average twice as many predictions?
7. Pick one prediction from the flight log. Trace how its raw probability, previous probabilities, and your thresholds determined the command. What would a wider hover range change?
