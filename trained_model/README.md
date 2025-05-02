# Our trained models

We created a total of five models which has different distribution of Transformed Kd value dependent on the domain or overall dataset. To explore how we trained and more about the architecture of the model, [click here](../toolkit_to_train).
In this README, I will be going through the example of our main model-- [overall_prediction](overall_prediction). 

<img src="overall_prediction/information/loss_curve.png" width="300"> <img src="overall_prediction/information/accuracy_curve.png" width="300"> <img src="overall_prediction/information/r_square_predicted_actual.png" width="300">

**Performance Metrics**:
- Training Progress: Converged over 100 epochs with final loss of ~3.6; no signs of overfitting
- Validation Performance:
  - R^2 for validating: 0.82
  - Overall Accuracy: ~70%
  - TOp 25% accuracy: ~85%
- Testing Performance:
   - R^2 for testing: 0.77


## How to run and create a prediction
In this README, I will be using the overall_prediction model as an example on how to run and create a prediction. 
