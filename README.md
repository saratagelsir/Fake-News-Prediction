# Fake News Prediction

## Code Structure and Location

The code structure as of now includes:
- **data** folder: trained model and data used for training files
- **def** folder: contains definition json-files
- **output** folder: contains model output files
- "**\*.py**": python code files

## How to Run the Model

While you are in the main folder of the model, you can execute the model in 2 modes ("training" or "prediction") by the following:

To call the entry function/method of the model from an instance of model object (for example in Predix):

    ```python
    from FakeNewsPrediction.FakeNewsPrediction import FakeNewsPrediction
    caller_string = '{"modelrun":"news_file_prediction","newsurl":0}'
    mdl = FakeNewsPrediction() # Create an empty instance of the object/class FakeNewsPrediction
    model_output = mdl.main(caller_string)
	```

*TODO* complete and enhance this documentation later!