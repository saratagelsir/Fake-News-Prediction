import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FakeNewsPrediction.FakeNewsPrediction import FakeNewsPrediction

if __name__ == "__main__":
    # Define the caller json string
    modelrun = sys.argv[1] if len(sys.argv) > 1 else "general_news_prediction"
    # modelrun = sys.argv[1] if len(sys.argv) > 1 else "covid19_news_prediction"

    # Define the caller json string
    news = "https://www.bbc.com/news/world-africa-59033142"
    caller_string = '{"modelrun":"%s","news_contents":"%s"}' % (modelrun, news)

    # Run the model: python ModelTesting [Model_Run]
    mdl = FakeNewsPrediction()
    model_output = mdl.main(caller_string)

# python ModelTesting.py covid19_news_prediction
# python ModelTesting.py covid19_news_training
