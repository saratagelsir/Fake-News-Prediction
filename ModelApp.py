import os
import sys

import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FakeNewsPrediction.FakeNewsPrediction import FakeNewsPrediction


def main_caller(news, news_type):
    # Define the caller json string
    modelrun = news_type.lower().replace(' ', '_') + '_prediction'
    caller_string = '{"modelrun":"' + modelrun + '","news_contents":"' + news + '"}'

    mdl = FakeNewsPrediction()
    preds_interpreter = mdl.main(caller_string)
    model_output = preds_interpreter[0]
    return model_output


info = 'Machine learning based app for fake news detection'
iface = gr.Interface(fn=main_caller,
    inputs=[gr.inputs.Textbox(lines=20, placeholder="News Text or URL Here..."), gr.inputs.Radio(['General News', 'COVID19 News'])], outputs='text',
    title='Fake News Prediction', description=info)
iface.launch(inline=False, share=False)
