import os
import sys
import json

import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FakeNewsPrediction.FakeNewsPrediction import FakeNewsPrediction


def main_caller(news, news_type):
    # Define the caller json string
    modelrun = news_type.lower().replace(" ", "_") + "_prediction"
    caller_string = json.dumps({"modelrun": modelrun, "news_contents": news})

    mdl = FakeNewsPrediction()
    preds_interpreter = mdl.main(caller_string)

    # mdl.main() sometimes returns a JSON string on failure; keep UI robust
    if isinstance(preds_interpreter, str):
        return preds_interpreter

    model_output = preds_interpreter[0] if isinstance(preds_interpreter, (list, tuple)) and preds_interpreter else preds_interpreter
    return model_output


info = "Machine learning based app for fake news detection"
iface = gr.Interface(
    fn=main_caller,
    inputs=[gr.Textbox(lines=20, placeholder="News Text or URL Here..."), gr.Radio(["General News", "COVID19 News"])],
    outputs=gr.Textbox(),
    title="Fake News Prediction",
    description=info,
)
iface.launch(inline=False, share=False)