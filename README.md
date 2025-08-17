# Emotion-Detection

## About

This project was trained on [kaggle's FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and built using [PyTorch](https://pytorch.org) version 2.7.1, [D3](https://d3js.org) version 7.9.0, [transformers](https://github.com/huggingface/transformers) version 4.55.2, [accelerate](https://github.com/huggingface/accelerate) version 1.10.0, [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) version 0.47.0, [OpenCV](https://opencv.org) version 4.12.0.88 and [dlib's face-recognition](https://dlib.net) version 1.3.0.  
The system detects emotions using PyTorch and visualizes them using D3. Additionally, the attention level is estimated using OpenCV. dlib's face-recognition is used to rename and memorize faces during multiple runs per program execution. Tracked emotions can be summarized using either a
1. hard coded, heuristic approach
2. local LLM
3. remote LLM using an API URL and API key.

See [Program Usage](##program-usage) for instructions on how to use and adjust the local and remote LLM.

## Getting Started

To run a local copy of the project, please follow the instructions below.

### Data

datasets.py expects to find the two folder with test and training data to be found in: root/data, which should contain two folders - test and train - received from [kaggle's FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).

### Development server

To start a local development server, navigate to the server:

```bash
cd .\server\
```

and run:

```bash
python app.py
```

or run:

```bash
python .\server\app.py
```

directly.

Once the server is running, open your browser and navigate to `http://localhost:5000/`.

## Program Usage

### Basics

To start an emotion tracking session, start the live feed in the page header or upload a video in the header or by clicking on the canvas.  
Emotion probabilities for the current frame and their development over time will be visualized right of the canvas. Below the two charts visualizing emotions is the visualization of estimated engagement in percent.  
Set a summary method before clicking the "Stop & Summarize" on the top right. Available modes are heuristic summary, local LLM and remote LLM using an API.

### Status Summaries

Status summaries are located at the top right corner and show the avilability of a local LLM model, CUDA cores, an API URL and API key.

### Changing default values

The default values for the local LLM, the API URL and API key can be set under the global settings configuration located at the top of app.py.  
 To change the default local LLM model adjust the parameter DEFAULT_LOCAL_LLM_MODEL_NAME,  
 to change the default API URL adjust the parameter DEFAULT_LLM_API_URL and  
 to set a default API key adjust LLM_API_KEY.  
 See [huggingface.co/models](https://huggingface.co/models) to browse available local LLM models.

 
### Setting up the local LLM

The default local LLM used for this project is tiiuae/falcon-7b-instruct. It was chosen for its reasonable ratio of model size and processing time. On first use, huggingface will download the model if not present.  
Even tough tiiuae/falcon-7b-instruct is relatively fast regarding computation time, it can take up to several minutes to generate a summary and be extremely memory expensive.  
On program start, the local LLM model can be set. Previous LLM models will be stored using the browser's history.

### Setting up the remote LLM

The default remote LLM API is OpenAI's API.  
On program start, the remote API URL can be set. Previous API URLs will be stored using the browser's history.  
The API key is not set by default, but can be set on program start. Previous API keys will not be stored.


<!-- LICENSE -->
## License

Distributed under CC-0. See `LICENSE.txt` for more information.


<!-- CONTACT -->
## Contact

Michael Redl - [Personal Website](https://michaeljosefredl.at) - [@redl_m](https://www.instagram.com/redl__m/) - michael.redl14042004@gmail.com

Project Link: [https://github.com/redl-m/Emotion-Detection](https://github.com/redl-m/Emotion-Detection)
