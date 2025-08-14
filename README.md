# Emotion-Detection

## About

This project was trained on [kaggle's FER 2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and built using [PyTorch](https://pytorch.org) version 2.7.1, [D3](https://d3js.org) version 7.9.0,
[opencv-python](https://pypi.org/project/opencv-python/) version 4.12.0.88 and [face-recognition](https://pypi.org/project/face-recognition/) version 1.3.0.  
The system detects emotions using PyTorch and visualizes them using D3. Additionally, the attention level is estimated using opencv. face-recognition is used to rename and memorize face's during multiple runs per program execution.

## Getting Started

To run a local copy of the project, please follow the instructions below.

### Train and Test Data

datasets.py expects to find the two folder with test and training data to be found in:

```bash
root/data
```

which should contain two folders test and train received from [kaggle's FER 2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).

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

<!-- LICENSE -->
## License

Distributed under CC-0. See `LICENSE.txt` for more information.


<!-- CONTACT -->
## Contact

Michael Redl - [Personal Website](https://michaeljosefredl.at) - [@redl_m](https://www.instagram.com/redl__m/) - michael.redl14042004@gmail.com

Project Link: [https://github.com/redl-m/Emotion-Detection](https://github.com/redl-m/Emotion-Detection)
