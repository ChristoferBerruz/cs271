## Guide to using InferSent model

InferSent is a technique for generating semantic representations of 
English sentences. It is trained on natural language inference 
datasets, allowing it to generalize effectively across a wide range of tasks.

In this project, we use Meta's pretrained InferSent model from the
[Facebook Research InferSent repository](https://github.com/facebookresearch/InferSent). 
Specifically, this project uses the fastText (V2) vectors trained with InferSent (V2) model.

### Downloading the InferSent Model
After cloning this repository, download the publicly available fastText (V2) vectors and the 
InferSent (V2) model to the InferSent directory:
```bash
curl -Lo Infersent/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip Infersent/crawl-300d-2M.vec.zip -d Infersent/
curl -Lo Infersent/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl 
```
