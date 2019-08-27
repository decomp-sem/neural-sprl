# Neural Semantic Proto-Role Labeling

This is the state-of-the-art SPRL system, described in full detail [here](https://arxiv.org/abs/1804.07976).

If you have questions about this repository, please contact Rachel Rudinger: rudinger AT jhu DOT edu.

## Instructions

1. Clone the repository. In release v0.1.0, the pretrained torch models and word
embeddings are included. Place these files in the models/ subdirectory.

2. Requirements: If you use Ananconda, requirements can be installed with
[requirements.yml](requirements.yml). This code has been tested with pytorch v0.2.0 only.

3. The main labeling script is [source/predict.py](source/predict.py), which
runs by default on the sample data [data/mini.sprl](data/mini.sprl):

```
cd source
python predict.py
```

For help with customized arguments, run:

```
python predict.py --help
```

The GPU id is not set within predict.py. If you want to run with GPU, you must
set the correct GPU id externally, e.g.:

```
FREE_GPU=??? # get id of free gpu on your system
CUDA_VISIBLE_DEVICES=$FREE_GPU python predict.py --gpu
```

4. predict.py assumes an input file in .json format. See [data/mini.sprl](data/mini.sprl)
for an example. Output is identical .json structure, with additional field for
SPR label predictions.

## Errata

An earlier version of this paper contained transcription errors in Appendex Tables 7 and 8. Please consult the most recent version of the paper on [arxiv.org](https://arxiv.org) for the corrected version.
