# Sound it Out

## Installation

Requires python 3.10+. All other dependencies are listed in [requirements.txt](requirements.txt). You may want to install the gpu version of pytorch first, the cpu-version will be installed by default.

## Directory Structure

The current directory structure is as shown:

```
src/
  preprocessing/
    g2p/phonemizer/
      The custom fork of the phonemizer tool that includes number preservation.
    p2g/
      Includes the tools used for converting text to IPA/Romanization and back.
  pre_training/moddednanogpt/
    The moddednanogpt code used to pre-train the models.
  fine_tuning/
    The code used to fine-tune the models, specifically the training portion of fine-tuning.
  evaluation/
    The code used to evaluate the fine-tuned models.
scripts/
  Sample scripts used to verify the functionality of the code.
```

## Running the Code

You'll need to run all of the scripts from the root of the github repository (ie. `./scripts/foo.sh` and not `cd scripts; ./foo.sh`).

