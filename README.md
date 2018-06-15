Final Code for Machine Perception Course Project at ETH (263-3710-00L)

## Setup

The following two steps will prepare your environment to begin training and evaluating models.

1. Install dependencies by running (with `sudo` appended if necessary)
```
python3 setup.py install
```
2. Download training, validation and test datasets from kaggle project page.
3. Update data and output paths in `config.py`.
4. Train the model provided with source code by running 
```
python3 training.py
```
4. When your model has completed training, run the evaluation on the test set by running
```
python3 restore_and_evaluate.py \
    --model_name <experience_name> \
    --checkpoint_id 11440
```
This output can be found in the folder `runs/<experiment_name>/` as `submission_<experiment_name>.csv`.

