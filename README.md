# Multi
Multi_AI project

This model's code is implemented using [Tinygrad](https://github.com/tinygrad/tinygrad).

> **Note:** The paper associated with this project is currently under review. Model weights and the dataset will be released in the near future.

The training parameters of the model are configured in the YAML files located in the `config` directory.

Before using the model for prediction, make sure to configure the number of processes and the prediction batch size in `Multi/config/global_cfg.yaml`.

You can train the denoising model using `python denoise_model.py`.  
An example of correct execution is shown in `denoise_model_process.jpg`.

You can train the classification model using `python train.py`.  
An example of correct execution is shown in `train_model_process.jpg`.

You can fine-tune a pre-trained model using `python finetune.py`.  
An example of correct execution is shown in `finetune_process.jpg`.

The command to predict time-domain pulsar candidates is as follows:

```shell
python predict.py --ckpt ./trained_model/weight_0.9954_0.9830.pth --outfile clf_result.txt --pfd_dataloader test_dataloader.pkl --pfd_file ./test_pfdfile.txt --use_prob
python predict.py --ckpt ./trained_model/weight_0.9954_0.9830.pth --outfile ./test_pfdfile.txt --pfd_dir ./test_pfds/
```

A more detailed user doc will be released soon.