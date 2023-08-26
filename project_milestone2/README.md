[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/OKlRJRN2)
# README
## Project Milestone 2 Description
- Don't forget to read [the project description](https://docs.google.com/document/d/1SY1HAfrpoj9B6FnO3LEChne4vdf1GOuswu-H7oUUt8A/edit) before starting milestone 2.
- All references for the project, such as submission examples or tutorials will be in [this project reference folder](https://drive.google.com/drive/folders/1rc2w25A5_HfI3ieHxs4ya9UaiUO41dXz?usp=sharing).
- For a detailed documentation on how to use our GPT Wrapper package, please read [this document](https://docs.google.com/document/d/1ZifVg2lw0EzeiuyT20DvZz90GBi3RsoL5tOw22a7BK0/edit?usp=sharing) that is also in the reference folder.
- For a detailed documentation on how to use our GCP, please read [this document](https://docs.google.com/presentation/d/1GJqog51fZ4Yqkw6y0HsS1u28ggPaSWMMgKOAqi7gY1c/edit#slide=id.p) that is also in the reference folder.
    
## Deliverables

The second milestone deliverables require you to commit the following files in your github classroom repository:

- ✅ The reward model training dataset you've constructed, as detailed in the [project description](https://docs.google.com/document/d/1SY1HAfrpoj9B6FnO3LEChne4vdf1GOuswu-H7oUUt8A/edit) and referenced in [the project reference folder](https://drive.google.com/drive/folders/1rc2w25A5_HfI3ieHxs4ya9UaiUO41dXz?usp=sharing). You can also see the same example in this repository called `m2_reward_dataset_example.json`. You should name these as `m2_reward_dataset_<team_name>_<datasource>.json`. There may be multiple if you are using different sources.

- ✅ The reward model itself, as detailed in the [project description](https://docs.google.com/document/d/1SY1HAfrpoj9B6FnO3LEChne4vdf1GOuswu-H7oUUt8A/edit). You can also see an example of it in this repository called `models/reward-model` after you run the `evaluate.py` script.

Please verify that your dataset format and model are conforming with our constraints by running them with the `evaluate.py` example (explained in the following sections). 

## Creating your reward model dataset

What type of resources can you use to create your reward model dataset?
1. You are welcome to use the interactions we distributed you. These are part of the interactions that all students have collected over milestone 1. We will update this in the [the project reference folder](https://drive.google.com/drive/folders/1rc2w25A5_HfI3ieHxs4ya9UaiUO41dXz?usp=sharing) as more people submit their interactions.
    - There is one interactions json file where each dictionary has an interaction with a matching solution id and a confidence score.
    - There is also a solutions json file that you can match to interactions with the `sol_id`. Not all questions have the `answer` key so don't let the naming of `sol_id` fool you. This file has more metadata on each question such as the original question (`question`), the choices if it's an MCQ or TF question (`choices`), the solutions (`answer`), explanations (`explanation`) and so forth.
    - You are free to use both these files to build your reward modeling dataset.
2. You are welcome to create more interactions with ChatGPT with the GPTWrapper package and include those in the reward modeling dataset.
3. Aside from this, you are also welcome to augment the dataset as you wish, as long as you document the source of your data.

When submitting this dataset, please follow the format specified in `m2_reward_dataset_example.json`. This means that your chats should be concatenated with double new lines "\n\n" and that you should start each interaction by specifying the role such as "Human: ", "Assistant: ". Since this is a reward model, you do not need to handle the system interactions. If you wish to do so you can preempt the text with "System: ".

For each source of data you collect (e.g., demonstrations from ChatGPT, data found online, etc.), please submit a separate data file. In your final report (not a deliverable in this milestone), you should discuss the legal and ethical considerations of the data you collect.

## Verifying your model and dataset
The `evaluate.py` file serves as an example on how to load your model and verify that your dataset is in the correct format. This script will not be how we finally assess your datasets and models, therefore do not try to optimize over the accuracy metric.

- You can run your dataset with a simple pretrained model like "OpenAssistant/reward-model-deberta-v3-base".
- For the model, you should make sure that you can load it with the `load_model` function provided in the `Evaluator` class.
- You should make sure to specify the `problem_type` in the `config.json` file located inside the `models/reward-model` folder. There are only two options for `problem_type`:  `classification` or `regression`, depending on how you plan to implement your reward model. 
- Please read carefully the evaluation code to make sure that your model produces the correct output either for `classification` or `regression`.

Here we provide you some basic examples on how to use the `evaluate.py` file:

- To save the reward model that you have trained into a Huggingface pre-trained model:

```python
# If you are loading the save_hf_model from "evaluate.py". 
# Ignore if you copy pasted the "save_hf_model" function in your code already. 
from evaluate import  save_hf_model

tokenizer = YourRewardModelTokenizer()
model = YourRewardModelImplementation()
model_path = "path to your model"

save_hf_model(model, tokenizer, model_path)
```

The model will be saved to `model_path`. Now, specify `problem_type` in `config.json` (located in the model path) to be either `classification` or `regression`, for example:

``` json
...
"problem_type": "regression",
"relative_attention": true,
"share_att_key": true,
...
```

- To run the evaluation code for verification:
```bash
python evaluate.py --model_path path-to-your-model --data_path path-to-your-data
```
You should get results in the format of the following print outs:
```
Evaluation Complete, Accuracy: 0.5
```


## Requirement Reminders: API Keys and TA assignments

You should also have received an API key from an ML4ED lab member (if you filled out the consent form), which you will need to access our GPT Wrapper package
**This API key is very important as it will sign you interactions.**

Moreover, please contact your assigned team TA for help instead of reaching out to the mailing list.