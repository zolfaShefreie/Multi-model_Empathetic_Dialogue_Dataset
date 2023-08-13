# Multi-model_Empathetic_Dialogue_Dataset
This project is implementation of stages that are used for creating multi-model empathetic dialogue dataset from available datasets. four available dataset is considered to format, including:
- MELD
- DailyTalk
- AnnoMI
- MUStARD
## Setup
### Prerequisites for running
First of all run below code to clone the repository
```
git clone https://github.com/zolfaShefreie/Multi-model_Empathetic_Dialogue_Dataset.git 
```
Make an envierment and run below command to install all required packages
```
pip install -r requirements.txt
```
### Commands
Three scripts is implemented to reformat datasets, including:
1. dataset_formatter_info.py</br>
   This script gives you information about what is the stages of specific dataset.
   ```
   python dataset_formatter_info.py --dataset_name {dataset_name}
   ```
2. formatter_running_tracker.py</br>
   This script shows what are run stages for specific dataset.
   ```
   python formatter_running_tracker.py --dataset_name {dataset_name}
   ```
5. run_dataset_formatter.py</br>
   This script run the formatter stages for specific dataset.
   ```
   python run_dataset_formatter.py --dataset_name {dataset_name} --dataset_dir {dir of dataset} --save_at_dir {save at dir} --start_stage {start_stage} --stop_stage {stop_stage}
   ```
## Stages
