## Requirements:
- rdkit == 2019.09.1.0
- transformer == 4.2.2
- python 3.7

## Generate features:
### Generate feature with SMILES input
python3 extract_feature.py --input_smile = "C1CCCCC1C2CCCCC2"

### Generate feature with *csv file containing SMILES inputs
python3 extract_feature.py -- input_file= 'in_PATH/Intput.csv' --output_file = 'OUT_PATH/Output.csv'

## Download Pre-trained model:
[Link](https://drive.google.com/file/d/1e4-weMgCDEro3XdHrZoboNp4-OH6TMBx/view?usp=sharing)
