# reidammer.github.io

1. To clone the repo: git clone https://github.com/reidammer/CV5524.git

To install the requirements: pip install -r requirements.txt


2. Download link for models:

https://drive.google.com/file/d/1C2yE2jOLNvychw7TMjY13vWP4nholIDD/view

https://drive.google.com/file/d/1B-QfvBNF6EweVxHIU9yQiQGovah7P3vb/view

Put these in the CV5524 folder alongside main.py and the jupyter notebook.

3. To test these models, run the final cell in the jupyter notebook in the repo listed above.


4. For main.py basic approach:
use --current_step 4 --display --calibration test_case_1 --curve test_case_1 --detection_threshold 0.92 to run full program

for terminal running final code (basic approach), use:
python3 main.py --current_step 4 --display --calibration test_case_1 --detection_threshold 0.92 --approach basic --curve cubic
----> use test_case_1 calibration plate but any curve file

To generate data for advanced approach: python3 data_generation.py

To test advanced approach: run final cell in jupyter notebook