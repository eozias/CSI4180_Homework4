# CSI4180_Homework4
 
Homework 4 for CSI 4180 (NLP App)

Please visit my app at https://huggingface.co/spaces/emma7897/CSI_4180_Final_Project

**Files:**

userGuide.pdf: Required file for submitting this assignment. Not needed for running the code.

fine_tuning_number_one.py: This file was created using Google Colab. It contains code for fine tuning my BERT and DistilBERT models on a dataset for mask filling.

fine_tuning_number_two.py: This file was created using Google Colab. It contains code for fine tuning my BERT and DistilBERT models on a dataset of AI-generated stories for children.

requirements.txt: This file contains the necessary packages that Hugging Face needs to install for my space to run properly.

app.py: This file contains all of my code that is needed to run my space on Hugging Face.

**Instructions for running app.py:**

Example command: python3 app.py

Change the last line of the code to "screen.launch(share=True)". This will print one local and one public URL to access the application. Before running this file, make sure that the following packages are installed: transformers, torch, nltk, and datasets.

**Credit to AI:**

I utilized GitHub Copilot to assist me. I used this tool to help me adjust the training arguments in the files: fine_tuning_number_one.py and fine_tuning_number_two.py.
