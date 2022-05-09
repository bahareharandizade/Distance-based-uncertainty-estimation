# Distance-based-uncertainty-estimation
This is the starter code for homework 3 part of the code is 
based on this github https://github.com/google/TrustScore.git.


## Setup

- See `requirements.txt` for required packages.
- Tested with python3.8

## How to use?
- **Embdding models**: Use `sentiment_embed.py` for getting Bert Sentiment embedding and use `fine_tune_embed.py` to get the embdding from morality dataset. As we cannot share the dataset publicly(Twitter law) we can not put the tweets here. 
- **Confidence and uncertainty Scores**: Use `main_score.py` to get all the used socres in the report including: Mahalonobis and Euclidean confidence score, Trust Score, Nearest Neighbor Score, Entropy and annotators scores.      
