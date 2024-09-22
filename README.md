# Sentiment-Analysis-and-Topic-Modelling-based-on-News-Headlines-and-Articles
 
## Background
In this digital era, the amount of News arrivals is increasing day by day. This is because News can appear in people's view through many platforms such as news websites and social media. News needed to take a lot of time to read as each News contained much information and this information is crucial because it can help people to identify current trends. However, each category of News has its own trends and people need to spend time reading it to identify the News whether is their preference category to understand trends. Spending time to read news can cause people to hesitate in making decisions and thereby lose profits. To facilitate people to quickly identify their preference category of News to identify trends, this project introduce a **Sentiment Analysis and Topic Modelling to Detect Trends Based on News Headlines and News Articles**. 

## Scopes
### Sentiment Analysis on News Headlines
  - Classify news headlines sentiment as positive, negative, or neutral
### Topic Modelling on News Articles
  - Extract topics from news articles to identify trends

## Methodology / Folders Breakdown
![test drawio (1)](https://github.com/user-attachments/assets/f41a375c-cc6f-4d12-be5a-78dbd42bf4c5)

## Link of Datasets (Kaggle)
- CNN news (2011-2022)
https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning/data
- CNN news (2023)
https://www.kaggle.com/datasets/pedroaribe/4000-cnn-articles-as-of-1062023

## Result & Discussion
### Sentiment Analysis Models (News Headlines)
- The result shows that **SVM** and **LR** models achieved the highest accuracy, indicating they performed the best in sentiment analysis on news headlines. Among these, the LR model has been selected for deployment
![image](https://github.com/user-attachments/assets/c6edd5f0-219a-4456-82a4-a14431326490)
### Topic Modelling Models (News Articles)
- The result shows that BERTopic model achieved the highest C_V and NPMI scores, indicating that it generates more semantically coherent topics and has stronger word association within topics
![image](https://github.com/user-attachments/assets/c8afc5ae-2d1f-4815-ae4b-213ee8f20805)

## Deployment
![image](https://github.com/user-attachments/assets/a18cfb15-d884-43a9-8560-6964d0981b12)
![image](https://github.com/user-attachments/assets/a7b3eeb7-889f-49b0-887d-d5aa5a8c5c28)
![image](https://github.com/user-attachments/assets/925d53e9-6a86-4758-9e4e-89dae7976d83)

## Future Improvement
- **Quality of Dataset**
  - A large dataset with over 100,000 rows ensures robust analysis.
  - A balanced dataset with diverse news categories is essential for unbiased results.
  - A labeled dataset with sentiment annotations enables supervised sentiment analysis.
- **Other Models**
  - Incorporate deep learning, ensemble methods, and pretrained models for sentiment analysis to improve performance.
  - Utilize pretrained models for topic modeling to enhance topic extraction accuracy.
- **Enhanced Computer Resources**
  - Deep learning and pretrained models require significant computational power for training and fine-tuning.

 ## Setup 
- git clone this repository
- Go through the requirement.txt (Note that the **numpy** must == 1.26.4)
- Download the datasets from kaggle
- Once everything is complete, feel free to proceed with execution




