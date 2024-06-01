 # Data Science Term Project
 
 [![code](https://img.shields.io/badge/Code-Python3.7-blue)](https://docs.python.org/3/license.html)
 [![data](https://img.shields.io/badge/Data-Kaggle-blueviolet)](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings/data)


## Object Setting
1. Video Game Success Prediction
   - To predict the success of video games based on sales volumes, genre, and platform
     - using Classification algorithm
     - using Clustering the video game market by genre, platform

Goal: assist game developers and publishers in formulating development and marketing strategies

2. Genre-Based Market Segmentation of Video Games
   - To analyze the characteristic of each segmented group
     - using segment(cluster) the video game market bt genre

Goal: Help game developers and marketers to formulate more effective targeting strategies


## Data 
### Video Game Sale with Rating (Video game sales from Vgchartz and corresponding ratings from Metacritic)
<img width="1307" alt="image" src="https://github.com/Jeonseungwoo1/DataScience/assets/149984505/f0048d3b-e81e-4099-b2c3-2ba313c1683a">

Numerical: Year_of_Release, Sales, Global_Sales, Critic_Score, User_Score, Rating

Categorical: Platform, Genre, Publisher, Developer

16 Columns, 16719 Rows

## Project Result
### 1. Classification


   Decision Tree
   <img width="1307" alt="image" src="https://github.com/Jeonseungwoo1/DataScience/assets/149984505/2b303c17-dd68-428f-8310-a604bc702c9b">

   
   K-Nearest Neighbor
   <img width="1307" alt="image" src="https://github.com/Jeonseungwoo1/DataScience/assets/149984505/50975dab-7b86-48c9-b8c9-39f01ff9f92b">
   

   Random Forest
   <img width="1307" alt="image" src="https://github.com/Jeonseungwoo1/DataScience/assets/149984505/cc7164c9-b690-4016-ac63-40c7ee40f58a">


### Classification Model Evaluation

   <img width="502" alt="image" src="https://github.com/Jeonseungwoo1/DataScience/assets/149984505/68195bc2-a8e9-4ae9-9abb-c560ad4da803">


### 2. Clustering

  K-Means Clustering
  <img width="1307" alt="image" src="https://github.com/Jeonseungwoo1/DataScience/assets/149984505/042a5d1e-d44b-4f48-99eb-bba195dd8005">

  
  Elbow Method (Best K is 3~5)
  
  <img width="630" alt="image" src="https://github.com/Jeonseungwoo1/DataScience/assets/149984505/c87d59a2-c930-4ab3-b333-0ba4a22136ee">



  Silhouette Plot
  
  <img width="320" alt="image" src="https://github.com/Jeonseungwoo1/DataScience/assets/149984505/5795b182-1d8c-4a85-ae20-8a2aeffd1d48">
  <img width="320" alt="image" src="https://github.com/Jeonseungwoo1/DataScience/assets/149984505/71751866-c30b-44d0-8162-d8584e45643c">
  <img width="318" alt="image" src="https://github.com/Jeonseungwoo1/DataScience/assets/149984505/5857dc24-4ba9-41c0-938b-fa046b9ccd16">
  <img width="318" alt="image" src="https://github.com/Jeonseungwoo1/DataScience/assets/149984505/3c047fa9-0da1-4bd3-a6ab-fd8d0cbfc96f">


### Clustering Result
  <img width="633" alt="image" src="https://github.com/Jeonseungwoo1/DataScience/assets/149984505/b18145b9-7100-4b7a-a3f9-9734b108d0e3">

  Cluster 3 is the largest in Shooting
  
  Cluster 2 is the largest in Action

  








