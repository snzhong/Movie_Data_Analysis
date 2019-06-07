# Movie Data Analysis
## by Sean Zhong


## Dataset

> This dataset was provided to me by Udacity as part of their Data Analyst Nanodegree program. It contains movie data from The Movie Database (TMDb) via Kaggle (https://www.kaggle.com/tmdb/tmdb-movie-metadata) and came to me mostly cleaned by Udacity. The dataset contains information about 10,000 movies collected from TMDb, including user ratings and revenue. Additionally, below are a few notes about the data.

- Certain columns, like ‘cast’ and ‘genres’, contain multiple values separated by pipe (|) characters.
- The final two columns ending with “_adj” show the budget and revenue of the associated movie in terms of 2010 dollars, accounting for inflation over time.


## Summary of Findings

> I've gathered below a summary of my findings:

1) It appears that **'Adventure'**, **'Animation'**, and **'Family'** movies generate the most revenue on average. However, due to data quality issues, I had to remove approximately half of the available data (**movies with zero revenue**). This, combined with the finding that some movie genres were much more likely to report zero revenue (most notable the **'Foreign'** and **'TV Movie'** genres), the results could be skewed. Additionally, I worked under the assumption that the main genre would be the most important. This has certain limitations as the other genre labels in tandem could potentially have more of an effect on the revenue than I realize.

2) Movies that are better rated also tend to have higher revenues on average. It should be noted however that there is only a weak to moderate correlation, as well as the fact that I had to drop approximately half of the available data due to the same data quality issue stated above.

3) Movies that have longer runtimes also tend to have better ratings on average. It should be noted however that there is only a weak to moderate correlation. However, unlike the other two findings above, most of the data points were able to be used.

