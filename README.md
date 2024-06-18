HOW TO RUN THE TOOL

1. Go to the 'Final Visual.py' file and simply run the code in a Python environment.
2. The output of the code will be 'Dash is running on '[LINK]. Copy this link in your browser.
3. The tool is now completely set up and is ready to be used.


CODE FROM EXISTING LIBRARIES

The base of the Dash layout is from the lecture slides. We later modified it to our style with the help of dash bootstrap components. More information about the dash bootstrap components can be seen here: https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/.

The interactive plots were created with the help of the Plotly library, which is a library that supports interaction in plots and is easy to combine with Dash. Again the basic layout of these plots are used that are provided by the official documentation of Plotly itself, which we then later modified for our use.
The football pitch plot was created with the help of mplsoccer library, which is a library designed to plot a football pitch with two teams that have specific formations.
The full documentation of mplsoccer can be read here: https://mplsoccer.readthedocs.io/en/latest/index.html/

Callbacks to ensure interactions are created from scratch by ourselves.


DATA USED

The data we use comes from Kaggle.
We later made our own Parquet files, which are a modified version of some of the data that we got from Kaggle.

Kaggle data:
• FIFA World Cup 2022 Player Data (https://www.kaggle.com/datasets/swaptr/fifa-world-cup-2022-player-data)
• FIFA World Cup 2022 Match Data (https://www.kaggle.com/datasets/swaptr/fifa-world-cup-2022-match-data)
• FIFA World Cup 2022 Team Data (https://www.kaggle.com/datasets/swaptr/fifa-world-cup-2022-statistics)
• FIFA World Cup 2022 Twitter Dataset (https://www.kaggle.com/datasets/kumari2000/fifa-world-cup-twitter-dataset-2022)
• FIFA World Cup 2022 Prediction (https://www.kaggle.com/datasets/shilongzhuang/soccer-world-cup-challenge)
• FIFA World Cup 2022 Player Images (https://www.kaggle.com/datasets/soumendraprasad/fifa-2022-all-players-image-dataset)
• FIFA World Cup Historic (https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup)
• FIFA World Cup Penalty Shootouts (https://www.kaggle.com/datasets/pablollanderos33/world-cup-penalty-shootouts,
https://www.kaggle.com/datasets/jandimovski/world-cup-penalty-shootouts-2022
