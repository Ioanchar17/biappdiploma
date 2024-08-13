import pandas as pd

iphone_official_reviews = pd.read_csv('Official_datasets/Apple_Iphone_11_Reviews_new.CSV', sep=';')

airpods_official_reviews = pd.read_csv('Official_datasets/Apple_AirPods_2nd_Gen_Reviews.csv')

macbook_official_reviews = pd.read_csv('Official_datasets/Apple_Macbook_Air_M1_final.csv')

iphone_official_reviews['rating'] = iphone_official_reviews['rating'].div(10)