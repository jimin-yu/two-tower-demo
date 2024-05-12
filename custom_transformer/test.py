from db_config import *
import mysql.connector as sql
import pandas as pd
import time

db_connection = sql.connect(host=HOST, database=DATABASE_NAME, user=USERNAME, password=PASSWORD)
articles_fv = pd.read_sql('SELECT * FROM rec_articles', con=db_connection)
articles_features = articles_fv.columns.to_list()
print(articles_features)

time.sleep(1000)