import pymongo
import pandas as pd
import json
from ins.config import mongo_client


DATA_FILE_PATH="/config/workspace/insurance.csv"
DATABASE_NAME ="ipp"
COLLECTION_NAME = "insurance"

if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

# Convert dataframe to JSON so that we can dump these records in MongoDb
    df.reset_index(drop=True,inplace=True)

    json_record = list((json.loads(df.T.to_json()).values()))
    print(json_record[0])


    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)






