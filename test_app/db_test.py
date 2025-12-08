import pymongo

if __name__ == "__main__":
    print("working!")
    client=pymongo.MongoClient("mongodb+srv://madhuripanchware711_db_user:807WtlFHEMgsQUt9@autokyc.ekzlr0w.mongodb.net/?appName=AutoKYC")
    db=client['test']
    collection=db['test_collection']
    collection.insert_one({"name":"madhuri"})