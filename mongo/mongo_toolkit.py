from pymongo import MongoClient
import certifi
import os
from genson import SchemaBuilder
from bson.json_util import dumps
import json

client = MongoClient(os.environ['MONGODB'], tlsCAFile=certifi.where())


class MongoToolkit():
    def __init__(self, dbName):
        self.client = MongoClient(
            os.environ['MONGODB'], tlsCAFile=certifi.where())
        self.dbName = dbName

    def getSchema(self, collection, findOneQuery):
        coll = self.client[self.dbName][collection]
        document = coll.find_one(findOneQuery)
        jsonDoc = json.loads(dumps(document))
        builder = SchemaBuilder()
        builder.add_object(jsonDoc)
        schema = builder.to_schema()
        return json.dumps(schema, indent=2)
