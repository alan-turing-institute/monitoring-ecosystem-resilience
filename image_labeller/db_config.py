"""
Database can be either an sqlite file or a postgress server
"""

import os

## Default connection string points to a local sqlite file in /tmp/data.db
DB_CONNECTION_STRING = "sqlite:////tmp/ildata.db"

## Check that we're not trying to set location for both sqlite and postgres
if "ILDBFile" in os.environ.keys() and \
   "ILDBUri" in os.environ.keys():
    raise RuntimeError("Please choose only ONE of ILDBFile and ILDBUri")

## location of sqlite file overridden by an env var
if "ILDBFile" in os.environ.keys():
    DB_CONNECTION_STRING = "sqlite:///{}".format(os.environ["ILDBFile"])

## location of postgres server
if "ILDBURI" in os.environ.keys():
    DB_CONNECTION_STRING = "postgres+psycopg2://il:il@{}/img-labeller".format(os.environ["ILDBURI"])
#    DB_CONNECTION_STRING = "postgres://{}/img-labeller?check_same_thread=False".format(os.environ["ILDBURI"])
