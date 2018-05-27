#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:19:08 2018

@author: RobinKraft
"""

import database_query

conn = sqlite3.connect("prices.db")
cur = conn.cursor()

database_query.upload_data('prices_db','prices',conn = conn)

database_query.make_query(statement = """SELECT * FROM prices WHERE date = '18.05.12'""",
                          conn = conn, cur = cur)