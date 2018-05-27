#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 12:33:23 2018

@author: RobinKraft
"""

import pandas as pd
import sqlite3


if __name__ == '__main__':
    conn = sqlite3.connect("prices.db")
    cur = conn.cursor()
    
       
    cur.execute("""CREATE TABLE if not exists prices 
                (date TEXT, 
                 AAPL FLOAT,
                 Tesla FLOAT,
                 KU2 FLOAT,
                 Facebook FLOAT
                 )""")

    conn.commit()
    conn.close()


def upload_data(file_name, database, conn):
    with conn:    
        df = pd.read_csv(file_name + ".csv", delimiter = ";")
        df.to_sql(database, conn, if_exists='append', index=False)

def make_query(statement, conn, cur):
    with conn:
        print(cur.execute(statement).fetchall())


def get_price_data(tick_list, database, conn):
    query = "SELECT "

    for list_memb in tick_list:
        if list_memb != tick_list[-1]:
            query = query + list_memb + ", "
        else:
            query = query +list_memb
    
    sql_query = query + " FROM " + database

    data = pd.read_sql_query(sql_query, conn)
    return data
