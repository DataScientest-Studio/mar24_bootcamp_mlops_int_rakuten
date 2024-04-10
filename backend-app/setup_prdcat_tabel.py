#!python
from app.sql_db.crud import add_product_category 
from sqlalchemy import create_engine


dic_code_to_category = [
    [0,   10 , "Second-hand Books"],
    [1,   2280 , "Second-hand Newspapers and Magazines"],
    [2,   2403 , "Books, Comic Books, Magazines"],
    [3,   2522 , "Stationery Supplies and Office Accessories"],
    [4,   2705 , "New Books"],
    [5,   40 , "Video Games, CDs, Equipment, Cables"],
    [6,   50 , "Gaming Accessories"],
    [7,   60 , "Games Consoles"],
    [8,   2462 , "Second-hand Video Games"],
    [9,   2905 , "PC Video Games"],
    [10,  1140 , "Figurines, Pop Culture Objects"],
    [11,  1160 , "Trading Games Cards"],
    [12,  1180 , "Figurines and Role-playing Games"],
    [13,  1280 , "Children's Toys"],
    [14,  1281 , "Board Games for Children"],
    [15,  1300 , "Model-making"],
    [16,  1302 , "Outdoor Games, Clothing"],
    [17,  1560 , "General Furniture: Furniture, Matelas, Sofas, Lamps, Chairs"],
    [18,  2582 , "Garden Furniture: Furniture and Tools for Garden"],
    [19,  1320 , "Childcare, Baby Accessories"],
    [20,  2220 , "Pet Shop"],
    [21,  2583 , "Swimming Pool and Accessories"],
    [22,  2585 , "Garden Tools, Outdoor Technical Equipment for Homes, Swimming Pools"],
    [23,  1920 , "Household Linen, Pillows, Cushions"],
    [24,  2060 , "Decoration"],
    [25,  1301 , "Baby Socks, Smalls Pictures"],
    [26,  1940 , "Confectionery"]
]
engine = create_engine(
    'sqlite:///sql_app.db', connect_args={"check_same_thread": False} # WARNING:'check_same_thread only sqllite remove late
)
with engine.connect() as connection:
    # Execute the SQL statement directly
    result = connection.execute("SELECT * FROM users")

    # Fetch and print the results
    for row in result:
        print(row)
