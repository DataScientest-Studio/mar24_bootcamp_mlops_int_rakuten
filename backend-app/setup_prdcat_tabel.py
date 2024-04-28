#!python
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError


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
        'postgresql://admin:admin@postgres:5432/rakuten_db'
)
with engine.connect() as connection:

    for entry in dic_code_to_category:
        label = entry[0]
        prdtypeid = entry[1]
        category = entry[2]
        
        query = text(f'insert into product_category(label, prdtypeid, category)\
                        values(:label, :prdtypeid, :category)'
                     )
        try:
            connection.execute(query, {'label': label, 'prdtypeid': prdtypeid, 'category': category})
            connection.commit()
        except IntegrityError as e:
            # check if the error code indicates a unique violation
            if e.orig.pgcode == '23505':  # postgresql error code for unique violation
                print(f"record already exists: {label}, {prdtypeid}, {category}")
            else:
                # handle other types of integrityerror if needed
                print("an integrityerror occurred:", e)
