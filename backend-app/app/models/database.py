from sqlalchemy import Boolean, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.sql_db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=None)

# WARNING: temporary 
class Product_Category(Base):
    __tablename__ = "product_category"

    id = Column(Integer, primary_key=True)
    label = Column(Integer, unique=True, index=True)
    prdtypeid = Column(Integer, index=True)
    category = Column(String)

class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc))
    user = Column(String)
    endpoint = Column(String)
    request = Column(String)
    response_code = Column(Integer)
    response_body = Column(String)

# class model_data(Base):
#     __tablename__ = "model_data"
#
#     id = Column(Integer, primary_key=True)
#     designation = Column(String)
#     description = Column(String)
#     img = Column()
#     label = Column(String)

# class text_feature(Base):
#     __tablename__ = "text_feature"
#
#     id = Column(Integer, primary_key=True)
#     designation = Column(String)
#     description = Column(String)
#     img_id = Column(Integer, foreign_key=True) #BUG: check
#
# class img_feature(Base):
#     __tablename__ = "img_feature"
#
#     id = Column(Integer, primary_key=True)
#     img = Column(Image) 
#     img_id = Column(Integer, foreign_key=True) #BUG: check
