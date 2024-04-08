from fastapi import HTTPException
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.models import database as sql_m
from app.models import user as api_m
from app.core.security import get_password_hashed
from app.sql_db.database import SessionLocal

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user(db: Session, user_id: int):
    return db.query(sql_m.User).filter(sql_m.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(sql_m.User).filter(sql_m.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(sql_m.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: api_m.UserCreate):
    hashed_password = get_password_hashed(user.password)
    db_user = sql_m.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_is_admin(db: Session, user: api_m.User):
    user.is_admin = True
    db.commit()
    return user


def update_is_active(db: Session, user: api_m.User, new_state: bool):
    user.is_active= new_state
    db.commit()
    return user
# TODO:
# function set_inactive


