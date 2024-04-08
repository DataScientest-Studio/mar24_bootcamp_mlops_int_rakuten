import pytest
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from app.sql_db import crud
from app.models.database import User as db_user
from app.sql_db.database import Base
from app.models.user import UserCreate, User

@pytest.fixture(scope="session")
def test_db():
    # Create an in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")

    # Create tables in the database
    Base.metadata.create_all(engine)

    # Create a sessionmaker bound to the engine
    Session = sessionmaker(bind=engine)

    session = Session()

    session.add(db_user(id=1, email="user1@example.com", hashed_password="testfake_hash", is_active=True, is_admin=True))
    session.add(db_user(id=2, email="user2@example.com", hashed_password="test2fake_hash", is_active=True, is_admin=False))
    session.add(db_user(id=3, email="user3@example.com", hashed_password="test3fake_hash", is_active=False, is_admin=False))

    session.commit()
    # Return a session to the test database
    yield Session()

@pytest.fixture(scope="function")
def test_user():
    return UserCreate(email="test@example.com", password="testpassword")

@pytest.fixture(scope="function")
def test_user_exists():
    return UserCreate(email="user1@example.com", password="testpassword")

# Test get_user
def test_get_user_existing(test_db):

    user = crud.get_user(test_db, user_id=1)
    
    assert user is not None
    assert user.id == 1
    assert user.email == "user1@example.com"
    assert user.hashed_password == "testfake_hash"
    assert user.is_active is True
    assert user.is_admin is True

def test_get_user_missing(test_db):

    missing_user = crud.get_user(test_db, user_id=4)

    assert missing_user is None


# Test get_user_by_mail 
def test_get_user_by_email_existing(test_db):

    user = crud.get_user_by_email(test_db, email='user1@example.com')

    assert user is not None
    assert user.id == 1
    assert user.email == "user1@example.com"
    assert user.hashed_password == "testfake_hash"
    assert user.is_active is True
    assert user.is_admin is True


def test_get_user_by_email_missing(test_db):

    missing_user = crud.get_user(test_db, user_id=4)

    assert missing_user is None


# Test get_users
def test_get_users_default(test_db):

    user_list = crud.get_users(test_db)

    assert len(user_list) == 3
    assert user_list[0].id == 1
    assert user_list[1].id == 2
    assert user_list[2].id == 3

def test_get_users_skip_two(test_db):

    user_list = crud.get_users(test_db, skip=2)

    assert len(user_list) == 1
    assert user_list[0].id == 3


def test_get_users_limit_two(test_db):

    user_list = crud.get_users(test_db, limit=2)

    assert len(user_list) == 2
    assert user_list[0].id == 1
    assert user_list[1].id == 2


def test_get_users_skip_one_limit_two(test_db):

    user_list = crud.get_users(test_db, skip=1,  limit=2)

    assert len(user_list) == 2
    assert user_list[0].id == 2
    assert user_list[1].id == 3

def test_get_users_skip_two_limit_two(test_db):

    user_list = crud.get_users(test_db, skip=2,  limit=2)

    assert len(user_list) == 1
    assert user_list[0].id == 3


def test_get_users_none(test_db):

    user_list = crud.get_users(test_db, skip=3, limit=2)

    assert len(user_list) == 0

def test_create_user_ok(test_db, test_user):
    created_user = crud.create_user(test_db, test_user)

    assert test_db.query(db_user).filter(db_user.email == test_user.email).first() is not None

    assert created_user.email == test_user.email


def test_create_user_exists(test_db, test_user_exists):

    with pytest.raises(IntegrityError):
        crud.create_user(test_db, test_user_exists)

# def test_update_is_admin(test_db, test_user):
#     user = test_db.query(db_user).filter(db_user.email==test_user.email).first()
#     # updated_user = crud.update_is_admin(test_db, test_user)
#     #
#     # retrieved_user = test_db.query(db_user).filter_by(id=2).first()
#
#     assert user.is_admin == True
#     # assert updated_user.is_admin == True
#     # assert retrieved_user.isadmin == True
