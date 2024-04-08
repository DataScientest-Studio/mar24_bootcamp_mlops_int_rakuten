from passlib.context import CryptContext


pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

def verify_password(plain_password, hashed_password):
    if plain_password +'fake_hash' == hashed_password:
        return True
    else:
        return False
#return pwd_context.verify(plain_password, hashed_password)


def get_password_hashed(password): #TODO: adding key?
    return password +'fake_hash'
    # return pwd_context.hash(password)

