import logging
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
from sqlalchemy import create_engine
from app.models.database import Log  # Assuming your SQLAlchemy models are defined in models.py
from app.sql_db.database import SessionLocal
# Configure database connection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom handler to save logs to the database
class DatabaseHandler(logging.Handler):
    def emit(self, record):
        message = self.format(record)
        with SessionLocal() as session:
            log_entry = Log(user=record.user,
                            endpoint=record.endpoint,
                            request=record.request,
                            response_code=record.response_code)#,
                            # response_body=record.response_body)
            session.add(log_entry)
            session.commit()

logger.addHandler(DatabaseHandler())

async def log_requests(request, request_body, response, response_body):
    log_data = {
        "user": request.client.host,  # Example: request.client.host
        "endpoint": request.url.path,
        "request": request_body,#.decode("utf-8"),
        "response_code": response.status_code#,
         # "response_body": response_body[0].decode() 
    }
    logger.info("message", extra=log_data)
