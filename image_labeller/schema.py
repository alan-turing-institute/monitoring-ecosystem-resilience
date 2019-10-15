"""
Interface to the SQL database.
Use SQLAlchemy to convert between DB tables and python objects.
"""
import os
import time

from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, scoped_session
from sqlalchemy import create_engine, desc
from contextlib import contextmanager

from db_config import DB_CONNECTION_STRING


Base = declarative_base()

class User(Base):
    __tablename__ = "user"
    user_id = Column(Integer, primary_key=True, nullable=False)
    user_name = Column(String(100), nullable=False)


class Image(Base):
    __tablename__ = "image"
    image_id = Column(Integer, primary_key=True, nullable=False)
    image_filename = Column(String(100), nullable=False)


class Label(Base):
    __tablename__ = "label"
    label_id = Column(Integer, primary_key=True, nullable=False)
    label = Column(String(10), nullable=False)
    notes = Column(String(200), nullable=True)
    user = relationship("User", uselist=False)
    user_id = Column(Integer, ForeignKey('user.user_id'))
    image = relationship("Image", uselist=False)
    image_id = Column(Integer, ForeignKey('image.image_id'))

# wait a bit to allow postgres server to come up when running with docker-compose
time.sleep(5)
engine = create_engine(DB_CONNECTION_STRING)

Base.metadata.create_all(engine)
# Bind the engine to the metadata of the Base class so that the
# declaratives can be accessed through a DBSession instance
Base.metadata.bind = engine

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    db_session = sessionmaker(bind=engine, autoflush=False)
    session = db_session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def remove_db_session():
    scoped_session(sessionmaker(bind=engine)).remove()
