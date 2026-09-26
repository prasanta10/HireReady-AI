from sqlmodel import SQLModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite+aiosqlite:///./database.db"

engine = create_async_engine(DATABASE_URL, echo=True)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


async def _sync_sqlite_columns(conn) -> None:
    """Add columns introduced after a database.db was first created.

    SQLModel's create_all only creates missing tables, so a cache file from an
    older version would otherwise fail with "no such column".
    """
    for table in SQLModel.metadata.sorted_tables:
        result = await conn.execute(text(f"PRAGMA table_info('{table.name}')"))
        existing = {row[1] for row in result.fetchall()}
        if not existing:
            continue
        for column in table.columns:
            if column.name in existing:
                continue
            column_type = column.type.compile(conn.dialect)
            await conn.execute(
                text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}')
            )
            print(f"DEBUG: Added missing column {table.name}.{column.name}")


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await _sync_sqlite_columns(conn)

async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
