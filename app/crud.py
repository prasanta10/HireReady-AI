from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import ProjectSummary

async def get_project(session: AsyncSession, user_id: str, repo_name: str):
    statement = select(ProjectSummary).where(
        ProjectSummary.user_id == user_id,
        ProjectSummary.repo_name == repo_name
    )
    result = await session.execute(statement)
    return result.scalar_one_or_none()

async def create_project(session: AsyncSession, project: ProjectSummary):
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return project