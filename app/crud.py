from datetime import datetime
from typing import Optional

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


async def mark_indexed(session: AsyncSession, project: ProjectSummary, content_hash: str):
    """Record that this repo's current content is present in the vector index."""
    project.content_hash = content_hash
    project.indexed_at = datetime.utcnow()
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return project


async def upsert_project(
    session: AsyncSession,
    user_id: str,
    repo_name: str,
    summary: str,
    url: str,
    language: Optional[str] = None,
    readme: Optional[str] = None,
) -> ProjectSummary:
    """Insert the repo, or refresh an existing cached row in place."""
    existing = await get_project(session, user_id, repo_name)
    if existing:
        existing.summary = summary
        existing.url = url
        existing.language = language
        existing.readme = readme
        existing.updated_at = datetime.utcnow()
        session.add(existing)
        await session.commit()
        await session.refresh(existing)
        return existing

    project = ProjectSummary(
        user_id=user_id,
        repo_name=repo_name,
        summary=summary,
        url=url,
        language=language,
        readme=readme,
    )
    return await create_project(session, project)
