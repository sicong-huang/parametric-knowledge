"""Shared Pydantic response models for LLM structured output."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class EntityNamesResponse(BaseModel):
    names: list[str] = Field(..., description="List of generated entity names")


class Relation(BaseModel):
    relation_type: str = Field(..., description="snake_case relation type")
    source_type: str = Field(..., description="source entity type")
    target_type: str = Field(..., description="target entity type")
    description: str = Field(..., description="human readable explanation")


class RelationOrNull(BaseModel):
    relations: list[Relation] | None = Field(
        None,
        description="List of relations, or null if no sensible relation exists between this pair",
    )


class ConstraintsResponse(BaseModel):
    min_per_source: Optional[int] = Field(
        None, description="minimum edges per source entity"
    )
    max_per_source: Optional[int] = Field(
        None, description="maximum edges per source entity"
    )
    min_per_target: Optional[int] = Field(
        None, description="minimum edges per target entity"
    )
    max_per_target: Optional[int] = Field(
        None, description="maximum edges per target entity"
    )
