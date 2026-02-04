from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TYPE_CHECKING, Literal

from langchain_aws.vectorstores.valkey.constants import VALKEY_VECTOR_DTYPE_MAP

if TYPE_CHECKING:
    from valkey.commands.search.field import (  # type: ignore
        NumericField,
        TagField,
        VectorField,
    )


class ValkeyDistanceMetric(str, Enum):
    """Distance metrics for Valkey vector fields."""

    l2 = "L2"
    cosine = "COSINE"
    ip = "IP"


class ValkeyField(BaseModel):
    """Base class for Valkey fields."""

    name: str = Field(...)


class TagFieldSchema(ValkeyField):
    """Schema for tag fields in Valkey."""

    separator: str = ","
    case_sensitive: bool = False
    no_index: bool = False
    sortable: Optional[bool] = False

    def as_field(self) -> TagField:
        from valkey.commands.search.field import TagField  # type: ignore

        return TagField(
            self.name,
            separator=self.separator,
            case_sensitive=self.case_sensitive,
            sortable=self.sortable,
            no_index=self.no_index,
        )


class NumericFieldSchema(ValkeyField):
    """Schema for numeric fields in Valkey."""

    no_index: bool = False
    sortable: Optional[bool] = False

    def as_field(self) -> NumericField:
        from valkey.commands.search.field import NumericField  # type: ignore

        return NumericField(self.name, sortable=self.sortable, no_index=self.no_index)


class ValkeyVectorField(ValkeyField):
    """Base class for Valkey vector fields."""

    dims: int = Field(...)
    algorithm: object = Field(...)
    datatype: str = Field(default="FLOAT32")
    distance_metric: ValkeyDistanceMetric = Field(default="COSINE")  # type: ignore
    initial_cap: Optional[int] = None

    @field_validator("algorithm", "datatype", "distance_metric", mode="before")
    @classmethod
    def uppercase_strings(cls, v: str) -> str:
        return v.upper()

    @field_validator("datatype", mode="before")
    @classmethod
    def uppercase_and_check_dtype(cls, v: str) -> str:
        if v.upper() not in VALKEY_VECTOR_DTYPE_MAP:
            raise ValueError(
                f"datatype must be one of {VALKEY_VECTOR_DTYPE_MAP.keys()}. Got {v}"
            )
        return v.upper()

    def _fields(self) -> Dict[str, Any]:
        field_data = {
            "TYPE": self.datatype,
            "DIM": self.dims,
            "DISTANCE_METRIC": self.distance_metric,
        }
        if self.initial_cap is not None:
            field_data["INITIAL_CAP"] = self.initial_cap
        return field_data


class FlatVectorField(ValkeyVectorField):
    """Schema for flat vector fields in Valkey."""

    algorithm: Literal["FLAT"] = "FLAT"
    block_size: Optional[int] = None

    def as_field(self) -> VectorField:
        from valkey.commands.search.field import VectorField  # type: ignore

        field_data = super()._fields()
        if self.block_size is not None:
            field_data["BLOCK_SIZE"] = self.block_size
        return VectorField(self.name, self.algorithm, field_data)


class HNSWVectorField(ValkeyVectorField):
    """Schema for HNSW vector fields in Valkey."""

    algorithm: Literal["HNSW"] = "HNSW"
    m: int = Field(default=16)
    ef_construction: int = Field(default=200)
    ef_runtime: int = Field(default=10)
    epsilon: float = Field(default=0.01)

    def as_field(self) -> VectorField:
        from valkey.commands.search.field import VectorField  # type: ignore

        field_data = super()._fields()
        field_data.update(
            {
                "M": self.m,
                "EF_CONSTRUCTION": self.ef_construction,
                "EF_RUNTIME": self.ef_runtime,
                "EPSILON": self.epsilon,
            }
        )
        return VectorField(self.name, self.algorithm, field_data)


class ValkeyModel(BaseModel):
    """Schema for Valkey index."""

    tag: Optional[List[TagFieldSchema]] = None
    numeric: Optional[List[NumericFieldSchema]] = None
    extra: Optional[List[ValkeyField]] = None
    vector: Optional[List[Union[FlatVectorField, HNSWVectorField]]] = None
    content_key: str = "content"
    content_vector_key: str = "content_vector"

    def add_vector_field(self, vector_field: Dict[str, Any]) -> None:
        if self.vector is None:
            self.vector = []

        if vector_field["algorithm"] == "FLAT":
            self.vector.append(FlatVectorField(**vector_field))  # type: ignore
        elif vector_field["algorithm"] == "HNSW":
            self.vector.append(HNSWVectorField(**vector_field))  # type: ignore
        else:
            raise ValueError(
                f"algorithm must be either FLAT or HNSW. Got "
                f"{vector_field['algorithm']}"
            )

    def as_dict(self) -> Dict[str, List[Any]]:
        schemas: Dict[str, List[Any]] = {"tag": [], "numeric": []}
        for attr, attr_value in self.__dict__.items():
            if isinstance(attr_value, list) and len(attr_value) > 0:
                field_values: List[Dict[str, Any]] = []
                for val in attr_value:
                    value: Dict[str, Any] = {}
                    for field, field_value in val.__dict__.items():
                        if isinstance(field_value, Enum):
                            value[field] = field_value.value
                        elif field_value is not None:
                            value[field] = field_value
                    field_values.append(value)
                schemas[attr] = field_values

        schema: Dict[str, List[Any]] = {}
        for k, v in schemas.items():
            if len(v) > 0:
                schema[k] = v
        return schema

    @property
    def content_vector(self) -> Union[FlatVectorField, HNSWVectorField]:
        if not self.vector:
            raise ValueError("No vector fields found")
        for field in self.vector:
            if field.name == self.content_vector_key:
                return field
        raise ValueError("No content_vector field found")

    @property
    def vector_dtype(self) -> np.dtype:
        return VALKEY_VECTOR_DTYPE_MAP[self.content_vector.datatype]

    @property
    def is_empty(self) -> bool:
        return all(
            field is None for field in [self.tag, self.numeric, self.vector]
        )

    def get_fields(self) -> List["ValkeyField"]:
        valkey_fields: List["ValkeyField"] = []
        if self.is_empty:
            return valkey_fields

        for field_name in self.__fields__.keys():
            if field_name not in ["content_key", "content_vector_key", "extra"]:
                field_group = getattr(self, field_name)
                if field_group is not None:
                    for field in field_group:
                        valkey_fields.append(field.as_field())
        return valkey_fields

    @property
    def metadata_keys(self) -> List[str]:
        keys: List[str] = []
        if self.is_empty:
            return keys

        for field_name in self.__fields__.keys():
            field_group = getattr(self, field_name)
            if field_group is not None:
                for field in field_group:
                    if not isinstance(field, str) and field.name not in [
                        self.content_key,
                        self.content_vector_key,
                    ]:
                        keys.append(field.name)
        return keys


def read_schema(
    index_schema: Optional[Union[Dict[str, List[Any]], str, os.PathLike]],
) -> Dict[str, Any]:
    """Read in the index schema from a dict or yaml file."""
    if isinstance(index_schema, dict):
        return index_schema
    elif isinstance(index_schema, Path):
        with open(index_schema, "rb") as f:
            return yaml.safe_load(f)
    elif isinstance(index_schema, str):
        if Path(index_schema).resolve().is_file():
            with open(index_schema, "rb") as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"index_schema file {index_schema} does not exist")
    else:
        raise TypeError(
            f"index_schema must be a dict, or path to a yaml file "
            f"Got {type(index_schema)}"
        )
