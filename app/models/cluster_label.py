from pydantic import BaseModel, Field, validator


class ClusterLabel(BaseModel):
    label: str = Field(description="Short label for the cluster")

    @validator("label")
    def label_must_be_less_than_10_words(cls, v):
        if len(v.split()) > 10:
            raise ValueError("Label must be less than 10 words")
        return v
