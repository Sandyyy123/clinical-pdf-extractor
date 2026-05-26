from pydantic import BaseModel, Field
from typing import Optional, List

class ClinicalTrialData(BaseModel):
    title: str = Field(description="Full title of the clinical trial")
    phase: Optional[str] = Field(None, description="Trial phase (I/II/III/IV)")
    sample_size: Optional[int] = Field(None, description="Total enrolled participants")
    primary_endpoint: str = Field(description="Primary efficacy endpoint")
    primary_endpoint_result: Optional[str] = Field(None, description="Result of primary endpoint")
    p_value: Optional[str] = Field(None, description="Primary endpoint p-value")
    adverse_events_serious: Optional[int] = Field(None, description="Count of serious adverse events")
    study_population: Optional[str] = Field(None, description="Description of study population")
    intervention: str = Field(description="Study intervention or drug")
    comparator: Optional[str] = Field(None, description="Comparator or control arm")
    duration_weeks: Optional[int] = Field(None, description="Study duration in weeks")
    confidence_intervals: Optional[str] = Field(None, description="Primary endpoint CI")

class TableRow(BaseModel):
    row_index: int
    cells: List[str]

class ExtractedTable(BaseModel):
    table_id: str
    caption: Optional[str]
    headers: List[str]
    rows: List[TableRow]
    confidence: float = Field(description="Extraction confidence 0-1")

class ExtractionResult(BaseModel):
    document_id: str
    clinical_data: ClinicalTrialData
    tables: List[ExtractedTable]
    extraction_confidence: float
    warnings: List[str] = Field(default_factory=list)
