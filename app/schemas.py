from typing import Literal
from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    school: Literal["GP", "MS"] = Field(..., description="Escuela (GP o MS)")
    sex: Literal["F", "M"] = Field(..., description="Sexo (F o M)")
    age: int = Field(..., ge=15, le=22, description="Edad entre 15 y 22 años")
    studytime: int = Field(..., ge=1, le=4, description="Tiempo de estudio semanal (1 a 4)")
    failures: int = Field(..., ge=0, le=4, description="Materias reprobadas previas (0 a 4)")
    famsup: Literal["yes", "no"] = Field(..., description="Apoyo educativo familiar")
    higher: Literal["yes", "no"] = Field(..., description="Desea cursar educación superior")
    internet: Literal["yes", "no"] = Field(..., description="Acceso a internet en casa")
    goout: int = Field(..., ge=1, le=5, description="Frecuencia de salidas con amigos (1 a 5)")
    absences: int = Field(..., ge=0, le=93, description="Número de faltas escolares")