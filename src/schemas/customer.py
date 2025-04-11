from pydantic import BaseModel

class CustomerData(BaseModel):
    gender: str
    seniorCitizen: int
    partner: int
    dependents: int
    tenure: float
    phoneService: int
    multipleLines: str
    internetService: str
    onlineSecurity: str
    onlineBackup: str
    deviceProtection: str
    techSupport: str
    streamingTV: str
    streamingMovies: str
    contract: str
    paperlessBilling: int
    paymentMethod: str
    monthlyCharges: float
    totalCharges: float

    class Config:
        allow_population_by_field_name = True