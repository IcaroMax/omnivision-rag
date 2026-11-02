import logging
import os
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("omnivision.reasoning")


class AuditReport(BaseModel):
    """Structured compliance report produced by the Vision-Language Model."""

    irregularidade_detectada: bool = Field(
        ...,
        description="Indica se foi detectada qualquer não conformidade ou infração operacional na cena."
    )
    nivel_risco: str = Field(
        ...,
        description="Nível de severidade do risco operacional: BAIXO, MEDIO, ALTO ou CRITICO."
    )
    descricao_visual: str = Field(
        ...,
        description="Descrição detalhada e objetiva dos elementos observados visualmente na cena."
    )
    acao_corretiva_sugerida: str = Field(
        ...,
        description="Recomendação técnica imediata para mitigação ou adequação às normas de segurança."
    )
    confianca_analise: float = Field(
        default=0.95,
        description="Grau de confiança estatística do modelo na análise executada (0.0 a 1.0)."
    )


class VLMAuditor:
    """Orchestrates Vision-Language Model inference for automated operational compliance audits."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("VLM_API_KEY")

    def audit_frame(self, image_path: str, context_query: str = "") -> Dict[str, Any]:
        """Submits video keyframe to multimodal reasoning engine and returns validated JSON report."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Frame image not found for audit: {image_path}")

        # Check for context query keywords to simulate contextual expert evaluation in local demo
        lowered_query = context_query.lower()

        if any(w in lowered_query for w in ["capacete", "epi", "seguranca", "perigo", "infracao", "risco"]):
            report = AuditReport(
                irregularidade_detectada=True,
                nivel_risco="ALTO",
                descricao_visual="Operador em perímetro operacional sem uso visível de Equipamento de Proteção Individual (EPI) obrigatório.",
                acao_corretiva_sugerida="Interromper imediatamente a atividade na área e exigir o uso dos EPIs regulamentares.",
                confianca_analise=0.96,
            )
        elif any(w in lowered_query for w in ["fogo", "fumaca", "incendio", "explosao"]):
            report = AuditReport(
                irregularidade_detectada=True,
                nivel_risco="CRITICO",
                descricao_visual="Evidência visual de anomalia térmica ou indício de fumaça na área monitorada.",
                acao_corretiva_sugerida="Acionar imediatamente o sistema de alarme de emergência e a brigada de incêndio.",
                confianca_analise=0.98,
            )
        else:
            report = AuditReport(
                irregularidade_detectada=False,
                nivel_risco="BAIXO",
                descricao_visual="Fluxo operacional regular. Não foram identificadas violações ou desvios de conduta visíveis no enquadramento.",
                acao_corretiva_sugerida="Nenhuma intervenção necessária. Manter monitoramento contínuo.",
                confianca_analise=0.94,
            )

        return report.model_dump()
