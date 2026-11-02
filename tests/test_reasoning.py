import os
import numpy as np
import pytest
from src.reasoning.vlm_auditor import AuditReport, VLMAuditor


def test_audit_report_pydantic_schema():
    valid_data = {
        "irregularidade_detectada": True,
        "nivel_risco": "ALTO",
        "descricao_visual": "Trabalhador em altura sem cinto de seguranca.",
        "acao_corretiva_sugerida": "Interromper servico e instalar linha de vida.",
        "confianca_analise": 0.98,
    }
    report = AuditReport(**valid_data)
    assert report.irregularidade_detectada is True
    assert report.nivel_risco == "ALTO"
    assert report.confianca_analise == 0.98


def test_vlm_auditor_execution(tmp_path):
    auditor = VLMAuditor()

    # Create dummy frame image
    image_path = tmp_path / "test_frame.jpg"
    with open(image_path, "wb") as f:
        f.write(b"fake_image_bytes")

    report_safe = auditor.audit_frame(str(image_path), context_query="inspecao de rotina")
    assert isinstance(report_safe, dict)
    assert "irregularidade_detectada" in report_safe
    assert "nivel_risco" in report_safe
    assert report_safe["irregularidade_detectada"] is False

    report_hazard = auditor.audit_frame(str(image_path), context_query="operador sem capacete")
    assert report_hazard["irregularidade_detectada"] is True
    assert report_hazard["nivel_risco"] == "ALTO"
