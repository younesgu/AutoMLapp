"""Backward-compatible import for older integrations.

Use ``automl_app.services.document_qa.answer_document`` in new code.
"""

from automl_app.services.document_qa import answer_document


def qadocument(file_path: str, query: str) -> str:
    """Answer a question about a PDF document."""
    return answer_document(file_path, query)
