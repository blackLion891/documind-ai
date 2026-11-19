
from dataclasses import dataclass


@dataclass
class DocumentPage:
    page_number: int
    text: str


@dataclass
class ExtractedDocument:
    file_name: str
    file_type: str
    file_path: str
    pages: list[DocumentPage]

    @property
    def full_text(self) -> str:
        return "\n".join(
            page.text for page in self.pages
        )
