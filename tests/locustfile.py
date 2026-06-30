from locust import HttpUser, task, between
import io
import fitz


def make_pdf_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "MTHFR C677T Heterozygous")
    page.insert_text((50, 140), "COMT V158M Homozygous Variant")
    pdf = doc.tobytes()
    doc.close()
    return pdf


class RAGAPIUser(HttpUser):
    wait_time = between(5, 15)
    host = "http://localhost:8000"

    def on_start(self):
        self.pdf_bytes = make_pdf_bytes()

    @task
    def generate_plan(self):
        self.client.post(
            "/generate-nutrition-plan",
            headers={"X-API-Key": "your-secret-api-key-change-this"},
            files={
                "file": ("report.pdf", io.BytesIO(self.pdf_bytes), "application/pdf")
            },
            data={"patient_id": "load-test-patient"},
            timeout=120,
        )
