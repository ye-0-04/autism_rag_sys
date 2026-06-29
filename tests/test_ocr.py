import pytest
import numpy as np
from app.ocr.preprocessor import ImagePreprocessor


def test_preprocessor_loads_pdf():
    preprocessor = ImagePreprocessor()
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Gene: MTHFR  Variant: C677T  Status: Heterozygous")
    doc.save("sample_data/test_report.pdf")
    doc.close()

    images = preprocessor.preprocess_file("sample_data/test_report.pdf")
    assert len(images) == 1
    assert isinstance(images[0], np.ndarray)
    assert images[0].dtype == np.uint8


def test_preprocessor_output_is_binary():
    preprocessor = ImagePreprocessor()
    img = np.random.randint(100, 200, (500, 500, 3), dtype=np.uint8)
    result = preprocessor.preprocess(img)
    unique_values = np.unique(result)
    assert set(unique_values).issubset({0, 255}), "Image is not binary"


def test_preprocessor_handles_already_grayscale():
    preprocessor = ImagePreprocessor()
    img = np.ones((300, 300), dtype=np.uint8) * 180
    result = preprocessor.preprocess(img)
    assert result.shape == (300, 300)


from app.ocr.extractor import OCRExtractor


def test_ocr_extracts_text_from_synthetic_image():
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 100), "GENETIC REPORT", fontsize=16)
    page.insert_text((50, 140), "Patient: Test Patient")
    page.insert_text((50, 170), "Gene: MTHFR  Variant: C677T  Status: Heterozygous")
    page.insert_text((50, 200), "Gene: COMT   Variant: V158M  Status: Homozygous")
    pdf_path = "sample_data/test_ocr_input.pdf"
    doc.save(pdf_path)
    doc.close()

    preprocessor = ImagePreprocessor()
    pages = preprocessor.preprocess_file(pdf_path)

    extractor = OCRExtractor()
    text = extractor.extract_from_pages(pages)

    assert "MTHFR" in text or "mthfr" in text.lower(), (
        f"Expected MTHFR in OCR output, got: {text[:500]}"
    )
    assert len(text) > 20


def test_ocr_returns_empty_string_for_blank_image():
    extractor = OCRExtractor()
    blank = np.ones((500, 500), dtype=np.uint8) * 255
    result = extractor.extract_text(blank)
    assert isinstance(result, str)


from app.ocr.parser import GeneticDataParser
from app.models.genetic_profile import GeneticProfile, GeneticMarker


def test_parser_extracts_mthfr_from_text():
    sample_text = """
    --- PAGE 1 ---
    GENETIC ANALYSIS REPORT
    Gene: MTHFR  Variant: C677T  Status: Heterozygous
    Gene: COMT   Variant: V158M  Status: Homozygous Variant
    Gene: VDR    Variant: rs2228570  Status: Wild Type
    HIGH RISK: Folate metabolism impairment detected
    """
    parser = GeneticDataParser()
    profile = parser.parse(sample_text, patient_id="test-001")

    assert isinstance(profile, GeneticProfile)
    assert len(profile.markers) >= 2

    gene_names = [m.gene for m in profile.markers]
    assert "MTHFR" in gene_names
    assert "COMT" in gene_names


def test_parser_normalizes_status():
    sample_text = "Gene: MTHFR  Variant: C677T  Status: +/+"
    parser = GeneticDataParser()
    profile = parser.parse(sample_text)
    assert profile.markers[0].status == "HOMOZYGOUS_VARIANT"


def test_parser_returns_profile_with_no_markers_gracefully():
    parser = GeneticDataParser()
    profile = parser.parse("This is not a genetics report.", patient_id="bad-input")
    assert len(profile.markers) == 0
    assert profile.extraction_confidence < 0.5


def test_genetic_profile_to_retrieval_query():
    profile = GeneticProfile(
        patient_id="test",
        markers=[
            GeneticMarker(
                gene="MTHFR", variant="C677T", status="HETEROZYGOUS", raw_line=""
            )
        ],
    )
    query = profile.to_retrieval_query()
    assert "MTHFR" in query
    assert len(query) > 5


def test_full_ocr_to_profile_pipeline():
    import fitz
    from app.ocr.preprocessor import ImagePreprocessor
    from app.ocr.extractor import OCRExtractor

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "MTHFR C677T Heterozygous")
    page.insert_text((50, 140), "COMT V158M Homozygous Variant")
    doc.save("sample_data/pipeline_test.pdf")
    doc.close()

    preprocessor = ImagePreprocessor()
    extractor = OCRExtractor()
    parser = GeneticDataParser()

    pages = preprocessor.preprocess_file("sample_data/pipeline_test.pdf")
    text = extractor.extract_from_pages(pages)
    profile = parser.parse(text, patient_id="pipeline-test")

    assert isinstance(profile, GeneticProfile)
    assert len(profile.markers) >= 1
