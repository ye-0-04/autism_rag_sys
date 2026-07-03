import fitz
import os

def create_report():
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    
    y = 50
    # Header
    page.insert_text((50, y), "GENETIC NUTRIGENOMICS LABORATORY REPORT", fontsize=16)
    y += 40
    
    # Patient Info
    page.insert_text((50, y), "Patient Name: Alex Carter", fontsize=10)
    y += 15
    page.insert_text((50, y), "Date of Birth: 2018-05-12 (Age: 8)", fontsize=10)
    y += 15
    page.insert_text((50, y), "Patient ID: pat-101", fontsize=10)
    y += 15
    page.insert_text((50, y), "Physician: Dr. Sarah Vance, MD", fontsize=10)
    y += 40
    
    # Section
    page.insert_text((50, y), "GENETIC METABOLIC MARKERS FOUND:", fontsize=12)
    y += 25
    
    # Markers (format that OCR + parser expects)
    page.insert_text((50, y), "Gene: MTHFR  Variant: C677T  Status: Heterozygous", fontsize=10)
    y += 20
    page.insert_text((50, y), "Gene: COMT  Variant: V158M  Status: Homozygous Variant", fontsize=10)
    y += 20
    page.insert_text((50, y), "Gene: VDR  Variant: rs2228570  Status: Wild Type", fontsize=10)
    y += 40
    
    # Clinical Notes
    page.insert_text((50, y), "CLINICAL LABORATORY OBSERVATIONS:", fontsize=12)
    y += 20
    notes = [
        "Patient exhibits moderate folate pathway reduction (~30-40% efficiency due to MTHFR +/-).",
        "COMT homozygosity (+/+) indicates slow catecholamine degradation, predisposing patient",
        "to higher dopamine retention and sensitivity to external stimulants or methyl donors.",
        "VDR is wild-type, suggesting normal Vitamin D receptor binding efficiency."
    ]
    for note in notes:
        page.insert_text((50, y), note, fontsize=10)
        y += 15
        
    os.makedirs("autism_rag_sys/sample_data", exist_ok=True)
    output_path = "autism_rag_sys/sample_data/sample_report.pdf"
    doc.save(output_path)
    doc.close()
    print(f"Generated sample report PDF at: {output_path}")

if __name__ == "__main__":
    create_report()
