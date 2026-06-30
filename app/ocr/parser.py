import re
import json
import logging
from typing import List, Optional
from app.models.genetic_profile import GeneticMarker, GeneticProfile

logger = logging.getLogger(__name__)

GENE_PATTERN = re.compile(
    r"(?i)(MTHFR|COMT|VDR|APOE|FTO|TCF7L2|BCMO1|FADS1|FADS2|SOD2|CBS|MTR|MTRR|BHMT|"
    r"OPHN1|PAK3|RPS6KA3|IL1RAPL1?|FMR2|GDI1|CASK|TM4SF2|PTPRD|ADCY8|GRM3|BDNF|"
    r"GAD1|GRIN1|GRIN2[AB]|TBR1|NPAS4|CNTNAP2|TBX1)"
)

VARIANT_PATTERN = re.compile(
    r"(?i)(rs\d+|C677T|A1298C|V158M|TaqI|A66G|rs429358|rs7412|P199P|Q158R)"
)

STATUS_PATTERN = re.compile(
    r"(?i)(homozygous\s+(?:variant|risk|alternate|mutant)|heterozygous|wild[\s-]?type|"
    r"\+/\+|\+/-|-/-|normal|positive|negative|detected|not\s+detected)"
)

FLAG_PATTERN = re.compile(
    r"(?i)(HIGH RISK|MODERATE RISK|LOW RISK|FLAG|ABNORMAL|ATTENTION)"
)


class GeneticDataParser:
    def parse(self, raw_text: str, patient_id: str = "unknown") -> GeneticProfile:
        markers = self._regex_extract(raw_text)
        flagged = self._extract_flags(raw_text)
        raw_text_summary = self._clean_text(raw_text)

        if len(markers) == 0:
            logger.warning(
                "Regex found no markers — text may be poorly structured. "
                "Consider LLM-assisted extraction."
            )

        profile = GeneticProfile(
            patient_id=patient_id,
            markers=markers,
            flagged_values=flagged,
            raw_text_summary=raw_text_summary[:2000],
            extraction_confidence=self._score_confidence(markers, raw_text),
        )

        logger.info(f"Parsed {len(markers)} genetic markers for patient {patient_id}")
        return profile

    def _is_structured_line(self, line: str) -> bool:
        structured = bool(
            re.search(r"(?i)(Gene|Variant|Status|Result|Genotype)[\s:]", line)
        )
        return structured

    def _regex_extract(self, text: str) -> List[GeneticMarker]:
        lines = text.split("\n")
        markers = []
        seen_genes = set()

        for i, line in enumerate(lines):
            gene_match = GENE_PATTERN.search(line)
            if not gene_match:
                continue

            gene = gene_match.group(0).upper()
            if gene in seen_genes:
                continue
            seen_genes.add(gene)

            context = " ".join(lines[max(0, i - 1) : i + 3])
            variant_match = VARIANT_PATTERN.search(context)
            status_match = STATUS_PATTERN.search(context)

            variant = "UNKNOWN"
            status = "UNKNOWN"

            if self._is_structured_line(line):
                if variant_match:
                    variant = variant_match.group(0).upper()
                if status_match:
                    status = self._normalize_status(status_match.group(0))

            if status == "UNKNOWN" and not self._is_structured_line(line):
                logger.debug(f"Skipping variant/status for gene {gene} in running text")

            markers.append(
                GeneticMarker(
                    gene=gene,
                    variant=variant,
                    status=status,
                    raw_line=line.strip(),
                )
            )

        return markers

    def _normalize_status(self, raw_status: str) -> str:
        s = raw_status.lower().strip()
        if any(x in s for x in ["+/+", "homozygous variant", "homozygous risk"]):
            return "HOMOZYGOUS_VARIANT"
        if any(x in s for x in ["+/-", "heterozygous"]):
            return "HETEROZYGOUS"
        if any(x in s for x in ["-/-", "wild type", "wild-type", "normal"]):
            return "WILD_TYPE"
        return raw_status.upper()

    def _extract_flags(self, text: str) -> List[str]:
        return list(set(FLAG_PATTERN.findall(text)))

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"--- PAGE \d+ ---", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _score_confidence(self, markers: List[GeneticMarker], text: str) -> float:
        if len(markers) == 0:
            return 0.1
        known_markers = sum(1 for m in markers if m.status != "UNKNOWN")
        known_variants = sum(1 for m in markers if m.variant != "UNKNOWN")
        score = (known_markers + known_variants) / (len(markers) * 2)
        return round(min(score, 1.0), 2)
