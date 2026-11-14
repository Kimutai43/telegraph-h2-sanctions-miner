import re
import unicodedata
from dataclasses import dataclass
from typing import Optional
from rapidfuzz import fuzz

CORP_SUFFIXES = {
    "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation",
    "gmbh", "sa", "plc", "co", "company", "ag", "bv", "srl", "pte"
}

COUNTRY_ALIASES = {
    "usa": "united states",
    "us": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "great britain": "united kingdom",
    "uae": "united arab emirates",
    "russia": "russia",
    "russian": "russia",
    "russian federation": "russia",
    "syria": "syria",
    "syrian": "syria",
    "syrian arab republic": "syria",
    "iran": "iran",
    "iranian": "iran",
    "dprk": "north korea",
    "north korea": "north korea",
}


@dataclass
class EntityProfile:
    raw_name: str
    normalized_name: str
    dob: Optional[str] = None
    country: Optional[str] = None
    entity_type: str = "individual"  # "individual" or "organization"
    corp_suffix: Optional[str] = None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def extract_entity_profile(segment: str) -> EntityProfile:
    # Extract quoted name
    name_match = re.search(r"['\"]([^'\"]+)['\"]", segment)
    raw_name = name_match.group(1).strip() if name_match else segment.strip()

    # Detect corporate vs individual
    tokens = normalize_text(raw_name).split()
    found_suffixes = [t for t in tokens if t in CORP_SUFFIXES]
    is_org = bool(found_suffixes) or any(
        kw in normalize_text(raw_name)
        for kw in ["trading", "group", "holdings", "bank", "import", "export", "enterprise", "foundation"]
    )
    entity_type = "organization" if is_org else "individual"
    corp_suffix = found_suffixes[-1] if found_suffixes else None

    # Strip legal suffixes for core name comparison
    core_tokens = [t for t in tokens if t not in CORP_SUFFIXES]
    normalized_name = " ".join(core_tokens)

    # Extract DOB (YYYY-MM-DD)
    dob_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", segment)
    dob = dob_match.group(1) if dob_match else None

    # Extract Country / Nationality
    country = None
    lower_seg = normalize_text(segment)
    for alias, canonical in sorted(COUNTRY_ALIASES.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(alias)}\b", lower_seg):
            country = canonical
            break

    return EntityProfile(
        raw_name=raw_name,
        normalized_name=normalized_name,
        dob=dob,
        country=country,
        entity_type=entity_type,
        corp_suffix=corp_suffix,
    )


def names_compatible(a: EntityProfile, b: EntityProfile) -> tuple[float, bool]:
    """Returns (similarity_score_0_to_100, initial_expansion_match)."""
    tokens_a = a.normalized_name.split()
    tokens_b = b.normalized_name.split()

    token_set = fuzz.token_set_ratio(a.normalized_name, b.normalized_name)
    token_sort = fuzz.token_sort_ratio(a.normalized_name, b.normalized_name)
    base_score = max(token_set, token_sort)

    # Check middle initial expansion (e.g., "Vladimir A Petrov" vs "Vladimir Alexeyevich Petrov")
    initial_match = False
    if a.entity_type == "individual" and len(tokens_a) >= 2 and len(tokens_b) >= 2:
        if tokens_a[0] == tokens_b[0] and tokens_a[-1] == tokens_b[-1]:
            mid_a = tokens_a[1:-1]
            mid_b = tokens_b[1:-1]
            if mid_a and mid_b:
                if any(
                    (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y))
                    for x in mid_a for y in mid_b
                ):
                    initial_match = True
                    base_score = max(base_score, 96.0)

    return base_score, initial_match


def evaluate_sanctions_prompt(prompt: str) -> dict:
    # Split comparative screening prompt into target vs watchlist entry
    parts = re.split(r"\bagainst\b", prompt, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        target = extract_entity_profile(parts[0])
        sanctioned = extract_entity_profile(parts[1])
    else:
        target = extract_entity_profile(prompt)
        sanctioned = extract_entity_profile(prompt)

    name_score, initial_match = names_compatible(target, sanctioned)

    dob_match = (
        target.dob == sanctioned.dob
        if (target.dob and sanctioned.dob)
        else None
    )
    country_match = (
        target.country == sanctioned.country
        if (target.country and sanctioned.country)
        else None
    )
    suffix_differs = (
        target.corp_suffix != sanctioned.corp_suffix
        if (target.corp_suffix and sanctioned.corp_suffix)
        else False
    )

    # Decision Matrix
    # 1. Explicit demographic contradiction -> False Positive / Disambiguated
    if (dob_match is False) or (
        target.entity_type == "individual" and country_match is False and dob_match is not True
    ):
        reasons = []
        if dob_match is False:
            reasons.append("date of birth")
        if country_match is False:
            reasons.append("country of origin")
        diff_str = " and ".join(reasons) if reasons else "secondary demographic identifiers"
        verdict = "False Positive / Disambiguated"
        rationale = (
            f"{verdict}: Name shares surface similarity, but {diff_str} "
            f"conclusively exclude the target entity."
        )
        risk_level = "LOW"
        confidence = 0.96

    # 2. Strong name + confirming DOB/country -> Exact Match / High Risk
    elif name_score >= 90 and (dob_match is True or country_match is True) and not suffix_differs:
        aligned = ["name"]
        if dob_match is True:
            aligned.append("date of birth")
        if country_match is True:
            aligned.append("nationality/jurisdiction")
        aligned_str = ", ".join(aligned[:-1]) + f", and {aligned[-1]}" if len(aligned) > 1 else aligned[0]
        verdict = "Exact Match / High Risk"
        rationale = (
            f"{verdict}: {aligned_str.capitalize()} fully align with the sanctions list entry. "
            f"Flag for immediate transaction freeze and secondary compliance review."
        )
        risk_level = "HIGH"
        confidence = 0.98

    # 3. Partial / fuzzy name match with differing jurisdiction or entity structure -> Potential Match / Medium Risk
    elif name_score >= 65:
        diffs = []
        if country_match is False:
            diffs.append("jurisdiction")
        if suffix_differs:
            diffs.append("entity type")
        diff_clause = (
            f", but {' and '.join(diffs)} differ"
            if diffs
            else ", with incomplete secondary identifiers"
        )
        verdict = "Potential Match / Medium Risk"
        rationale = (
            f"{verdict}: Name similarity exists{diff_clause}. "
            f"Escalated for manual PEP and corporate registry verification prior to clearance."
        )
        risk_level = "MEDIUM"
        confidence = 0.84

    # 4. Low name similarity -> False Positive / Disambiguated
    else:
        verdict = "False Positive / Disambiguated"
        rationale = (
            f"{verdict}: Entity name and identifying attributes do not match the sanctions entry."
        )
        risk_level = "LOW"
        confidence = 0.95

    return {
        "verdict": verdict,
        "risk_level": risk_level,
        "confidence": confidence,
        "name_similarity": round(name_score / 100.0, 3),
        "analysis": rationale,
        "response": rationale,
    }