"""Static token maps for normalisation (plan group B, 05 §3): hand-typed domain knowledge.

Legal forms, honorifics, street-type abbreviations, ordinal words, old/new Indian city
names and every written form of a region (full name, postal code, the ``anyascii`` form of
its native-script name). No gazetteer or external data: the Indic state spellings are the
transliterations of the names as they appear in the training addresses (checked against
train-fold pairs), the rest is general knowledge. Learned maps live in ``normalize``
(``fit_token_map``).
"""
from __future__ import annotations

# Legal forms (names only): token -> canonical form. Removed from name_core, kept in
# legal_form. Multi-letter forms written with dots ("L.L.C.", "P.C.") are joined into one
# token first (_join_initials), so "l l c" never reaches this map.
LEGAL_FORMS = {
    "inc": "inc", "incorporated": "inc", "incorporation": "inc",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "cie": "co",
    "llc": "llc", "lc": "llc", "llp": "llp", "lp": "lp", "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt", "pte": "pte", "plc": "plc", "pc": "pc", "pllc": "pllc",
    "pa": "pa", "pbc": "pbc", "opc": "opc", "gmbh": "gmbh",
    "sa": "sa", "sas": "sas", "sasu": "sas", "sarl": "sarl", "eurl": "sarl", "sci": "sci",
    "snc": "snc", "ei": "ei", "eirl": "ei", "selarl": "selarl", "scp": "scp",
}
# anyascii output of Indic-script legal words (प्राइवेट लिमिटेड -> "praivet limited",
# प्रा. लि. -> "pra li", एलएलपी -> "elelpi"), applied to transliterated names only so a
# Latin "Li" (a surname) is never dropped.
TRANSLIT_LEGAL = {
    "praivet": "pvt", "praivett": "pvt", "prayvet": "pvt", "piraivet": "pvt",
    "praibhet": "pvt", "praivrr": "pvt", "pra": "pvt",
    "limitet": "ltd", "limirrd": "ltd", "limitedd": "ltd", "limittedd": "ltd", "li": "ltd",
    "elelpi": "llp", "imk": "inc", "kampni": "co",
}
# Leading honorifics injected into pool names ("Mr Vistaar Traders", "Smt ...", "M/s ...").
HONORIFIC_RE = r"^(?:(?:mr|mrs|ms|dr|smt|shri|sri|messrs|the)\s+)+"
# Leet digits folded to letters inside tokens that also hold letters ("f0rman", "5tar").
LEET = str.maketrans("0134578", "oleastb")

# Address tokens -> canonical short form. Both spellings of a pair map to the same token
# (the direction does not matter). The generator writes "Saint" for "St" (Government
# Street -> Government Saint), so street, st and saint share one token.
ADDRESS_TOKENS = {
    **dict.fromkeys(["street", "str", "st", "saint", "sreet", "stree"], "st"),
    **dict.fromkeys(["road", "rd"], "rd"),
    **dict.fromkeys(["avenue", "ave", "av", "aven"], "ave"),
    **dict.fromkeys(["boulevard", "blvd", "bd", "boul", "blv"], "blvd"),
    **dict.fromkeys(["drive", "dr", "drv"], "dr"), **dict.fromkeys(["lane", "ln", "lne"], "ln"),
    **dict.fromkeys(["court", "ct", "crt"], "ct"),
    **dict.fromkeys(["circle", "cir", "circ"], "cir"),
    **dict.fromkeys(["highway", "hwy", "hiway"], "hwy"), **dict.fromkeys(["place", "pl"], "pl"),
    **dict.fromkeys(["suite", "ste"], "ste"), **dict.fromkeys(["apartment", "apt"], "apt"),
    **dict.fromkeys(["floor", "fl", "flr"], "fl"),
    **dict.fromkeys(["building", "bldg", "bld"], "bldg"),
    **dict.fromkeys(["north", "n"], "n"), **dict.fromkeys(["south", "s"], "s"),
    **dict.fromkeys(["east", "e"], "e"), **dict.fromkeys(["west", "w"], "w"),
    **dict.fromkeys(["northeast", "ne"], "ne"), **dict.fromkeys(["northwest", "nw"], "nw"),
    **dict.fromkeys(["southeast", "se"], "se"), **dict.fromkeys(["southwest", "sw"], "sw"),
    **dict.fromkeys(["parkway", "pkwy", "pky"], "pkwy"),
    **dict.fromkeys(["terrace", "ter", "terr"], "ter"),
    **dict.fromkeys(["trail", "trl"], "trl"), **dict.fromkeys(["square", "sq"], "sq"),
    **dict.fromkeys(["mount", "mt"], "mt"), **dict.fromkeys(["fort", "ft"], "ft"),
    **dict.fromkeys(["point", "pt"], "pt"), **dict.fromkeys(["route", "rte", "rt"], "rte"),
    **dict.fromkeys(["center", "centre", "ctr"], "ctr"),
    **dict.fromkeys(["heights", "hts"], "hts"),
    **dict.fromkeys(["junction", "jct"], "jct"), **dict.fromkeys(["plaza", "plz"], "plz"),
    **dict.fromkeys(["station", "stn", "sta"], "stn"),
    **dict.fromkeys(["expressway", "expy"], "expy"),
    **dict.fromkeys(["freeway", "fwy"], "fwy"), **dict.fromkeys(["crossing", "xing"], "xing"),
    **dict.fromkeys(["number", "no", "num", "nos", "hno"], "no"),
    **dict.fromkeys(["near", "nr"], "nr"), **dict.fromkeys(["opposite", "opp"], "opp"),
    **dict.fromkeys(["market", "mkt"], "mkt"), **dict.fromkeys(["railway", "rly"], "rly"),
    **dict.fromkeys(["colony", "col"], "col"), **dict.fromkeys(["sector", "sec"], "sec"),
    **dict.fromkeys(["district", "dist", "distt"], "dist"),
    **dict.fromkeys(["rue", "r"], "rue"), **dict.fromkeys(["chemin", "che", "ch"], "chemin"),
    **dict.fromkeys(["impasse", "imp"], "imp"), **dict.fromkeys(["allee", "all"], "allee"),
    **dict.fromkeys(["esplanade", "espl"], "espl"), **dict.fromkeys(["faubourg", "fbg"], "fbg"),
    **dict.fromkeys(["residence", "res"], "res"), **dict.fromkeys(["quai", "q"], "quai"),
    # ordinal words (digits lose their suffix earlier: 8th -> 8)
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12",
    # old and new Indian city names
    "bengaluru": "bangalore", "kolkata": "calcutta", "kolakta": "calcutta",
    "bombay": "mumbai", "madras": "chennai", "gurugram": "gurgaon",
    "trivandrum": "thiruvananthapuram", "visakhapatnam": "vishakhapatnam",
    "vizag": "vishakhapatnam", "pondicherry": "puducherry", "mysuru": "mysore",
    "mangaluru": "mangalore", "baroda": "vadodara", "poona": "pune", "cochin": "kochi",
    "calicut": "kozhikode", "benares": "varanasi", "allahabad": "prayagraj", "simla": "shimla",
    # missing-value markers written into addresses
    "null": "", "none": "", "nan": "", "nil": "",
}

# Regions: every written form (full name, postal code, anyascii of the native-script name)
# -> one canonical code. A code shared by two countries (AR = Arkansas / Arunachal Pradesh,
# CT = Connecticut / Chhattisgarh, OR = Oregon / Odisha) is harmless because true pairs
# always share the country (01 §4); a code must never map to a different code, or two
# spellings of the same state disagree (a later table would overwrite an earlier alias).
_US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "dc": "district of columbia",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho", "il": "illinois",
    "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan",
    "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota",
    "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island",
    "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "pr": "puerto rico",
}
_INDIA_STATES = {
    "ap": ["andhra pradesh", "amdhrprdes", "amdhr prdes"], "ar": ["arunachal pradesh"],
    "as": ["assam"], "br": ["bihar"], "ct": ["chhattisgarh", "chattisgarh", "cg"],
    "ga": ["goa"], "gj": ["gujarat", "gujrat"], "hr": ["haryana", "hriyana"],
    "hp": ["himachal pradesh"], "jh": ["jharkhand"], "ka": ["karnataka", "krnatk"],
    "kl": ["kerala", "kerlm"], "mp": ["madhya pradesh", "mdhy prdes"],
    "mh": ["maharashtra", "mharastr"], "mn": ["manipur"], "ml": ["meghalaya"],
    "mz": ["mizoram"], "nl": ["nagaland"], "or": ["odisha", "orissa", "odisa", "od"],
    "pb": ["punjab", "pmjab"], "rj": ["rajasthan", "rajsthan"], "sk": ["sikkim"],
    "tn": ["tamil nadu", "tamilnadu", "tmilnatu"], "ts": ["telangana", "telmgan", "tg"],
    "tr": ["tripura"], "up": ["uttar pradesh", "uttr prdes"],
    "uk": ["uttarakhand", "uttaranchal", "ua"], "wb": ["west bengal", "pscimbng", "pscim bng"],
    "dl": ["delhi", "dilli", "nct of delhi", "new delhi"], "jk": ["jammu and kashmir"],
    "la": ["ladakh"], "py": ["puducherry", "pondicherry"], "chd": ["chandigarh"],
    "an": ["andaman and nicobar islands"], "ld": ["lakshadweep"],
    "dn": ["dadra and nagar haveli", "daman and diu"],
}
_FRANCE_REGIONS = {
    # departements written in the state slot (test: Nord 151k, Gironde 150k,
    # Loire-Atlantique 127k, Pas-de-Calais 28k components) map to their region's code
    "idf": ["ile de france"], "hdf": ["hauts de france", "nord", "pas de calais"],
    "naq": ["nouvelle aquitaine", "gironde"],
    "pdl": ["pays de la loire", "loire atlantique"], "ara": ["auvergne rhone alpes"],
    "paca": ["provence alpes cote d azur", "provence alpes cote dazur"],
    "occ": ["occitanie"], "ges": ["grand est"], "nor": ["normandie"], "bre": ["bretagne"],
    "bfc": ["bourgogne franche comte"], "cvl": ["centre val de loire"], "cor": ["corse"],
}


def _region_map() -> dict[str, str]:
    """Written region form -> canonical code, over every country's regions."""
    out = {}
    for code, name in _US_STATES.items():
        out[code] = out[name] = code
    for table in (_INDIA_STATES, _FRANCE_REGIONS):
        for code, names in table.items():
            out[code] = code
            out.update(dict.fromkeys(names, code))
    out["new delhi"] = "dl"  # the capital is written as the state in half of the addresses
    return out


REGION_ABBREV = _region_map()
