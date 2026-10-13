"""The extraction prompt, one constant per version. AcruxCore stores the versions;
these strings are what scripts/setup_acruxcore.py commits."""

USER = "Receipt text:\n\n{{ transcript }}"

V1_SYSTEM = (
    "You read the text of a store receipt and return three fields as JSON: "
    '"company" (the store name), "date" (as DD/MM/YYYY) and "total" (the total amount, as a '
    "number). Reply with the JSON object only."
)

V2_SYSTEM = (
    "You read the text of a store receipt and return three fields as JSON.\n"
    '- "company": the name of the store that issued the receipt, as printed at the top, '
    "including suffixes such as SDN BHD. Not the customer, not the branch address.\n"
    '- "date": the date of the sale as DD/MM/YYYY. Malaysian receipts print day first. '
    "Ignore print times and due dates.\n"
    '- "total": the amount the customer owes after tax and rounding, as a plain number with '
    'a decimal point and no currency symbol, for example "31.90". It is NOT the subtotal, NOT '
    "the amount excluding GST, and NOT the cash handed over.\n"
    'Reply with one JSON object with exactly the keys "company", "date" and "total", and '
    "nothing else."
)

VERSIONS = {1: V1_SYSTEM, 2: V2_SYSTEM}


def messages(version: int) -> list[dict[str, str]]:
    """Chat messages for a prompt version, with the transcript left as a template variable."""
    return [{"role": "system", "content": VERSIONS[version]}, {"role": "user", "content": USER}]
