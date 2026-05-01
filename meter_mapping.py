#Mapping file
METER_TO_HOUSEHOLD = {
    "M2000": "H1000",
    "M2001": "H1001",
    "M2002": "H1002"
}

METER_TO_PERSON = {
    "M2000": "Amina Njeri",
    "M2001": "Peter Mwangi",
    "M2002": "Grace Wanjiku"
}


def get_household_id(meter_id):
    """Get household ID from meter ID"""
    return METER_TO_HOUSEHOLD.get(meter_id)


def get_person_name(meter_number):
    """Get person name from meter number."""
    return METER_TO_PERSON.get(meter_number, "Unknown Person")


