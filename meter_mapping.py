#Create  mapping file
METER_TO_HOUSEHOLD = {
    "M2000": "H1000",
    "M2001": "H1001",
    "M2002": "H1002"
}
def get_household_id(meter_id):
    """Get household ID from meter ID"""
    return METER_TO_HOUSEHOLD.get(meter_id)


