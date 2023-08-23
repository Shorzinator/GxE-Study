CROSS_VALIDATOR = "KF"
GRID_SEARCHER = "GCV"
BALANCER = "SMOTE"
IT = "IT"   # Interaction term
FEATURES = ['PolygenicScoreEXT', 'Age', 'DelinquentPeer', 'SchoolConnect',
            'NeighborConnect', 'ParentalWarmth', 'Is_Male']

TARGET_1 = "AST"
TARGET_2 = "SUT"

COMBINED = f"{BALANCER}_{GRID_SEARCHER}_{CROSS_VALIDATOR}"
