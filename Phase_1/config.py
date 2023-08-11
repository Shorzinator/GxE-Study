CROSS_VALIDATOR = "KF"
GRID_SEARCHER = "GCV"
BALANCER = "SMOTE"
IT = "IT"   # Interaction term
FEATURES = ['PolygenicScoreEXT', 'Age', 'DelinquentPeer', 'SchoolConnect',
            'NeighborConnect', 'ParentalWarmth', 'Is_Male', 'Race_1.0', 'Race_2.0',
            'Race_3.0', 'Race_4.0', 'Race_5.0', 'Race_nan']
TARGET_1 = "AST"
TARGET_2 = "SUT"

COMBINED = f"{BALANCER}_{GRID_SEARCHER}_{CROSS_VALIDATOR}"
