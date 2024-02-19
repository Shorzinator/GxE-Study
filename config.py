FEATURES = ['PolygenicScoreEXT', 'Age', 'DelinquentPeer', 'SchoolConnect',
            'NeighborConnect', 'ParentalWarmth', 'Sex']

TARGET_1 = "AntisocialTrajectory"
TARGET_2 = "SubstanceUseTrajectory"

FEATURES_FOR_AST_old = FEATURES + [TARGET_2]
FEATURES_FOR_SUT_old = FEATURES + [TARGET_1]

FEATURES_FOR_AST_new = FEATURES + [TARGET_2] + ["Race"]
FEATURES_FOR_SUT_new = FEATURES + [TARGET_1] + ["Race"]
