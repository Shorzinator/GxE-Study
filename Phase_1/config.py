FEATURES = ['PolygenicScoreEXT', 'Age', 'DelinquentPeer', 'SchoolConnect',
            'NeighborConnect', 'ParentalWarmth', 'Is_Male']

TARGET_1 = "AntisocialTrajectory"
TARGET_2 = "SubstanceUseTrajectory"

FEATURES_FOR_AST = FEATURES + [TARGET_2]
FEATURES_FOR_SUT = FEATURES + [TARGET_1]
