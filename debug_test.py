from build_subgraphs import *
from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://72.61.201.127:7687', auth=('neo4j', 'neo4j123456'))
result = build_data('What drugs target CYP3A4?', [15450], driver, debug=True)
print('Result:', result)
driver.close()
