#!/usr/bin/env python3
"""
Load triplets from a .pt file into Neo4j database.
Triplets format: [(subject, predicate, object), ...]
"""

import torch
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
from typing import List
from tqdm import tqdm


def load_credentials():
    """Load Neo4j credentials from .env file"""
    load_dotenv('lecture_corpus.env', override=True)
    return {
        'uri': os.getenv('NEO4J_URI'),
        'database': os.getenv('NEO4J_DATABASE'),
        'username': os.getenv('NEO4J_USERNAME'), 
        'password': os.getenv('NEO4J_PASSWORD')
    }


def normalize_entity(entity: str) -> str:
    """Normalize entity names to handle case sensitivity"""
    return entity.strip().lower()


def normalize_predicate(predicate: str) -> str:
    """Remove invalid characters from predicate for Neo4j relationship type"""
    import re
    # Remove invalid characters - Neo4j relationship types can only contain letters, numbers, and underscores
    normalized = re.sub(r'[^A-Za-z0-9_]', '', predicate.strip().upper())
    # Add underscore prefix if it starts with a number
    if normalized and normalized[0].isdigit():
        normalized = '_' + normalized
    return normalized


def chunks(data: List, chunk_size: int = 10000):
    """Split data into chunks for batch processing"""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def load_triplets_to_neo4j(triplets_file: str, credentials: dict, chunk_size: int = 5000):
    """
    Load triplets from .pt file into Neo4j
    
    Args:
        triplets_file: Path to .pt file containing triplets
        credentials: Neo4j connection credentials
        chunk_size: Number of triplets to process in each batch
    """
    
    # Load triplets from .pt file
    print(f"Loading triplets from {triplets_file}...")
    triplets = torch.load(triplets_file)
    print(f"Loaded {len(triplets):,} triplets")
    
    # Normalize and deduplicate
    print("Normalizing and preparing data...")
    unique_entities = set()
    unique_relationships = []
    
    for subject, predicate, obj in tqdm(triplets):
        norm_subject = normalize_entity(subject)
        norm_object = normalize_entity(obj)
        norm_predicate = normalize_predicate(predicate)
        
        unique_entities.add(norm_subject)
        unique_entities.add(norm_object)
        
        unique_relationships.append({
            'source': norm_subject,
            'target': norm_object,
            'relationship_type': norm_predicate,
            'original_predicate': predicate
        })
    
    print(f"Found {len(unique_entities):,} unique entities")
    print(f"Found {len(unique_relationships):,} relationships")
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        credentials['uri'],
        auth=(credentials['username'], credentials['password']),
        database=credentials['database']
    )
    
    try:
        with driver.session() as session:
            # Check if there is existing data
            print(f"There are {session.run('MATCH (n:Entity) RETURN count(n) as node_count').single()['node_count']} nodes in the database")
            print(f"There are {session.run('MATCH ()-[r]->() RETURN count(r) as rel_count').single()['rel_count']} relationships in the database")
            
            # Create constraint for unique entities
            print("Creating constraints...")
            session.run("""
                CREATE CONSTRAINT unique_entity_name IF NOT EXISTS 
                FOR (n:Entity) REQUIRE n.entity_name IS UNIQUE
            """)
            
            # Load entities in batches
            print("Loading entities...")
            entity_records = [{'entity_name': entity} for entity in unique_entities]
            
            total_entities = len(entity_records)
            cumulative_count = 0
            
            for entity_batch in chunks(entity_records, chunk_size):
                result = session.run("""
                    UNWIND $entities AS entity
                    MERGE (n:Entity {entity_name: entity.entity_name})
                    RETURN count(n) as created_count
                """, entities=entity_batch)
                
                count = result.single()['created_count']
                cumulative_count += count
                print(f"Loaded {cumulative_count:,} of {total_entities:,} entities")
            
            # Load relationships in batches, grouped by relationship type
            print("Loading relationships...")
            
            # Group relationships by type for efficient loading
            rel_by_type = {}
            for rel in unique_relationships:
                rel_type = rel['relationship_type']
                if rel_type not in rel_by_type:
                    rel_by_type[rel_type] = []
                rel_by_type[rel_type].append({
                    'source': rel['source'],
                    'target': rel['target']
                })
            
            total_relationships = len(unique_relationships)
            cumulative_count = 0
            
            # Load each relationship type separately
            for rel_type, relationships in rel_by_type.items():
                print(f"Loading {len(relationships):,} relationships of type '{rel_type}'...")
                
                for rel_batch in chunks(relationships, chunk_size):
                    # Use Cypher MERGE to create relationships dynamically
                    query = f"""
                    UNWIND $relationships AS rel
                    MATCH (source:Entity {{entity_name: rel.source}})
                    MATCH (target:Entity {{entity_name: rel.target}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    RETURN count(r) as created_count
                    """
                    
                    result = session.run(query, relationships=rel_batch)
                    count = result.single()['created_count']
                    cumulative_count += count
                    print(f"Loaded {cumulative_count:,} of {total_relationships:,} relationships")
            
            # Print summary statistics
            print("\n=== Loading Complete ===")
            result = session.run("MATCH (n:Entity) RETURN count(n) as node_count")
            node_count = result.single()['node_count']
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = result.single()['rel_count']
            
            print(f"Total nodes loaded: {node_count:,}")
            print(f"Total relationships loaded: {rel_count:,}")
            
    finally:
        driver.close()


if __name__ == "__main__":
    # Configuration
    TRIPLETS_FILE = "techqa/Llama-3.1-Nemotron-Nano-4B-v1.1--20251021_144242--raw_triples.pt"
    CHUNK_SIZE = 5000
    
    # Load credentials
    credentials = load_credentials()
    
    if not all(credentials.values()):
        print("Error: Missing Neo4j credentials in .env file")
        print("Required: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")
        exit(1)
    
    print(f"Connecting to Neo4j at: {credentials['uri']}")
    
    # Load triplets
    load_triplets_to_neo4j(TRIPLETS_FILE, credentials, CHUNK_SIZE)