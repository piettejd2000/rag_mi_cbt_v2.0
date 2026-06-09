#!/usr/bin/env python3

import json
import sqlite3
import os
from pathlib import Path
from collections import defaultdict
import re
from anthropic import Anthropic
from typing import Dict, List, Any, Optional
import time
import sys

# Configuration
DB_PATH = "/Users/johnpiette/healthcare_rl/mi_prototype/tuning/mi_validation_database-4-tables/mi_validation.db"
SCENARIOS_PATH = "/Users/johnpiette/healthcare_rl/mi_prototype/testing/test_transcripts/clinical_scenarios.json"
OUTPUT_DIR = "/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/rag_mi_cbt/chunked_MI_labeled_utterances"
API_KEY_PATH = "/Users/johnpiette/healthcare_rl/mi_prototype/.env"

# Processing configuration
BATCH_SIZE = 50  # Process this many at a time
SAVE_FREQUENCY = 25  # Save after this many processed

# MI behavior code mappings
BEHAVIOR_DESCRIPTIONS = {
    "CR": "Complex Reflection - adds substantial meaning or emphasis",
    "SR": "Simple Reflection - repeats or rephrases",
    "AF": "Affirmation - recognizes strengths or efforts",
    "OQ": "Open Question - invites elaboration",
    "CQ": "Closed Question - seeks specific information",
    "SC": "Seeking Collaboration - emphasizes partnership",
    "AS": "Autonomy Support - emphasizes client's control",
    "PWP": "Persuade with Permission - provides advice with permission",
    "GI": "Giving Information - provides neutral information"
}

# Anti-MI behaviors to exclude
ANTI_MI_BEHAVIORS = ["CO", "PE", "Confront", "Persuade"]

def load_api_key():
    """Load Anthropic API key from .env file"""
    with open(API_KEY_PATH, 'r') as f:
        for line in f:
            if line.startswith('ANTHROPIC_API_KEY'):
                return line.split('=')[1].strip()
    raise ValueError("API key not found in .env file")

def load_clinical_scenarios():
    """Load clinical scenario metadata"""
    with open(SCENARIOS_PATH, 'r') as f:
        scenarios = json.load(f)
    
    # Create lookup by transcript code
    scenario_lookup = {}
    for scenario in scenarios:
        if scenario.get('transcript_code'):
            # Extract number from codes like "Claude_1_10"
            match = re.search(r'(\d+)', scenario['transcript_code'])
            if match:
                scenario_lookup[match.group(1)] = scenario
    return scenario_lookup

def extract_transcript_number(filename):
    """Extract transcript number from filename like '1_Trans_Claude_5.txt'"""
    match = re.match(r'^(\d+)_', filename)
    if match:
        return match.group(1)
    return None

def load_expert_exchanges(limit=None):
    """Load expert-labeled exchanges from database, excluding anti-MI behaviors"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = """
    SELECT 
        id,
        dialogue_turn_id,
        source_transcript,
        client_utterance,
        clinician_utterance,
        expert_technique_codes,
        expert_notes
    FROM validated_exchanges
    WHERE expert_technique_codes IS NOT NULL 
    AND LENGTH(expert_technique_codes) > 0
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    exchanges = []
    
    for row in cursor.fetchall():
        try:
            # Parse expert codes
            codes = json.loads(row[5]) if row[5] else []
            
            # Map full names to abbreviations
            mi_codes = []
            has_anti_mi = False
            
            for code in codes:
                if code in ANTI_MI_BEHAVIORS or code == "Confront" or code == "Persuade":
                    has_anti_mi = True
                    break
                
                # Map to standard abbreviations
                code_map = {
                    "Complex Reflection": "CR",
                    "Simple Reflection": "SR",
                    "Affirm": "AF",
                    "Open Question": "OQ",
                    "Closed Question": "CQ",
                    "Seeking Collaboration": "SC",
                    "Emphasizing Autonomy": "AS",
                    "Persuade with Permission": "PWP",
                    "Giving Information": "GI"
                }
                
                if code in code_map:
                    mi_codes.append(code_map[code])
            
            # Skip if contains anti-MI behaviors or no valid codes
            if has_anti_mi or not mi_codes:
                continue
            
            exchanges.append({
                'id': row[0],
                'dialogue_turn_id': row[1],
                'source_transcript': row[2],
                'client_utterance': row[3],
                'clinician_utterance': row[4],
                'mi_codes': mi_codes,
                'expert_notes': row[6]
            })
            
        except (json.JSONDecodeError, TypeError):
            continue
    
    conn.close()
    return exchanges

def generate_simple_metadata(exchange, scenario):
    """Generate metadata without API call for faster processing"""
    
    # Detect primary issue from utterances
    text = (exchange['client_utterance'] + " " + exchange['clinician_utterance']).lower()
    
    primary_issue = "general_counseling"
    if any(word in text for word in ["medication", "pills", "dose", "prescription"]):
        primary_issue = "medication_adherence"
    elif any(word in text for word in ["diabetes", "blood sugar", "glucose", "insulin"]):
        primary_issue = "diabetes_management"
    elif any(word in text for word in ["alcohol", "drinking", "drunk", "wine", "beer"]):
        primary_issue = "alcohol_use"
    elif any(word in text for word in ["smoke", "cigarette", "tobacco", "nicotine", "vape"]):
        primary_issue = "tobacco_use"
    elif any(word in text for word in ["weight", "diet", "exercise", "eating"]):
        primary_issue = "weight_management"
    elif any(word in text for word in ["marijuana", "cannabis", "weed", "pot"]):
        primary_issue = "cannabis_use"
    elif any(word in text for word in ["stress", "anxiety", "depression", "mental"]):
        primary_issue = "mental_health"
    
    # Detect client state
    client_state = "contemplating"
    if any(word in text for word in ["don't want", "won't", "can't", "refuse"]):
        client_state = "resistant"
    elif any(word in text for word in ["but", "however", "though", "although"]):
        client_state = "ambivalent"
    elif any(word in text for word in ["ready", "want to", "going to", "will"]):
        client_state = "motivated"
    
    # Detect emotional tone
    emotional_tone = "neutral"
    if any(word in text for word in ["scared", "afraid", "worried", "anxious"]):
        emotional_tone = "anxious"
    elif any(word in text for word in ["frustrated", "angry", "annoyed", "irritated"]):
        emotional_tone = "frustrated"
    elif any(word in text for word in ["sad", "depressed", "hopeless", "overwhelmed"]):
        emotional_tone = "overwhelmed"
    elif any(word in text for word in ["hopeful", "confident", "positive", "good"]):
        emotional_tone = "hopeful"
    
    # Generate scenario tags based on MI behaviors
    scenario_tags = []
    if "CR" in exchange['mi_codes'] or "SR" in exchange['mi_codes']:
        scenario_tags.append("reflective_listening")
    if "SC" in exchange['mi_codes']:
        scenario_tags.append("collaborative_planning")
    if "AS" in exchange['mi_codes']:
        scenario_tags.append("supporting_autonomy")
    if "AF" in exchange['mi_codes']:
        scenario_tags.append("building_confidence")
    if "OQ" in exchange['mi_codes']:
        scenario_tags.append("exploring_motivation")
    
    if not scenario_tags:
        scenario_tags = ["therapeutic_conversation"]
    
    # Extract keywords from utterances
    keywords = []
    # Get key phrases from client utterance
    client_words = exchange['client_utterance'].lower().split()
    important_phrases = []
    for i in range(len(client_words) - 2):
        phrase = " ".join(client_words[i:i+3])
        if len(phrase) > 10:
            important_phrases.append(phrase)
    
    keywords = important_phrases[:5] if important_phrases else [exchange['client_utterance'][:50]]
    
    return {
        "primary_issue": primary_issue,
        "client_state": client_state,
        "emotional_tone": emotional_tone,
        "clinical_scenario_tags": scenario_tags,
        "therapeutic_goals": ["build_rapport", "explore_ambivalence"],
        "search_keywords": keywords,
        "barriers_mentioned": [],
        "change_talk_present": "want" in text or "will" in text or "going to" in text,
        "sustain_talk_present": "can't" in text or "won't" in text or "don't" in text
    }

def create_embedding_text(exchange, metadata):
    """Create optimized text for embedding generation"""
    
    scenario_desc = f"Client expressing {metadata['primary_issue']} in {metadata['client_state']} state"
    therapeutic_approach = f"Clinician using {', '.join(exchange['mi_codes'])}"
    
    embedding_text = f"{scenario_desc}. Client: {exchange['client_utterance']}. {therapeutic_approach}. Clinician: {exchange['clinician_utterance']}"
    
    return embedding_text

def create_chunk(exchange, metadata, scenario, chunk_id):
    """Create a structured chunk for the RAG system"""
    
    chunk = {
        "chunk_id": chunk_id,
        "exchange": {
            "client_utterance": exchange['client_utterance'],
            "clinician_response": exchange['clinician_utterance']
        },
        "metadata": {
            "clinical_context": {
                "primary_issue": metadata['primary_issue'],
                "client_state": metadata['client_state'], 
                "emotional_tone": metadata['emotional_tone']
            },
            "mi_behaviors": exchange['mi_codes'],
            "behavior_descriptions": {code: BEHAVIOR_DESCRIPTIONS.get(code, "") for code in exchange['mi_codes']},
            "clinical_scenario_tags": metadata['clinical_scenario_tags'],
            "therapeutic_goals": metadata['therapeutic_goals'],
            "search_keywords": metadata['search_keywords'],
            "quality_indicators": {
                "expert_validated": True,
                "expert_notes": exchange.get('expert_notes', ''),
                "source_transcript": exchange['source_transcript']
            },
            "change_dynamics": {
                "change_talk_present": metadata.get('change_talk_present', False),
                "sustain_talk_present": metadata.get('sustain_talk_present', False),
                "barriers_mentioned": metadata.get('barriers_mentioned', [])
            }
        },
        "embedding_text": create_embedding_text(exchange, metadata),
        "source": "mi_validation_database",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add demographic info if available
    if scenario:
        chunk['metadata']['demographics'] = {
            "age": scenario.get('age'),
            "gender": scenario.get('gender'),
            "condition": scenario.get('condition')
        }
    
    return chunk

def main():
    """Main chunking process - simplified version"""
    
    print("Starting MI RAG chunking process (simplified version)...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading clinical scenarios...")
    scenarios = load_clinical_scenarios()
    
    print("Loading expert-labeled exchanges (excluding anti-MI)...")
    exchanges = load_expert_exchanges()
    print(f"Found {len(exchanges)} exchanges without anti-MI behaviors")
    
    # Process exchanges
    chunks = []
    output_file = os.path.join(OUTPUT_DIR, "mi_labeled_exchanges_chunks.jsonl")
    
    # Open file for incremental writing
    with open(output_file, 'w') as f:
        for i, exchange in enumerate(exchanges):
            if i % 10 == 0:
                print(f"Processing exchange {i+1}/{len(exchanges)}...")
            
            # Extract transcript number
            transcript_num = extract_transcript_number(exchange['source_transcript'])
            scenario = scenarios.get(transcript_num) if transcript_num else None
            
            # Generate metadata (simplified, no API call)
            metadata = generate_simple_metadata(exchange, scenario)
            
            # Create chunk
            chunk_id = f"mi_exchange_{i+1:05d}"
            chunk = create_chunk(exchange, metadata, scenario, chunk_id)
            
            # Write immediately
            f.write(json.dumps(chunk) + '\n')
            chunks.append(chunk)
            
            # Save progress indicator
            if (i + 1) % SAVE_FREQUENCY == 0:
                print(f"  Saved {i+1} chunks...")
    
    print(f"\nChunking complete!")
    print(f"Created {len(chunks)} chunks")
    print(f"Saved to: {output_file}")
    
    # Generate summary statistics
    behavior_counts = defaultdict(int)
    issue_counts = defaultdict(int)
    
    for chunk in chunks:
        for code in chunk['metadata']['mi_behaviors']:
            behavior_counts[code] += 1
        issue_counts[chunk['metadata']['clinical_context']['primary_issue']] += 1
    
    print("\n=== Summary Statistics ===")
    print("\nMI Behaviors in chunks:")
    for behavior, count in sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {behavior}: {count}")
    
    print("\nPrimary issues covered:")
    for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {issue}: {count}")

if __name__ == "__main__":
    main()