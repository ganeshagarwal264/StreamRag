import json
import os
from pathlib import Path

CORPUS = [
    # AI/ML
    {"doc_id": "ai_001", "section": "Transformer Architecture", "text": "Transformers introduced in 2017 Vaswani et al, self-attention mechanisms, input tokens as key-query-value triplets, parallel processing vs RNNs sequential", "metadata": {"domain": "AI", "year": 2017, "source": "synthetic_corpus"}},
    {"doc_id": "ai_002", "section": "Attention Mechanisms", "text": "Multi-head attention allows model to focus on different positions, scaled dot-product attention formula, learned representations for each head", "metadata": {"domain": "AI", "year": 2017, "source": "synthetic_corpus"}},
    {"doc_id": "ai_003", "section": "RAG Overview", "text": "Retrieval-Augmented Generation combines parametric LLM knowledge with non-parametric retrieval, reduces hallucinations, enables knowledge updates without retraining", "metadata": {"domain": "AI", "year": 2020, "source": "synthetic_corpus"}},
    {"doc_id": "ai_004", "section": "RAG Architecture", "text": "RAG has retriever and generator components, retriever fetches k-nearest neighbors from vector DB, generator conditions on retrieved passages", "metadata": {"domain": "AI", "year": 2020, "source": "synthetic_corpus"}},
    {"doc_id": "ai_005", "section": "Large Language Models", "text": "LLMs trained on billions of tokens, GPT-4 released 2023, instruction tuning improves task following, emergent capabilities at scale", "metadata": {"domain": "AI", "year": 2023, "source": "synthetic_corpus"}},
    {"doc_id": "ai_006", "section": "Fine-tuning", "text": "Fine-tuning updates model weights on domain-specific data, LoRA uses low-rank adapters to reduce trainable parameters, PEFT methods enable efficient adaptation", "metadata": {"domain": "AI", "year": 2021, "source": "synthetic_corpus"}},
    {"doc_id": "ai_007", "section": "RLHF", "text": "Reinforcement Learning from Human Feedback uses reward model trained on preference data, PPO algorithm optimizes LLM policy, InstructGPT pioneered this in 2022", "metadata": {"domain": "AI", "year": 2022, "source": "synthetic_corpus"}},
    {"doc_id": "ai_008", "section": "Vector Databases", "text": "Vector databases store high-dimensional embeddings, HNSW algorithm enables approximate nearest-neighbor search at scale, Chroma/Qdrant/Pinecone are popular options", "metadata": {"domain": "AI", "year": 2020, "source": "synthetic_corpus"}},
    {"doc_id": "ai_009", "section": "Prompt Engineering", "text": "Chain-of-thought prompting improves complex reasoning, few-shot examples improve task performance, system prompts guide model behavior", "metadata": {"domain": "AI", "year": 2022, "source": "synthetic_corpus"}},
    {"doc_id": "ai_010", "section": "Hallucination", "text": "LLM hallucination occurs when model generates plausible but false information, RAG reduces hallucination by grounding in retrieved evidence, citation requirements enforce accountability", "metadata": {"domain": "AI", "year": 2023, "source": "synthetic_corpus"}},

    # Climate Science
    {"doc_id": "cl_001", "section": "Greenhouse Effect", "text": "CO2 traps infrared radiation, greenhouse gases include CO2 methane and N2O, current concentration 420ppm vs 280ppm pre-industrial", "metadata": {"domain": "Climate", "year": 2023, "source": "synthetic_corpus"}},
    {"doc_id": "cl_002", "section": "Carbon Cycle", "text": "Oceans absorb 25% of CO2 emissions annually, terrestrial biosphere absorbs another 25%, remaining 50% stays in atmosphere", "metadata": {"domain": "Climate", "year": 2020, "source": "synthetic_corpus"}},
    {"doc_id": "cl_003", "section": "IPCC Reports", "text": "IPCC AR6 (2021) states 1.5C warming likely by 2030s, attributed to human activities with 95% confidence, urgent emissions cuts required", "metadata": {"domain": "Climate", "year": 2021, "source": "synthetic_corpus"}},
    {"doc_id": "cl_004", "section": "Sea Level Rise", "text": "Global mean sea level rising at 3.6mm/year, ice sheet loss accelerating since 2006, coastal flooding events 10x more frequent by 2050", "metadata": {"domain": "Climate", "year": 2022, "source": "synthetic_corpus"}},
    {"doc_id": "cl_005", "section": "Renewable Energy", "text": "Solar costs fell 89% between 2010-2020, wind capacity tripled in same period, renewable energy now cheapest power source in history", "metadata": {"domain": "Climate", "year": 2021, "source": "synthetic_corpus"}},
    {"doc_id": "cl_006", "section": "Carbon Capture", "text": "Direct Air Capture removes CO2 from atmosphere, costs currently $300-1000 per ton, required at gigaton scale by 2050 per net-zero scenarios", "metadata": {"domain": "Climate", "year": 2022, "source": "synthetic_corpus"}},
    {"doc_id": "cl_007", "section": "Tipping Points", "text": "Arctic ice loss, Amazon dieback, and permafrost thaw are climate tipping points, cascading effects if 1.5-2C threshold crossed, non-linear responses", "metadata": {"domain": "Climate", "year": 2023, "source": "synthetic_corpus"}},
    {"doc_id": "cl_008", "section": "Climate Models", "text": "General Circulation Models simulate ocean-atmosphere interactions, CMIP6 ensemble shows 2-5C warming by 2100 under high emissions, uncertainty from clouds", "metadata": {"domain": "Climate", "year": 2021, "source": "synthetic_corpus"}},
    {"doc_id": "cl_009", "section": "Ocean Acidification", "text": "Ocean pH fallen from 8.2 to 8.1 since industrial revolution (30% more acidic), coral reef bleaching events increasing, shellfish calcification impaired", "metadata": {"domain": "Climate", "year": 2020, "source": "synthetic_corpus"}},
    {"doc_id": "cl_010", "section": "Arctic Ice Loss", "text": "Arctic sea ice extent declining at 13% per decade, summer ice-free Arctic possible by 2040, albedo feedback amplifies warming", "metadata": {"domain": "Climate", "year": 2023, "source": "synthetic_corpus"}},

    # Space
    {"doc_id": "sp_001", "section": "Mars Missions", "text": "NASA Perseverance rover landed February 2021, Ingenuity helicopter completed 72 flights, organic molecule detection supports past habitability hypothesis", "metadata": {"domain": "Space", "year": 2024, "source": "synthetic_corpus"}},
    {"doc_id": "sp_002", "section": "James Webb Telescope", "text": "JWST launched December 2021, L2 orbit, 6.5m primary mirror, observes in infrared, first deep field images released July 2022", "metadata": {"domain": "Space", "year": 2022, "source": "synthetic_corpus"}},
    {"doc_id": "sp_003", "section": "Artemis Program", "text": "NASA Artemis aims to return humans to Moon by 2025-2026, first woman and person of color, Gateway lunar station planned, SLS rocket used", "metadata": {"domain": "Space", "year": 2023, "source": "synthetic_corpus"}},
    {"doc_id": "sp_004", "section": "SpaceX Starship", "text": "Starship fully reusable launch vehicle, first integrated flight test 2023, Super Heavy booster 33 Raptor engines, designed for Mars transport", "metadata": {"domain": "Space", "year": 2023, "source": "synthetic_corpus"}},
    {"doc_id": "sp_005", "section": "Exoplanets", "text": "Over 5500 confirmed exoplanets as of 2023, TRAPPIST-1 system has 3 planets in habitable zone, transit photometry and radial velocity main detection methods", "metadata": {"domain": "Space", "year": 2023, "source": "synthetic_corpus"}},
    {"doc_id": "sp_006", "section": "Black Holes", "text": "First black hole image captured by Event Horizon Telescope 2019, M87 galaxy center, Sagittarius A* imaged 2022, accretion disk visible", "metadata": {"domain": "Space", "year": 2022, "source": "synthetic_corpus"}},
    {"doc_id": "sp_007", "section": "Dark Matter", "text": "Dark matter comprises 27% of universe mass-energy, detected only through gravitational effects, WIMP candidates not yet found, axion searches ongoing", "metadata": {"domain": "Space", "year": 2021, "source": "synthetic_corpus"}},
    {"doc_id": "sp_008", "section": "ISS", "text": "International Space Station inhabited continuously since 2000, orbits at 400km altitude, 16 nations involved, planned deorbit by 2030", "metadata": {"domain": "Space", "year": 2023, "source": "synthetic_corpus"}},
    {"doc_id": "sp_009", "section": "Solar Flares", "text": "Solar flares release 10^25 joules, X-class flares can disrupt GPS and power grids, solar cycle 25 more active than predicted, Parker Solar Probe studying sun", "metadata": {"domain": "Space", "year": 2024, "source": "synthetic_corpus"}},
    {"doc_id": "sp_010", "section": "Gravitational Waves", "text": "LIGO detected first gravitational wave 2015 from binary black hole merger, Nobel Prize 2017, LISA space observatory planned for 2030s", "metadata": {"domain": "Space", "year": 2017, "source": "synthetic_corpus"}},

    # Medicine/Biotech
    {"doc_id": "med_001", "section": "mRNA Vaccines", "text": "Pfizer-BioNTech and Moderna COVID vaccines use mRNA technology, lipid nanoparticles deliver mRNA, cells produce spike protein to train immune system", "metadata": {"domain": "Medicine", "year": 2020, "source": "synthetic_corpus"}},
    {"doc_id": "med_002", "section": "mRNA Safety", "text": "mRNA vaccines do not alter DNA, mRNA degrades within days, spike protein expressed for 1-2 weeks, technology development spanned 30+ years before COVID", "metadata": {"domain": "Medicine", "year": 2021, "source": "synthetic_corpus"}},
    {"doc_id": "med_003", "section": "CRISPR", "text": "CRISPR-Cas9 discovered 2012 by Doudna and Charpentier (Nobel 2020), gene editing uses guide RNA to target specific DNA sequences, base editing allows single nucleotide changes", "metadata": {"domain": "Medicine", "year": 2020, "source": "synthetic_corpus"}},
    {"doc_id": "med_004", "section": "CRISPR Applications", "text": "Casgevy first approved CRISPR therapy 2023 for sickle cell disease, in-vivo vs ex-vivo delivery approaches, liver targeting most advanced clinical applications", "metadata": {"domain": "Medicine", "year": 2023, "source": "synthetic_corpus"}},
    {"doc_id": "med_005", "section": "Cancer Immunotherapy", "text": "CAR-T cell therapy engineers patient T-cells to target cancer antigens, checkpoint inhibitors block PD-1/PD-L1, nivolumab approved for melanoma lung cancer", "metadata": {"domain": "Medicine", "year": 2022, "source": "synthetic_corpus"}},
    {"doc_id": "med_006", "section": "Alzheimers", "text": "Lecanemab FDA approved 2023 first disease-modifying Alzheimers drug, clears amyloid plaques, 27% slowing of cognitive decline in trials, early-stage treatment only", "metadata": {"domain": "Medicine", "year": 2023, "source": "synthetic_corpus"}},
    {"doc_id": "med_007", "section": "Antibiotic Resistance", "text": "AMR kills 1.27 million annually per Lancet 2022, WHO critical priority pathogen list, phage therapy and antimicrobial peptides as alternatives", "metadata": {"domain": "Medicine", "year": 2022, "source": "synthetic_corpus"}},
    {"doc_id": "med_008", "section": "Microbiome", "text": "Human gut microbiome contains 38 trillion bacteria, Firmicutes/Bacteroidetes ratio linked to obesity, FMT (fecal transplant) effective for C. difficile", "metadata": {"domain": "Medicine", "year": 2021, "source": "synthetic_corpus"}},
    {"doc_id": "med_009", "section": "Precision Medicine", "text": "Pharmacogenomics tailors drug dosing to genetic variants, liquid biopsy detects circulating tumor DNA, polygenic risk scores predict disease susceptibility", "metadata": {"domain": "Medicine", "year": 2023, "source": "synthetic_corpus"}},
    {"doc_id": "med_010", "section": "Drug Discovery AI", "text": "AlphaFold predicted 200M protein structures, accelerating drug target identification, Insilico Medicine AI-discovered drug entered Phase 2 trials 2023", "metadata": {"domain": "Medicine", "year": 2023, "source": "synthetic_corpus"}},
]

TRANSCRIPTS = [
  {"scenario_id": "sc_01", "name": "Simple Factual", "description": "Single atomic factual question about RAG", "tokens": ["What", "is", "retrieval", "augmented", "generation", "and", "how", "does", "it", "reduce", "hallucinations"], "expected_action": "RETRIEVE", "expected_sub_queries": 1},
  {"scenario_id": "sc_02", "name": "Compound Multi-Intent", "description": "Two distinct questions joined", "tokens": ["How", "do", "transformer", "models", "work", "and", "what", "is", "their", "relationship", "to", "RAG", "systems"], "expected_action": "RETRIEVE", "expected_sub_queries": 2},
  {"scenario_id": "sc_03", "name": "Late Constraint", "description": "Question then adds year filter", "tokens": ["Tell", "me", "about", "renewable", "energy", "progress", "specifically", "after", "2020"], "expected_action": "RETRIEVE", "expected_sub_queries": 1},
  {"scenario_id": "sc_04", "name": "Navigational SUPPRESS", "description": "Presentation control command", "tokens": ["next", "slide", "please"], "expected_action": "SUPPRESS", "expected_sub_queries": 0},
  {"scenario_id": "sc_05", "name": "Incomplete Fragment WAIT", "description": "Trailing conjunction", "tokens": ["And", "also", "what", "about", "the"], "expected_action": "WAIT", "expected_sub_queries": 0},
  {"scenario_id": "sc_06", "name": "Multi-Domain Compound", "description": "Space + medicine cross-domain question", "tokens": ["Compare", "mRNA", "vaccines", "with", "traditional", "vaccines", "and", "also", "explain", "what", "CRISPR", "gene", "editing", "is", "used", "for"], "expected_action": "RETRIEVE", "expected_sub_queries": 2},
  {"scenario_id": "sc_07", "name": "Insufficient Evidence", "description": "Question about quantum computing not in corpus", "tokens": ["Explain", "quantum", "error", "correction", "using", "topological", "qubits"], "expected_action": "RETRIEVE", "expected_sub_queries": 1},
  {"scenario_id": "sc_08", "name": "Filler Heavy WAIT", "description": "Mostly filler words", "tokens": ["um", "uh", "well", "so"], "expected_action": "WAIT", "expected_sub_queries": 0}
]

def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    with open(data_dir / "corpus.json", "w", encoding="utf-8") as f:
        json.dump(CORPUS, f, indent=2)
        
    with open(data_dir / "transcripts.json", "w", encoding="utf-8") as f:
        json.dump(TRANSCRIPTS, f, indent=2)
        
    ai_count = sum(1 for d in CORPUS if d["metadata"]["domain"] == "AI")
    climate_count = sum(1 for d in CORPUS if d["metadata"]["domain"] == "Climate")
    space_count = sum(1 for d in CORPUS if d["metadata"]["domain"] == "Space")
    medicine_count = sum(1 for d in CORPUS if d["metadata"]["domain"] == "Medicine")
    
    print(f"Generated corpus.json: {len(CORPUS)} chunks")
    print(f"  AI/ML:    {ai_count} chunks")
    print(f"  Climate:  {climate_count} chunks")
    print(f"  Space:    {space_count} chunks")
    print(f"  Medicine: {medicine_count} chunks")
    print(f"Generated transcripts.json: {len(TRANSCRIPTS)} scenarios")

if __name__ == "__main__":
    main()
