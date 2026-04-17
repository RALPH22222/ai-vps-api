import json
import random

# --- CONFIGURATION ---
NUM_EXAMPLES = 5000
OUTPUT_FILE = "mock_dataset_v2.json"

# --- WORD BANKS ---
adjectives = ["Advanced", "AI-Driven", "Community-Based", "Automated", "Integrated", "Sustainable", "Smart", "Hybrid", "Cloud-Based", "Real-Time", "Strategic", "Solar-Powered"]
topics = ["Traffic Monitoring", "Rice Field Irrigation", "Disaster Response", "Marine Biodiversity", "Soil Analysis", "Cybersecurity", "Health Data", "Waste Management", "Coral Reef Preservation", "E-Learning", "Public Safety", "Renewable Energy"]
systems = ["System", "Platform", "Framework", "Application", "Module", "Network", "Database", "Dashboard", "Mechanism", "Drone Fleet"]
locations = ["in Zamboanga City", "for Region IX", "in Mindanao", "for Coastal Communities", "in Rural Barangays", "for Urban Centers", "in Public Schools", "for Local Fisherfolk", "in Philippines"]

# "Bad" words that trigger rejection
bad_actions = ["Purchase of", "Procurement of", "Installation of", "General Repair of", "Supply of", "Rental of"]
bad_items = ["Laptops", "Office Chairs", "Air Conditioners", "Gaming PCs", "Vehicles", "CCTV Cameras", "Office Supplies"]

def generate_proposal(i):
    # target a 60% Acceptance Rate for a realistic mix.
    is_good_candidate = random.random() > 0.40

    if is_good_candidate:
        title = f"{random.choice(adjectives)} {random.choice(topics)} {random.choice(systems)} {random.choice(locations)}"
        months = random.randint(12, 36)      # Healthy duration (1-3 years)
        agencies = random.randint(1, 5)      # Healthy agency count
        budget = random.randint(500_000, 5_000_000) # 500k to 5M
        
        status = "ACCEPTED"
        
    else:
        # Randomly choose WHY it fails:
        fail_type = random.choice(["bad_title", "short_duration", "no_agencies", "salary_heavy"])
        
        if fail_type == "bad_title":
            title = f"{random.choice(bad_actions)} {random.choice(bad_items)} {random.choice(locations)}"
            months = random.randint(2, 12)
            agencies = 0
            budget = random.randint(50_000, 200_000)
            
        elif fail_type == "short_duration":
            title = f"Study on {random.choice(topics)}"
            months = random.randint(1, 5) # Too short (< 6 months)
            agencies = 1
            budget = random.randint(100_000, 300_000)
            
        elif fail_type == "no_agencies":
            title = f"{random.choice(adjectives)} {random.choice(topics)} {random.choice(systems)}"
            months = random.randint(12, 24)
            agencies = 0 
            budget = random.randint(500_000, 1_000_000)
            
        else:
            title = f"General {random.choice(topics)} Analysis"
            months = 8
            agencies = 0
            budget = 150_000
            
        status = "REJECTED"

    return {
        "id": f"PROP-{i+1:04d}",
        "title": title,
        "months": months,
        "cooperating_agencies": agencies,
        "budget": budget,
        "expected_result": status
    }

# --- MAIN GENERATION LOOP ---
print(f" Generating {NUM_EXAMPLES} mock proposals...")
dataset = [generate_proposal(i) for i in range(NUM_EXAMPLES)]

# Save to file
with open(OUTPUT_FILE, "w") as f:
    json.dump(dataset, f, indent=2)

print(f" Done! Saved to '{OUTPUT_FILE}'.")
print(f"   Example 1: {dataset[0]}")
print(f"   Example 2: {dataset[1]}")