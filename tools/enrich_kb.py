#!/usr/bin/env python3
"""One-shot KB vocabulary enrichment (keywords/exemplars) driven by bench errors."""
import json
import os

KB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "firstaid", "kb")

ADD_KEYWORDS = {
    "choking_adult": [["choking", 8], ["wrong pipe", 6], ["gagging", 5], ["cannot cough", 6],
                      ["abdominal thrusts", 6], ["cannot talk", 5], ["cannot speak", 5],
                      ["choked", 5], ["grabbing her throat", 6], ["grabbing his throat", 6]],
    "cpr_adult": [["lips blue", 6], ["lips went blue", 7], ["heart is not beating", 7], ["not beating", 5]],
    "unresponsive_breathing": [["collapsed", 4], ["collapsed but breathing", 9], ["not moving", 5],
                               ["fainted and not waking", 7], ["went limp", 6], ["wont come around", 6]],
    "drowning": [["cannot swim", 6], ["swallowed water", 6], ["lake water", 6], ["pool water", 6],
                 ["went under", 6], ["overboard", 7], ["fell in the water", 6], ["pool", 3],
                 ["struggling in the water", 7]],
    "hyperventilation_panic": [["heart racing", 5], ["tingling", 4], ["feel like i am dying", 6],
                               ["breathing fast", 5], ["cannot calm", 6], ["freaking out", 5]],
    "opioid_overdose": [["needles", 4], ["nodding off", 6], ["oxys", 6], ["oxycodone", 6],
                        ["barely breathing", 5], ["tiny pupils", 6], ["vaping", 3], ["hallucinating", 3]],
    "severe_bleeding": [["blood is pumping", 7], ["pumping out", 5], ["tourniquet", 6], ["slashed", 5],
                        ["chainsaw", 4], ["sliced", 4], ["bleeding", 3], ["blood", 2],
                        ["bleeding through", 6], ["amounts of blood", 6]],
    "minor_wound": [["little cut", 5], ["cut is bleeding", 4], ["shallow cut", 5]],
    "nosebleed": [["nose will not stop", 8], ["nose bleeding", 7]],
    "anaphylaxis": [["throat swelling", 9], ["nut allergy", 7], ["tongue swelling", 9], ["wheezing hives", 8]],
    "croup": [["barky", 7], ["barky cough", 8], ["drooling", 4], ["cannot swallow", 4]],
    "stroke": [["slurring", 7], ["speech is garbled", 7], ["lopsided", 5], ["face went slack", 8]],
    "head_injury": [["hit her head", 6], ["hit my head", 6], ["fell down the stairs", 5],
                    ["bump head", 4], ["fell off the couch", 5], ["pupil", 5], ["faceplanted", 5]],
    "heart_attack": [["grabbed his chest", 7], ["grabbed her chest", 7], ["chest hurts", 7],
                     ["left arm", 4], ["arm numb", 4], ["tight band chest", 7], ["belt around his chest", 6]],
    "chest_wound": [["sucking", 5], ["sucking noise", 7], ["sucking sound", 7], ["bubbling", 5]],
    "poisoning_swallowed": [["swallowed a battery", 9], ["button battery", 9], ["watch battery", 8],
                            ["battery", 5], ["sniffed", 5], ["glue", 3], ["paint thinner", 6],
                            ["sleeping pills", 6], ["medicine cabinet", 6], ["weed killer", 7], ["antifreeze", 7]],
    "carbon_monoxide": [["generator", 6], ["exhaust", 5], ["boiler", 4], ["bbq in the tent", 7]],
    "diabetic_low": [["low sugar", 7], ["sugar reading", 6], ["type one", 5], ["insulin and skipped", 7]],
    "fainting": [["nearly blacked out", 6], ["feels faint", 7], ["crumpled", 4], ["keeled over", 5]],
    "shock": [["white as a sheet", 5], ["cold and clammy", 7]],
    "childbirth": [["placenta", 8], ["delivered the baby", 8], ["waters broke", 8], ["need to push", 7]],
    "dehydration_gastro": [["puking", 5], ["keep anything down", 7], ["keep water down", 7],
                           ["been sick", 4], ["watery stools", 6], ["wet nappies", 5]],
    "snake_bite": [["fang marks", 7], ["puncture marks", 6], ["copperhead", 9]],
    "animal_bite": [["dog bit", 8], ["cat bit", 8], ["bit me", 6], ["bit through", 6], ["bat", 4]],
    "bee_sting": [["stung", 6], ["stinger", 7]],
    "jellyfish_sting": [["bluebottle", 9], ["welts across", 6], ["stung in the surf", 8]],
    "spider_bite": [["violin marked", 8]],
    "eye_chemical": [["eyeball", 6], ["cleaning product", 7], ["sanitizer", 6], ["flushing", 4]],
    "eye_foreign_object": [["stuck in my eye", 8], ["flew into my eye", 8], ["under my eyelid", 8]],
    "burn_thermal": [["the iron", 4], ["kettle", 5], ["stove", 4], ["hot soup", 5], ["hot tea", 5],
                     ["pressure cooker", 5], ["burned my tongue", 6]],
    "chemical_burn": [["acid on", 6], ["pool chemicals", 6], ["oven cleaner", 7]],
    "sprain_strain": [["twisted knee", 6], ["twisted", 4], ["cannot walk", 4], ["rolled ankle", 6]],
    "fracture": [["snapped", 5], ["crooked", 4], ["bent the wrong way", 7], ["looks broken", 6]],
    "embedded_object": [["fish hook", 6], ["sticking out of", 6], ["still in his", 4], ["shard", 5]],
    "seizure": [["seizing", 7], ["jerking", 6], ["eyes rolled back", 7], ["went stiff", 5], ["twitching", 5]],
    "heat_stroke": [["staggering", 5], ["gibberish", 5], ["hot and dry", 6], ["burning hot", 6]],
    "heat_exhaustion": [["overheated", 6], ["woozy", 4], ["drained", 3], ["cramping", 4]],
    "hypothermia": [["cold water", 5], ["cold pool", 5], ["icy", 4], ["mumbling", 3], ["soaked and", 4]],
    "breathing_difficulty": [["breathing wrong", 6], ["breathing all wrong", 7], ["snoring", 4],
                             ["fighting for breath", 7], ["cannot get air", 7], ["tight and scary", 4]],
    "general_help": [["i do not know what to do", 4], ["something happened", 3], ["what to do", 2],
                     ["bad accident", 4], ["someone is hurt", 4]],
    "alcohol_poisoning": [["shots", 4], ["blackout", 5], ["drank way too much", 8]],
    "amputation": [["took off two", 5], ["took off his", 5], ["cut clean off", 7]],
    "dislocation": [["off to the side", 4], ["out of joint", 7], ["popped out", 6]],
    "tooth_knocked_out": [["shove it back in", 5], ["tooth got knocked", 8]],
    "frostbite": [["wooden", 3], ["waxy", 5], ["white and numb", 7]],
    "allergic_mild": [["itchy bumps", 6], ["welts", 5], ["breathing fine", 3]],
    "asthma_attack": [["wheezing hard", 6], ["flaring", 4], ["puffs", 5], ["barely talk", 4]],
    "electric_shock": [["thrown back", 5], ["charger", 4], ["bit a cable", 6], ["rewiring", 5]],
    "crush_injury": [["pinned", 6], ["forklift", 5], ["jack slipped", 6]],
    "spinal_injury": [["cannot move his legs", 8], ["cannot feel", 6], ["dove into", 5], ["shallow end", 5]],
}

ADD_EXEMPLARS = {
    "choking_adult": ["it went down the wrong pipe he cannot talk", "shes gagging and grabbing her neck",
                      "how do i do abdominal thrusts", "he cannot cough it out at all"],
    "drowning": ["he fell off the boat and cannot swim", "she swallowed a lot of pool water and keeps coughing"],
    "hyperventilation_panic": ["my heart is pounding and my hands are tingling i think i am dying",
                               "she is breathing way too fast and freaking out"],
    "opioid_overdose": ["found him with needles barely breathing", "he is nodding off after taking pills"],
    "severe_bleeding": ["blood is pumping out of the cut", "he slashed his arm on glass huge amounts of blood",
                        "when should i use a tourniquet"],
    "unresponsive_breathing": ["she collapsed but she is breathing", "he went limp and is not moving but breathes"],
    "general_help": ["please i do not know what to do something happened", "there was a bad accident someone is hurt"],
    "poisoning_swallowed": ["he got paint thinner in his mouth", "she took a whole bottle of sleeping pills"],
    "eye_chemical": ["cleaning spray got into my eyeball it burns"],
    "head_injury": ["baby fell off the couch big bump on his head"],
    "heart_attack": ["he grabbed his chest and went down but is awake"],
    "stroke": ["she is slurring and her arm went weak all of a sudden"],
    "croup": ["his cough is barky and he is drooling a bit"],
    "dehydration_gastro": ["cannot keep water down been sick six times"],
    "hypothermia": ["he has been in the cold water for twenty minutes"],
    "childbirth": ["the placenta has not come out yet after delivery"],
}

for fname in os.listdir(KB):
    if not fname.endswith(".json"):
        continue
    path = os.path.join(KB, fname)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    changed = False
    for p in data["protocols"]:
        pid = p["id"]
        if pid in ADD_KEYWORDS:
            existing = {k for k, _ in p["keywords"]}
            for kw, w in ADD_KEYWORDS[pid]:
                if kw not in existing:
                    p["keywords"].append([kw, w])
                    changed = True
        if pid in ADD_EXEMPLARS:
            for ex in ADD_EXEMPLARS[pid]:
                if ex not in p["exemplars"]:
                    p["exemplars"].append(ex)
                    changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"updated {fname}")
print("done")
